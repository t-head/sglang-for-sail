import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ci/ppu"))
from ppu_pd import PD_CONFIG_PATH, load_pd_config


class TestDeepSeekV41FlashPDPpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux":
            raise RuntimeError("Run this test in a PPU Linux environment")
        config = load_pd_config()
        node = next(iter(config["nodes"].values()))
        environment = config["environment"]
        cls.model_path = Path(os.environ.get("MODEL_PATH", environment["model_path"]))
        tokenizer = cls.model_path / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(f"Mount the real model weights at {cls.model_path}")
        if importlib.util.find_spec("sglang_router") is None:
            raise ImportError("The test environment must provide sglang_router")
        width = node["devices"]
        default_devices = ",".join(str(index) for index in range(width))
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
        devices = [device.strip() for device in devices]
        if len(devices) != width or len(set(devices)) != width or not all(devices):
            raise ValueError(f"Expose exactly {width} PPU devices in CUDA_VISIBLE_DEVICES")
        cls.served_model = os.environ.get("SERVED_MODEL_NAME", str(cls.model_path))
        dummy_weights = os.environ.get("PD_TEST_DUMMY_WEIGHTS", "0")
        if dummy_weights not in ("0", "1"):
            raise ValueError("PD_TEST_DUMMY_WEIGHTS must be 0 or 1")
        cls.dummy_weights = dummy_weights == "1"
        cls.code_source = os.environ.get("PD_TEST_CODE_SOURCE", "source")
        if cls.code_source not in ("source", "installed"):
            raise ValueError("PD_TEST_CODE_SOURCE must be source or installed")
        cls.request_timeout = int(
            os.environ.get("PD_TEST_REQUEST_TIMEOUT", config["request_timeout_s"])
        )
        if cls.request_timeout <= 0:
            raise ValueError("PD_TEST_REQUEST_TIMEOUT must be positive")
        roles = config["roles"]
        router = config["router"]
        for name, role in (*roles.items(), ("router", router)):
            port = int(os.environ.get(f"PD_TEST_{name.upper()}_PORT", role["port"]))
            delta = port - role["port"]
            role["port"] = port
            for key in ("nccl_port", "bootstrap_port", "metrics_port"):
                if key in role:
                    role[key] += delta
        ports = [
            roles["prefill"]["port"],
            roles["decode"]["port"],
            router["port"],
            router["metrics_port"],
            roles["prefill"]["nccl_port"],
            roles["decode"]["nccl_port"],
            roles["prefill"]["bootstrap_port"],
            roles["decode"]["bootstrap_port"],
        ]
        if len(set(ports)) != len(ports) or not all(0 < port < 65536 for port in ports):
            raise ValueError(
                "HTTP, bootstrap and NCCL ports must be valid and distinct"
            )
        for port in ports:
            with socket.socket() as listener:
                listener.bind(("0.0.0.0", port))
        cls.base_url = f"http://{node['host']}:{router['port']}"
        prefill_url = f"http://{node['host']}:{roles['prefill']['port']}"
        decode_url = f"http://{node['host']}:{roles['decode']['port']}"
        artifact_dir = os.environ.get("PD_TEST_ARTIFACT_DIR")
        cls.artifacts = (
            Path(artifact_dir)
            if artifact_dir
            else Path(tempfile.mkdtemp(prefix="deepseek-v41-pd-"))
        )
        cls.artifacts.mkdir(parents=True, exist_ok=True)
        (cls.artifacts / "pd-config.yaml").write_bytes(PD_CONFIG_PATH.read_bytes())
        print(f"PD test logs: {cls.artifacts}", flush=True)
        cls.processes = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handler = signal.signal(sig, cls.interrupt)
            cls.addClassCleanup(signal.signal, sig, previous_handler)
        python_command = [sys.executable]
        python_path = ""
        if cls.code_source == "source":
            repo_python = str(Path(__file__).resolve().parents[3] / "python")
            python_path = os.pathsep.join(
                path for path in (repo_python, os.environ.get("PYTHONPATH")) if path
            )
        else:
            python_command.append("-I")
        ib_device = os.environ.get("IB_DEVICE", environment["ib_device"])
        server_env = {
            **os.environ,
            **config["common_env"],
            "PYTHONPATH": python_path,
            "MODEL_PATH": str(cls.model_path),
            "SERVED_MODEL_NAME": cls.served_model,
            "NCCL_IB_GID_INDEX": os.environ.get(
                "NCCL_IB_GID_INDEX", str(environment["gid_index"])
            ),
        }
        for name, role in roles.items():
            command = [
                *python_command,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(cls.model_path),
                "--served-model-name",
                cls.served_model,
                *config["common_args"],
                *role["args"],
                "--disaggregation-bootstrap-port",
                str(role["bootstrap_port"]),
                "--nccl-port",
                str(role["nccl_port"]),
                "--host",
                role["host"],
                "--port",
                str(role["port"]),
            ]
            if cls.dummy_weights:
                command += ["--load-format", "dummy"]
            if ib_device:
                command += ["--disaggregation-ib-device", ib_device]
            cls.start_process(
                name,
                command,
                {
                    **server_env,
                    **role["env"],
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        devices[index] for index in role["devices"]
                    ),
                    "SERVING_PORT": str(role["port"]),
                },
            )
        deadline = time.monotonic() + int(
            os.environ.get("PD_TEST_READY_TIMEOUT", config["ready_timeout_s"])
        )
        for role, url in (
            ("prefill", prefill_url),
            ("decode", decode_url),
        ):
            cls.wait_ready(role, url, deadline)
        router_args = [
            *python_command,
            "-m",
            "sglang_router.launch_router",
            *router["args"],
            "--prefill",
            prefill_url,
            str(roles["prefill"]["bootstrap_port"]),
            "--decode",
            decode_url,
            "--host",
            router["host"],
            "--port",
            str(router["port"]),
            "--prometheus-host",
            router["metrics_host"],
            "--prometheus-port",
            str(router["metrics_port"]),
            "--request-timeout-secs",
            str(cls.request_timeout),
        ]
        cls.start_process("router", router_args, server_env)
        cls.wait_ready("router", cls.base_url, deadline)

    @classmethod
    def interrupt(cls, signum, frame):
        for process in cls.processes.values():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise InterruptedError(f"PD test interrupted by signal {signum}")

    @classmethod
    def start_process(cls, role, command, env):
        log = (cls.artifacts / f"{role}.log").open("wb")
        cls.addClassCleanup(log.close)
        print(f"Starting {role}: {shlex.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        cls.processes[role] = process
        cls.addClassCleanup(cls.stop_process, process)

    @staticmethod
    def stop_process(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)

    @classmethod
    def check_processes(cls):
        for role, process in cls.processes.items():
            if process.poll() is not None:
                raise RuntimeError(
                    f"{role} exited with {process.returncode}; "
                    f"see {cls.artifacts / (role + '.log')}"
                )

    @classmethod
    def wait_ready(cls, role, url, deadline):
        while time.monotonic() < deadline:
            cls.check_processes()
            try:
                with urlopen(url + "/health", timeout=5) as response:
                    if response.status == 200:
                        cls.check_processes()
                        print(f"{role} is ready", flush=True)
                        return
            except (URLError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(
            f"{role} readiness timed out; see {cls.artifacts / (role + '.log')}"
        )

    def setUp(self):
        previous_handler = signal.signal(signal.SIGALRM, self.request_timed_out)
        self.addCleanup(signal.signal, signal.SIGALRM, previous_handler)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_REAL, self.request_timeout)

    def request_timed_out(self, signum, frame):
        # Killing owned workers also unblocks concurrent HTTP reads.
        for process in self.processes.values():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise TimeoutError(f"PD request exceeded {self.request_timeout} seconds")

    def chat(self, stream=False):
        self.check_processes()
        payload = {
            "model": self.served_model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2 + 3? Reply with only the integer.",
                }
            ],
            "temperature": 0.0,
            "seed": 42,
            "thinking": {"type": "disabled"},
            "max_tokens": 16 if self.dummy_weights else 128,
            "stream": stream,
        }
        return urlopen(
            Request(
                self.base_url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=self.request_timeout,
        )

    def check_completion(self):
        with self.chat() as response:
            self.assertEqual(response.status, 200)
            answer = json.load(response)
        self.assertNotIn("error", answer)
        choice = answer["choices"][0]
        message = choice["message"]
        if self.dummy_weights:
            self.assertTrue(
                any(
                    isinstance(message.get(field), str) and message[field].strip()
                    for field in ("content", "reasoning_content")
                ),
                "Missing non-streaming text",
            )
            self.assertIn(choice["finish_reason"], ("stop", "length"))
        else:
            content = message["content"]
            self.assertIsInstance(content, str)
            self.assertEqual(content.strip(), "5")
            self.assertEqual(choice["finish_reason"], "stop")
        self.check_processes()

    def test_non_streaming(self):
        self.check_completion()

    def test_streaming(self):
        content = []
        text_seen = False
        finished = False
        done = False
        with self.chat(stream=True) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                self.assertNotIn("error", event)
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content.append(delta.get("content") or "")
                    text_seen |= any(
                        isinstance(delta.get(field), str) and bool(delta[field].strip())
                        for field in ("content", "reasoning_content")
                    )
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        if self.dummy_weights:
                            self.assertIn(finish_reason, ("stop", "length"))
                        else:
                            self.assertEqual(finish_reason, "stop")
                        finished = True
        if self.dummy_weights:
            self.assertTrue(text_seen, "Missing stream text")
        else:
            self.assertEqual("".join(content).strip(), "5")
        self.assertTrue(finished, "Missing stream finish reason")
        self.assertTrue(done, "Missing SSE [DONE]")
        self.check_processes()

    def test_concurrent_requests(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.check_completion) for _ in range(4)]
            for future in futures:
                future.result()


if __name__ == "__main__":
    unittest.main()
