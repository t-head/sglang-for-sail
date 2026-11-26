"""Validate the PPU runner environment without importing SGLang."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path


class CheckFailure(RuntimeError):
    """A failed PPU environment contract check."""


def require(condition: bool, message: str) -> None:
    """Raise a concise contract failure when a condition is false."""
    if not condition:
        raise CheckFailure(message)


def print_host_summary() -> None:
    """Print capacity information without hostnames or network identifiers."""
    disk_usage = shutil.disk_usage("/")

    print(f"Runner OS: {platform.system()}")
    print(f"Runner architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")

    # Memory reporting is diagnostic-only; it must not fail the preflight.
    try:
        import psutil

        print(f"Memory total (bytes): {psutil.virtual_memory().total}")
    except Exception:
        print("Memory total (bytes): unavailable")
    print(f"Root disk total (bytes): {disk_usage.total}")
    print(f"Root disk available (bytes): {disk_usage.free}")


def check_ppu_inventory() -> int:
    """Return the number of PPUs reported by the management interface."""
    ppu_smi = shutil.which("ppu-smi")
    require(ppu_smi is not None, "ppu-smi is not available on PATH")

    try:
        result = subprocess.run(
            [ppu_smi, "--query-ppu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckFailure(
            f"ppu-smi could not query the PPU inventory: {error}"
        ) from error

    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = f" (rc={result.returncode})"
        if stderr:
            detail += f": {stderr}"
        raise CheckFailure(f"ppu-smi failed to query the PPU inventory{detail}")

    inventory_lines = result.stdout.splitlines()
    indices = [line.strip() for line in inventory_lines if line.strip()]
    require(indices, "ppu-smi did not report any PPU devices")
    require(
        all(re.fullmatch(r"[0-9]+", index) for index in indices),
        "ppu-smi returned an unexpected inventory format",
    )

    print(f"PPU devices reported: {len(indices)}")
    return len(indices)


def check_device_nodes() -> None:
    """Check the character devices used by the current PPU runtime."""
    for device_node in (Path("/dev/alixpu_ctl"), Path("/dev/alixpu")):
        try:
            is_character_device = stat.S_ISCHR(device_node.stat().st_mode)
        except OSError as error:
            raise CheckFailure(
                f"Required PPU character device is unavailable: {device_node}"
            ) from error
        require(
            is_character_device,
            f"Required PPU device is not a character device: {device_node}",
        )

    print("Required PPU device nodes are present.")


def check_ppu_sdk() -> None:
    """Verify that PPU_SDK names an installed SDK directory."""
    ppu_sdk = os.environ.get("PPU_SDK", "")
    require(ppu_sdk, "PPU_SDK is not set")
    require(
        Path(ppu_sdk).is_dir(),
        f"PPU_SDK must name an existing directory, got {ppu_sdk!r}",
    )
    print(f"PPU SDK directory is available: {ppu_sdk}")


def check_torch_compute() -> int:
    """Run a minimal PPU computation using PyTorch's CUDA-compatible API."""
    try:
        import torch
    except Exception as error:
        raise CheckFailure(f"PyTorch cannot be imported: {error}") from error

    require(
        torch.cuda.is_available(),
        "torch.cuda.is_available() returned false; PPU is reached through the "
        "CUDA-compatible device type, so this means no PPU is usable",
    )
    device_count = torch.cuda.device_count()
    require(
        device_count > 0,
        "torch.cuda.device_count() did not report a device",
    )

    # Device-name lookup is optional and must not block the compute check.
    try:
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception:
        print("torch.cuda.get_device_name(0): unavailable")

    device = torch.device("cuda:0")
    try:
        source = torch.tensor([1.0, 2.0, 3.0], device=device)
        actual = source * 2.0 + 1.0
        torch.cuda.synchronize(device)
        actual_cpu = actual.cpu()
        expected = torch.tensor([3.0, 5.0, 7.0])
        torch.testing.assert_close(actual_cpu, expected)
    except Exception as error:
        raise CheckFailure(f"Tensor compute on cuda:0 failed: {error}") from error

    print(f"PyTorch version: {torch.__version__}")
    print(f"PPU devices visible through torch.cuda: {device_count}")
    print("Tensor compute on cuda:0 passed.")
    return device_count


def check_visibility_consistency(reported: int, visible: int) -> None:
    """Require visible devices not to exceed the ppu-smi inventory.

    Device visibility restrictions may expose fewer devices to PyTorch.
    """
    require(
        visible <= reported,
        f"torch reports {visible} device(s) but ppu-smi reports only {reported}",
    )
    if visible < reported:
        print(
            f"Note: {visible} of {reported} PPU(s) visible to this job "
            f"(device visibility is restricted per runner job)."
        )
    else:
        print(f"ppu-smi and torch agree on {reported} device(s).")


def check_memory_limit(max_bytes, cgroup=Path("/sys/fs/cgroup"), meminfo=Path("/proc/meminfo")):
    limit = (cgroup / "memory.max").read_text().strip()
    require(
        limit.isdecimal() and 0 < int(limit) <= max_bytes,
        "Set a container memory limit within the PD case limit",
    )
    swap_disabled = (cgroup / "memory.swap.max").read_text().strip() == "0"
    if not swap_disabled:
        swap_disabled = bool(
            re.search(r"^SwapTotal:\s+0\s+kB$", meminfo.read_text(), re.MULTILINE)
        )
    require(swap_disabled, "Disable swap for the PD test container or host")
    return int(limit)


def check_model(model_path):
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        require((model_path / name).is_file(), f"Missing model file: {name}")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    require(shards, "Model index contains no weight shards")
    for name in shards:
        require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts,
            "Invalid model weight shard path",
        )
        path = model_path / name
        require(
            path.is_file() and path.stat().st_size > 0,
            f"Missing or empty shard: {name}",
        )
    return len(shards)


def check_rdma(devices, gid_index, sysfs=Path("/sys/class"), dev=Path("/dev/infiniband")):
    names = devices.split(",")
    require(
        all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) for name in names)
        and len(names) == len(set(names)),
        "IB_DEVICE must contain distinct RDMA device names",
    )
    require(gid_index >= 0, "NCCL_IB_GID_INDEX must not be negative")
    verbs = {
        path.read_text().strip(): dev / path.parent.name
        for path in (sysfs / "infiniband_verbs").glob("uverbs*/ibdev")
    }
    for name in names:
        active = False
        for port in (sysfs / "infiniband" / name / "ports").glob("*"):
            if not (port / "state").read_text().startswith("4:"):
                continue
            gid = (port / "gids" / str(gid_index)).read_text().strip()
            if gid.replace(":", "").strip("0"):
                active = True
        require(active, "RDMA device has no active port with the selected GID")
        require(name in verbs, "No verbs device is exposed for an RDMA device")
        descriptor = os.open(verbs[name], os.O_RDWR)
        os.close(descriptor)
    return names


def check_pd_devices(expected_count):
    import torch

    require(
        torch.cuda.device_count() == expected_count,
        f"The PD case requires {expected_count} visible PPUs",
    )
    for index in range(expected_count):
        value = torch.ones(1, device=f"cuda:{index}") + 1
        require(value.item() == 2, f"PPU {index} compute failed")


def check_pd_runtime():
    root = Path(__file__).resolve().parents[3]
    code_source = os.environ["PD_TEST_CODE_SOURCE"]
    require(code_source in ("source", "installed"), "Invalid PD_TEST_CODE_SOURCE")
    if code_source == "source":
        sys.path.insert(0, str(root / "python"))
    import sglang
    from sglang.srt.platforms import current_platform

    require(current_platform.is_sail(), "PD must select the SAIL platform")
    origin = Path(sglang.__file__).resolve()
    require(
        origin.is_relative_to(root / "python") == (code_source == "source"),
        "SGLang import does not match PD_TEST_CODE_SOURCE",
    )
    for name in (
        "sgl_kernel", "deep_gemm", "flash_mla", "mooncake.engine", "sglang_router"
    ):
        importlib.import_module(name)
    router = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "sglang_router.launch_router", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    for option in ("--mini-lb", "--pd-disaggregation", "--prefill", "--decode"):
        require(option in router.stdout, f"Installed Router does not support {option}")
    versions = {}
    for name in (
        "sglang", "sglang-kernel", "sgl-kernel", "torch", "sglang-router", "deep-gemm"
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    test = root / "test/ppu/e2e/test_deepseek_v41_flash_pd_ppu.py"
    return {
        "code_source": code_source,
        "sglang_import": str(origin),
        "versions": versions,
        "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        "jit_dir": os.environ["PPU_CI_RECIPE_JIT_DIR"],
    }


def main() -> int:
    """Run every check and return a process-compatible status code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd-case", choices=("deepseek-v41-flash-1p1d",))
    parser.add_argument("--pd-config", type=Path)
    args = parser.parse_args()
    if args.pd_config is not None and args.pd_case is None:
        parser.error("--pd-config requires --pd-case")
    try:
        print_host_summary()
        if args.pd_case:
            require(
                os.environ.get("PD_TEST_DUMMY_WEIGHTS", "0") == "0",
                "PD requires real weights",
            )
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from ppu_pd import config_digest, load_pd_config

            config = load_pd_config(args.pd_config) if args.pd_config else load_pd_config()
            environment = config["environment"]
            memory_limit = check_memory_limit(environment["memory_limit_gib"] * 1024**3)
            shards = check_model(Path(os.environ["MODEL_PATH"]))
            rdma = check_rdma(
                os.environ.get("IB_DEVICE", environment["ib_device"]),
                int(os.environ.get("NCCL_IB_GID_INDEX", environment["gid_index"])),
            )
        reported = check_ppu_inventory()
        check_device_nodes()
        check_ppu_sdk()
        visible = check_torch_compute()
        check_visibility_consistency(reported, visible)
        if args.pd_case:
            check_pd_devices(next(iter(config["nodes"].values()))["devices"])
            runtime = {
                **check_pd_runtime(),
                "case": args.pd_case,
                "config_digest": config_digest(config),
                "memory_limit_bytes": memory_limit,
                "weight_shards": shards,
                "rdma_devices": rdma,
            }
            output = Path(os.environ["PD_TEST_ARTIFACT_DIR"]) / "runtime.json"
            output.write_text(json.dumps(runtime, sort_keys=True) + "\n")
    except Exception:
        traceback.print_exc()
        return 1

    print("PPU runner environment preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
