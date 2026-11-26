import hashlib
import json
import sys
from pathlib import Path

import yaml


PD_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "test/ppu/recipes/deepseek-ai/DeepSeek-V4.1-Flash-PD.yaml"
)


def load_pd_config(path=PD_CONFIG_PATH):
    """Load the current local 1P1D case, not a general environment constraint."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or (
        config.get("schema_version") != "ppu-pd-test/v1"
        or config.get("case_id") != "deepseek-v41-flash-1p1d"
    ):
        raise ValueError("Unsupported PD case configuration")
    ppu = config.get("ppu")
    if not isinstance(ppu, dict):
        raise ValueError("PD ppu metadata must be a mapping")
    if ppu.get("framework") != "sglang" or ppu.get("visibility") != "public":
        raise ValueError("The current PD case requires public SGLang metadata")
    identities = [config.get("model_id"), ppu.get("hf_model_id"), ppu.get("golden_model_key")]
    if "served_name" in ppu:
        identities.append(ppu["served_name"])
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("PD model identities must be nonempty strings")
    assets = ppu.get("assets")
    if (
        not isinstance(assets, dict)
        or set(assets) != {"model", "tokenizer"}
        or any(not isinstance(value, str) or not value.strip() for value in assets.values())
    ):
        raise ValueError("PD assets must map model and tokenizer to nonempty strings")
    topology = ppu.get("topology")
    if (
        not isinstance(topology, dict)
        or set(topology) != {"kind", "nodes", "devices_per_node"}
        or topology["kind"] != "pd_disagg"
        or any(type(topology[key]) is not int or topology[key] <= 0
               for key in ("nodes", "devices_per_node"))
    ):
        raise ValueError("Invalid PD topology metadata")
    nodes = config["nodes"]
    roles = config["roles"]
    router = config["router"]
    if set(roles) != {"prefill", "decode"}:
        raise ValueError("The 1P1D case requires one prefill and one decode role")
    if len(nodes) != 1 or any(node["host"] != "127.0.0.1" for node in nodes.values()):
        raise NotImplementedError("The PD test executor does not support remote nodes yet")
    node_name = next(iter(nodes))
    width = nodes[node_name]["devices"]
    if type(width) is not int or width <= 0:
        raise ValueError("The node device count must be a positive integer")
    if topology["nodes"] != len(nodes) or topology["devices_per_node"] != width:
        raise ValueError("PD topology metadata must match node device allocations")
    allocated = []
    ports = [router["port"], router["metrics_port"]]
    for role in roles.values():
        if role["node"] != node_name:
            raise ValueError("A PD role references an unavailable node")
        devices = role["devices"]
        if not devices or any(
            type(index) is not int or not 0 <= index < width for index in devices
        ):
            raise ValueError("Invalid role device indices")
        allocated.extend(devices)
        ports.extend(role[key] for key in ("port", "nccl_port", "bootstrap_port"))
    if sorted(allocated) != list(range(width)):
        raise ValueError("Role device allocations must be distinct and cover the node")
    if router["node"] != node_name:
        raise ValueError("The Router references an unavailable node")
    if any(type(port) is not int or not 0 < port < 65536 for port in ports) or len(
        set(ports)
    ) != len(ports):
        raise ValueError("HTTP, bootstrap and NCCL ports must be valid and distinct")
    for args in (
        config["common_args"],
        *(role["args"] for role in roles.values()),
        router["args"],
    ):
        if not isinstance(args, list) or not all(
            isinstance(arg, str) and arg for arg in args
        ):
            raise ValueError("PD arguments must be lists of nonempty strings")
    for env in (config["common_env"], *(role["env"] for role in roles.values())):
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError("PD environment values must be strings")
    for key in ("ready_timeout_s", "request_timeout_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("PD timeouts must be positive integers")
    memory_limit = config["environment"]["memory_limit_gib"]
    if type(memory_limit) is not int or memory_limit <= 0:
        raise ValueError("PD memory limit must be a positive integer")
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    if sys.argv[1:] != ["device-count", "deepseek-v41-flash-1p1d"]:
        raise SystemExit(2)
    config = load_pd_config()
    print(next(iter(config["nodes"].values()))["devices"])
