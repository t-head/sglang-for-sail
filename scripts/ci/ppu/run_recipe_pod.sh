#!/usr/bin/env bash
# The device plugin assigns the requested cards to this isolated worker Pod.
set -euo pipefail
umask 077
root_artifact_dir=${PPU_CI_WORKER_ARTIFACT_DIR:?}
[[ -d "$root_artifact_dir" && ! -L "$root_artifact_dir" &&
    $(stat -c %a -- "$root_artifact_dir") == 700 ]] || exit 2

nnodes=${RECIPE_NNODES:-1}
node_rank=${NODE_RANK:-0}
[[ "$nnodes" =~ ^[1-9][0-9]*$ && "$node_rank" =~ ^[0-9]+$ ]] || exit 2
(( node_rank < nnodes )) || exit 2

ranks_dir=""
barrier_dir=""
if (( nnodes > 1 )); then
    ranks_dir="$root_artifact_dir/ranks"
    barrier_dir="$ranks_dir/barrier"
    mkdir -p "$ranks_dir" "$barrier_dir"
    if (( node_rank == 0 )); then
        export PPU_CI_ARTIFACT_DIR="$root_artifact_dir"
    else
        export PPU_CI_ARTIFACT_DIR="$ranks_dir/rank-$node_rank"
        mkdir -p "$PPU_CI_ARTIFACT_DIR"
    fi
    export PPU_CI_RANKS_DIR="$ranks_dir"
    export PPU_CI_NNODES="$nnodes"
    export PPU_CI_NODE_RANK="$node_rank"
else
    export PPU_CI_ARTIFACT_DIR="$root_artifact_dir"
fi

exec > "$PPU_CI_ARTIFACT_DIR/runtime.log" 2>&1

write_barrier_fallback() {
    local code=$1
    [[ -n "$barrier_dir" && -d "$barrier_dir" ]] || return 0
    local target tmp
    if (( node_rank == 0 )); then
        target="$barrier_dir/rank0.done"
        if [[ ! -f "$target" ]]; then
            tmp="${target}.partial.$$"
            printf '{"status":"%s","exit_code":%d,"source":"pod_trap"}\n' \
                "$( (( code == 0 )) && echo ok || echo error )" "$code" > "$tmp"
            mv -- "$tmp" "$target" || true
        fi
    else
        local propagated=false
        local exit_reason="pod_trap"
        if [[ -f "$barrier_dir/fail-rank-0.json" ]]; then
            propagated=true
            exit_reason="peer_failed"
        elif [[ -f "$barrier_dir/rank0.done" ]] &&
            ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$barrier_dir/rank0.done" 2>/dev/null; then
            propagated=true
            exit_reason="rank0_error"
        else
            local _peer_fail
            for _peer_fail in "$barrier_dir"/fail-rank-*.json; do
                if [[ -f "$_peer_fail" && "$_peer_fail" != "$barrier_dir/fail-rank-${node_rank}.json" ]]; then
                    propagated=true
                    exit_reason="peer_failed"
                    break
                fi
            done
        fi
        if (( code != 0 )) && [[ "$propagated" != true ]]; then
            target="$barrier_dir/fail-rank-${node_rank}.json"
            if [[ ! -f "$target" ]]; then
                tmp="${target}.partial.$$"
                printf '{"rank":%d,"exit_code":%d,"source":"pod_trap"}\n' "$node_rank" "$code" > "$tmp"
                mv -- "$tmp" "$target" || true
            fi
        fi
        target="$barrier_dir/exit-rank-${node_rank}.json"
        if [[ ! -f "$target" ]]; then
            tmp="${target}.partial.$$"
            printf '{"rank":%d,"exit_code":%d,"reason":"%s","source":"pod_trap"}\n' \
                "$node_rank" "$code" "$exit_reason" > "$tmp"
            mv -- "$tmp" "$target" || true
        fi
    fi
    if [[ -n "$ranks_dir" && -d "$ranks_dir" ]]; then
        local status_tmp="$ranks_dir/rank-${node_rank}.status.partial.$$"
        printf '%s\n' "$code" > "$status_tmp"
        mv -- "$status_tmp" "$ranks_dir/rank-${node_rank}.status" || true
    fi
}

# The runtime image may use a different UID than the CPU runner.
# Return ownership before the runner validates and retains private evidence.
finish() {
    code=$?
    trap - EXIT
    set +e
    write_barrier_fallback "$code"
    if [[ "$(stat -c %u -- "$root_artifact_dir")" != "$(id -u)" ]]; then
        if (( nnodes > 1 )); then
            if (( node_rank > 0 )); then
                chown -Rh --reference="$root_artifact_dir" "$PPU_CI_ARTIFACT_DIR" 2>/dev/null || code=1
                for _owned in \
                    "$ranks_dir/rank-${node_rank}.status" \
                    "$barrier_dir/fail-rank-${node_rank}.json" \
                    "$barrier_dir/exit-rank-${node_rank}.json"; do
                    [[ -e "$_owned" ]] && chown -h --reference="$root_artifact_dir" "$_owned" 2>/dev/null || true
                done
            else
                chown -Rh --reference="$root_artifact_dir" "$root_artifact_dir" 2>/dev/null || true
                for _owned in \
                    "$ranks_dir" \
                    "$barrier_dir" \
                    "$ranks_dir/rendezvous" \
                    "$ranks_dir/rank-0.status" \
                    "$barrier_dir/rank0.done"; do
                    [[ -e "$_owned" ]] && chown -h --reference="$root_artifact_dir" "$_owned" 2>/dev/null || true
                done
            fi
        else
            chown -Rh --reference="$root_artifact_dir" "$PPU_CI_ARTIFACT_DIR" || code=1
        fi
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

cd /workspace/source
# Cluster Run preserves the CPU runner's checkout ownership in the archive.
# Trust this one unpacked checkout, then verify its candidate revision.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export PPU_CI_WHEEL_DIR="$PWD/wheels" PPU_CI_RECIPE_SOURCE_DIR="$PWD"
[[ $(git rev-parse HEAD) == "${PPU_CI_SOURCE_SHA:?}" ]]
if [[ "$TIER" == legacy ]]; then
    export SGLANG_TEST_METRICS_FILE="$PPU_CI_ARTIFACT_DIR/suite-metrics.jsonl"
    bash scripts/ci/ppu/build_and_test.sh legacy
    exit
fi
printf '%s\n' wheel_verification > "$PPU_CI_ARTIFACT_DIR/worker-phase.txt"
skip_wheel_install="${PPU_CI_SKIP_WHEEL_INSTALL:-false}"
[[ "$skip_wheel_install" == true || "$skip_wheel_install" == false ]] || exit 2
if [[ "$skip_wheel_install" == true ]]; then
    [[ "${PPU_CI_RECIPE_TREE_SHA256:?}" =~ ^[0-9a-f]{64}$ ]]
    actual_tree=$(python3 -B scripts/ci/ppu/recipe_bundle.py checkout-digest "$PPU_CI_SOURCE_SHA")
    [[ "$actual_tree" == "$PPU_CI_RECIPE_TREE_SHA256" ]]
    cache_key="$actual_tree"
else
    python3 -B scripts/ci/ppu/recipe_bundle.py check "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA"
    cache_key=$(python3 -B scripts/ci/ppu/recipe_bundle.py cache-key "$PPU_CI_WHEEL_DIR" "$PPU_CI_SOURCE_SHA")
fi
[[ -d "${PPU_CI_HF_CACHE_ROOT:?}/hub" && -d /recipe-jit ]]
if (( nnodes > 1 )); then
    [[ "${RECIPE_PPU_COUNT:?}" == 8 ]]
    printf '%s\n' multinode_network > "$PPU_CI_ARTIFACT_DIR/worker-phase.txt"
    PPU_CI_DIST_INIT_ADDR=$(bash scripts/ci/ppu/recipe_multinode_rendezvous.sh "$ranks_dir")
    export PPU_CI_DIST_INIT_ADDR
    hca_list_file="$PPU_CI_ARTIFACT_DIR/roce-hca-list.txt"
    NCCL_IB_GID_INDEX=$(bash scripts/ci/ppu/recipe_multinode_gid_index.sh --hca-list-file "$hca_list_file")
    export NCCL_IB_GID_INDEX
    if [[ -s "$hca_list_file" ]]; then
        detected_hcas=$(tr -d '\r\n' < "$hca_list_file")
        existing_hcas="${NCCL_IB_HCA:-${PCCL_IB_HCA:-}}"
        effective_hcas=$(python3 -I -B - "$detected_hcas" "$existing_hcas" <<'PY'
import os
from pathlib import Path
import re
import sys

detected = [
    item.strip().lstrip("=")
    for item in sys.argv[1].lstrip("=").split(",")
    if item.strip().lstrip("=")
]
existing = sys.argv[2].strip()
if not existing:
    print("=" + ",".join(detected))
    raise SystemExit(0)

invert = existing.startswith("^")
rest = existing[1:] if invert else existing
exact = rest.startswith("=")
body = rest[1:] if exact else rest


def parse_port(raw_port: str) -> int:
    match = re.match(r"[0-9]+", raw_port)
    return int(match.group(0)) if match else 0


entries: list[tuple[str, int | None]] = []
for raw_tok in body.split(","):
    if not raw_tok:
        continue
    prefix, sep, port_str = raw_tok.partition(":")
    if not prefix:
        continue
    entries.append((prefix, parse_port(port_str) if sep else None))


def dev_matches(dev: str, prefix: str) -> bool:
    return dev == prefix if exact else dev.startswith(prefix)


sysfs = Path(os.environ.get("RECIPE_GID_SYSFS_ROOT", "/sys/class/infiniband"))
out_tokens: list[str] = []
for dev in detected:
    matched = [(prefix, port) for prefix, port in entries if dev_matches(dev, prefix)]
    if not invert:
        if not matched:
            continue
        if any(port is None for _, port in matched):
            out_tokens.append(dev)
        else:
            seen_ports: list[int] = []
            for _, port in matched:
                if port is not None and port not in seen_ports:
                    seen_ports.append(port)
                    out_tokens.append(f"{dev}:{port}")
    else:
        if any(port is None for _, port in matched):
            continue
        if not matched:
            out_tokens.append(dev)
            continue
        excluded_ports = {port for _, port in matched if port is not None}
        ports_dir = sysfs / dev / "ports"
        known_ports = (
            sorted(int(p.name) for p in ports_dir.iterdir() if p.name.isdigit())
            if ports_dir.is_dir()
            else [1]
        )
        for port in known_ports or [1]:
            if port not in excluded_ports:
                out_tokens.append(f"{dev}:{port}")

if not out_tokens:
    raise SystemExit(
        "gid: existing NCCL_IB_HCA/PCCL_IB_HCA excludes all RoCE rails supporting the selected GID index"
    )
print("=" + ",".join(out_tokens))
PY
)
        export NCCL_IB_HCA="$effective_hcas"
        export PCCL_IB_HCA="$effective_hcas"
    fi
    GLOO_SOCKET_IFNAME=$(bash scripts/ci/ppu/recipe_multinode_gloo_iface.sh "$PPU_CI_DIST_INIT_ADDR")
    export GLOO_SOCKET_IFNAME
    # Align MASTER_ADDR/MASTER_PORT with the resolved rendezvous IP so nothing
    # in the runtime or PCCL stack tries to resolve the cluster-internal
    # *.svc.cluster.local hostname injected under hostNetwork, and bind NCCL
    # out-of-band bootstrap to the exact peer-facing interface (`=iface` avoids
    # NCCL prefix matching VLAN subinterfaces such as bond0.100).
    export MASTER_ADDR="${PPU_CI_DIST_INIT_ADDR%:*}"
    export MASTER_PORT="${PPU_CI_DIST_INIT_ADDR##*:}"
    export NNODES="$nnodes"
    export RANK="$node_rank"
    export NCCL_SOCKET_IFNAME="=${GLOO_SOCKET_IFNAME}"
    export PCCL_SOCKET_IFNAME="=${GLOO_SOCKET_IFNAME}"
    export PCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"
    # Disable MNNVL and cuMem cross-node fabric auto-enablement (engine.py sets
    # NCCL_MNNVL_ENABLE=1 and NCCL_CUMEM_ENABLE=1 when nnodes>1 on SAIL, while
    # ZW-M890P multi-node runs over RoCE v2 without an IMEX daemon) and disable
    # custom all-reduce v2 multinode. Mirror both PCCL_* and NCCL_* names and
    # enable WARN-level private logs for deterministic diagnostics.
    export PCCL_MNNVL_ENABLE=0
    export NCCL_MNNVL_ENABLE=0
    export PCCL_CUMEM_ENABLE=0
    export NCCL_CUMEM_ENABLE=0
    # Stage cross-node RoCE v2 RDMA through host pinned memory (LOC = disable GDR)
    # to avoid ibv_reg_mr_iova2 EFAULT on device buffers across ZW-M890P nodes.
    export PCCL_NET_GDR_LEVEL=LOC
    export NCCL_NET_GDR_LEVEL=LOC
    export PCCL_DMABUF_ENABLE=0
    export NCCL_DMABUF_ENABLE=0
    export PCCL_DEBUG=WARN
    export NCCL_DEBUG=WARN
    export SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=0
    export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-3600}"
    echo "multinode: rank=${node_rank}/${nnodes} dist_init_addr=${PPU_CI_DIST_INIT_ADDR} gid=${NCCL_IB_GID_INDEX} iface=${GLOO_SOCKET_IFNAME}"
else
    [[ "${RECIPE_PPU_COUNT:?}" =~ ^(2|4|8)$ ]]
fi
if [[ "$TIER" == model_sanity ]] && (( node_rank == 0 )); then
    [[ -d "${PPU_CI_GSM8K_PATH:?}" ]]
    export PPU_CI_EVAL_WHEEL_DIR="$PWD/evalscope-wheels"
fi
[[ "${PPU_CI_RECIPE_IMAGE_KEY:?}" =~ ^[0-9a-f]{64}$ ]]
export PPU_CI_SKIP_WHEEL_INSTALL="$skip_wheel_install"
if (( nnodes > 1 )); then
    export PPU_CI_RECIPE_JIT_DIR="/recipe-jit/sglang-ci/$PPU_CI_RECIPE_IMAGE_KEY/$cache_key/$MODEL_KEY/rank-$node_rank"
else
    export PPU_CI_RECIPE_JIT_DIR="/recipe-jit/sglang-ci/$PPU_CI_RECIPE_IMAGE_KEY/$cache_key/$MODEL_KEY"
fi
bash scripts/ci/ppu/recipe_container.sh "test/ppu/policy/models/$MODEL_KEY.yaml" "$TIER" --queue-s 0
