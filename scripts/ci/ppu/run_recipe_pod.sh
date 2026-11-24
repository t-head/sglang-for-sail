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
        if (( code != 0 )); then
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
            printf '{"rank":%d,"exit_code":%d,"source":"pod_trap"}\n' "$node_rank" "$code" > "$tmp"
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
    NCCL_IB_GID_INDEX=$(bash scripts/ci/ppu/recipe_multinode_gid_index.sh)
    export NCCL_IB_GID_INDEX
    GLOO_SOCKET_IFNAME=$(bash scripts/ci/ppu/recipe_multinode_gloo_iface.sh "$PPU_CI_DIST_INIT_ADDR")
    export GLOO_SOCKET_IFNAME
    # Align MASTER_ADDR/MASTER_PORT with the resolved rendezvous IP so nothing
    # in the runtime or PCCL stack tries to resolve the cluster-internal
    # *.svc.cluster.local hostname injected under hostNetwork, and bind NCCL
    # out-of-band bootstrap to the same peer-facing interface as Gloo.
    export MASTER_ADDR="${PPU_CI_DIST_INIT_ADDR%:*}"
    export MASTER_PORT="${PPU_CI_DIST_INIT_ADDR##*:}"
    export NNODES="$nnodes"
    export RANK="$node_rank"
    export NCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
    export PCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
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
