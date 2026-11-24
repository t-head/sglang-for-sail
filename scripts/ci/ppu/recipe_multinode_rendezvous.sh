#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Exchange the multi-node rendezvous "<host>:<port>" over the shared ranks dir.
#
# Usage: recipe_multinode_rendezvous.sh <ranks_dir>
set -euo pipefail

ranks_dir="${1:-}"
if [[ -z "$ranks_dir" ]]; then
    echo "usage: recipe_multinode_rendezvous.sh <ranks_dir>" >&2
    exit 2
fi

node_rank="${PPU_CI_NODE_RANK:-${NODE_RANK:-0}}"
port="${MASTER_PORT:-29500}"
budget="${RECIPE_RENDEZVOUS_TIMEOUT_S:-600}"
poll_interval="${RECIPE_RENDEZVOUS_POLL_S:-5}"
address_file="${ranks_dir}/rendezvous"
barrier_dir="${ranks_dir}/barrier"

if ! [[ "$node_rank" =~ ^[0-9]+$ && "$port" =~ ^[1-9][0-9]*$ && "$port" -le 65535 ]]; then
    echo "rendezvous: invalid node_rank or MASTER_PORT" >&2
    exit 2
fi

if (( node_rank == 0 )); then
    host=$(python3 -I -B - <<'PYEOF'
import os
import socket

host = os.environ.get("SGLANG_HOST_IP") or os.environ.get("HOST_IP") or ""
if not host:
    for probe in (("8.8.8.8", 80), ("10.255.255.255", 1)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(probe)
                candidate = sock.getsockname()[0]
            if candidate and candidate not in ("127.0.0.1", "0.0.0.0"):
                host = candidate
                break
        except OSError:
            continue
if not host:
    try:
        candidate = socket.gethostbyname(socket.gethostname())
    except OSError:
        candidate = ""
    if candidate not in ("", "127.0.0.1", "0.0.0.0"):
        host = candidate
print(host)
PYEOF
    )
    if [[ -z "$host" ]]; then
        echo "rendezvous: rank 0 could not determine its routable IPv4 address" >&2
        exit 1
    fi
    mkdir -p "$ranks_dir"
    tmp_file="${address_file}.partial.$$"
    printf '%s:%s\n' "$host" "$port" > "$tmp_file"
    mv -- "$tmp_file" "$address_file"
    echo "rendezvous: rank 0 published ${host}:${port}" >&2
    printf '%s:%s\n' "$host" "$port"
    exit 0
fi

waited=0
while (( waited < budget )); do
    # Use readdir (ls) to bypass stale NFS negative dentry caches.
    if ls "$ranks_dir" 2>/dev/null | grep -qx 'rendezvous'; then
        address=$(cat "$address_file" 2>/dev/null || true)
        if [[ -n "$address" ]]; then
            echo "rendezvous: node ${node_rank} joined ${address} after ${waited}s" >&2
            printf '%s\n' "$address"
            exit 0
        fi
    fi
    if [[ -d "$barrier_dir" ]] && ls "$barrier_dir" 2>/dev/null | grep -Eq '^(rank0\.done|fail-rank-[0-9]+\.json)$'; then
        echo "rendezvous: node ${node_rank} aborted after ${waited}s because a peer already failed or finished" >&2
        exit 1
    fi
    sleep "$poll_interval"
    waited=$(( waited + poll_interval ))
done

echo "rendezvous: node ${node_rank} waited ${budget}s and rank 0 never published rendezvous" >&2
exit 1
