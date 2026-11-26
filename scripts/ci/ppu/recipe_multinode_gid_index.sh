#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Detect the routable RoCE v2 GID index for NCCL_IB_GID_INDEX across active RDMA devices.
#
# Skips inactive ports when ports/1/state is exposed, requires RoCE v2 when
# ports/1/gid_attrs/types (or gid_attr/types) is exposed, ignores link-local-only
# ports, prefers bonded fabric devices (*bond*) with RFC 4193 ULA (fc00::/7, e.g.
# fd03::/8) over non-ULA global IPv6 or IPv4-mapped management ports (falling back
# cleanly to non-bond active RDMA devices when bond ports have no routable GID),
# and selects the highest routable GID index common to the selected fabric devices
# (with strict-majority consensus when >=3 devices have an outlier rail).
# Fails closed if no routable RoCE v2 GID consensus exists.
set -euo pipefail

hca_list_file=""
while (( $# > 0 )); do
    case "$1" in
        --hca-list-file)
            [[ $# -ge 2 ]] || { echo "gid: --hca-list-file requires a path" >&2; exit 2; }
            hca_list_file="$2"
            shift 2
            ;;
        *)
            echo "gid: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

sysfs="${RECIPE_GID_SYSFS_ROOT:-/sys/class/infiniband}"

if [[ ! -d "$sysfs" ]]; then
    echo "gid: no RDMA devices under $sysfs" >&2
    exit 1
fi

active_devices=()

for device_path in "$sysfs"/*; do
    [[ -d "$device_path" ]] || continue
    [[ -d "$device_path/ports/1/gids" ]] || continue
    state_file="$device_path/ports/1/state"
    if [[ -f "$state_file" ]]; then
        port_state=$(tr '[:upper:]' '[:lower:]' < "$state_file" 2>/dev/null || true)
        case "$port_state" in
            *active*) ;;
            *) continue ;;
        esac
    fi
    active_devices+=("$device_path")
done

if (( ${#active_devices[@]} == 0 )); then
    echo "gid: found no active RDMA device with a GID table under $sysfs" >&2
    exit 1
fi

bond_ula_names=()
bond_ula_sets=()
any_ula_names=()
any_ula_sets=()
bond_v6_names=()
bond_v6_sets=()
any_v6_names=()
any_v6_sets=()
bond_all_names=()
bond_all_sets=()
any_all_names=()
any_all_sets=()

for device_path in "${active_devices[@]}"; do
    device=$(basename -- "$device_path")
    gid_dir="$device_path/ports/1/gids"
    is_bond=0
    case "$device" in
        *bond*) is_bond=1 ;;
    esac

    types_dir=""
    if [[ -d "$device_path/ports/1/gid_attrs/types" ]]; then
        types_dir="$device_path/ports/1/gid_attrs/types"
    elif [[ -d "$device_path/ports/1/gid_attr/types" ]]; then
        types_dir="$device_path/ports/1/gid_attr/types"
    fi

    dev_ula=""
    dev_v6=""
    dev_all=""
    for gid_file in "$gid_dir"/*; do
        [[ -f "$gid_file" ]] || continue
        index=$(basename -- "$gid_file")
        case "$index" in
            '' | *[!0-9]*) continue ;;
        esac
        value=$(tr '[:upper:]' '[:lower:]' < "$gid_file" 2>/dev/null | tr -d '[:space:]' || true)
        case "$value" in
            '' | 0000:0000:0000:0000:0000:0000:0000:0000) continue ;;
            fe80:*) continue ;;
        esac
        if [[ -n "$types_dir" ]]; then
            type_file="$types_dir/$index"
            [[ -f "$type_file" ]] || continue
            gid_type=$(tr '[:upper:]' '[:lower:]' < "$type_file" 2>/dev/null || true)
            case "$gid_type" in
                *"roce v2"*) ;;
                *) continue ;;
            esac
        fi
        norm=$(( 10#$index ))
        dev_all="${dev_all} ${norm}"
        case "$value" in
            0000:0000:0000:0000:0000:ffff:*) ;;
            fc* | fd*)
                dev_v6="${dev_v6} ${norm}"
                dev_ula="${dev_ula} ${norm}"
                ;;
            *)
                dev_v6="${dev_v6} ${norm}"
                ;;
        esac
    done

    if [[ -n "$dev_ula" ]]; then
        any_ula_names+=("$device")
        any_ula_sets+=("$dev_ula")
        if (( is_bond == 1 )); then
            bond_ula_names+=("$device")
            bond_ula_sets+=("$dev_ula")
        fi
    fi
    if [[ -n "$dev_v6" ]]; then
        any_v6_names+=("$device")
        any_v6_sets+=("$dev_v6")
        if (( is_bond == 1 )); then
            bond_v6_names+=("$device")
            bond_v6_sets+=("$dev_v6")
        fi
    fi
    if [[ -n "$dev_all" ]]; then
        any_all_names+=("$device")
        any_all_sets+=("$dev_all")
        if (( is_bond == 1 )); then
            bond_all_names+=("$device")
            bond_all_sets+=("$dev_all")
        fi
    fi
done

if (( ${#bond_ula_sets[@]} > 0 )); then
    selected_names=("${bond_ula_names[@]}")
    selected_sets=("${bond_ula_sets[@]}")
elif (( ${#any_ula_sets[@]} > 0 )); then
    selected_names=("${any_ula_names[@]}")
    selected_sets=("${any_ula_sets[@]}")
elif (( ${#bond_v6_sets[@]} > 0 )); then
    selected_names=("${bond_v6_names[@]}")
    selected_sets=("${bond_v6_sets[@]}")
elif (( ${#any_v6_sets[@]} > 0 )); then
    selected_names=("${any_v6_names[@]}")
    selected_sets=("${any_v6_sets[@]}")
elif (( ${#bond_all_sets[@]} > 0 )); then
    selected_names=("${bond_all_names[@]}")
    selected_sets=("${bond_all_sets[@]}")
elif (( ${#any_all_sets[@]} > 0 )); then
    selected_names=("${any_all_names[@]}")
    selected_sets=("${any_all_sets[@]}")
else
    echo "gid: active RDMA devices expose no routable RoCE v2 GID" >&2
    exit 1
fi

first_device="${selected_names[0]}"
common_indices="${selected_sets[0]}"
strict_ok=1
for (( i = 1; i < ${#selected_sets[@]}; i++ )); do
    device="${selected_names[$i]}"
    dev_indices="${selected_sets[$i]}"
    next_common=""
    for idx in $common_indices; do
        case " ${dev_indices} " in
            *" ${idx} "*) next_common="${next_common} ${idx}" ;;
        esac
    done
    if [[ -z "$next_common" ]]; then
        strict_ok=0
        break
    fi
    common_indices="$next_common"
done

chosen=""
if (( strict_ok == 1 )); then
    for idx in $common_indices; do
        if [[ -z "$chosen" ]] || (( idx > chosen )); then
            chosen="$idx"
        fi
    done
elif (( ${#selected_sets[@]} >= 3 )); then
    # Strict majority fallback when >=3 fabric rails exist and an outlier rail
    # carries its GID at a shifted slot. Two-device ties still fail closed.
    all_candidates=""
    for dev_indices in "${selected_sets[@]}"; do
        for idx in $dev_indices; do
            case " ${all_candidates} " in
                *" ${idx} "*) ;;
                *) all_candidates="${all_candidates} ${idx}" ;;
            esac
        done
    done
    best_count=0
    best_idx=""
    total="${#selected_sets[@]}"
    for idx in $all_candidates; do
        count=0
        for dev_indices in "${selected_sets[@]}"; do
            case " ${dev_indices} " in
                *" ${idx} "*) count=$(( count + 1 )) ;;
            esac
        done
        if (( count * 2 > total && count >= 2 )); then
            if (( count > best_count )) || { (( count == best_count )) && [[ -n "$best_idx" ]] && (( idx > best_idx )); }; then
                best_count="$count"
                best_idx="$idx"
            fi
        fi
    done
    if [[ -n "$best_idx" ]]; then
        chosen="$best_idx"
        echo "gid: using majority RoCE v2 index $chosen ($best_count/$total RDMA devices)" >&2
    fi
fi

if [[ -z "$chosen" ]]; then
    echo "gid: ${device:-$first_device} resolved no common RoCE v2 GID index with $first_device" >&2
    exit 1
fi

matched_hcas=()
for (( i = 0; i < ${#selected_sets[@]}; i++ )); do
    case " ${selected_sets[$i]} " in
        *" ${chosen} "*) matched_hcas+=("${selected_names[$i]}") ;;
    esac
done

if [[ -n "$hca_list_file" ]]; then
    if (( ${#matched_hcas[@]} < ${#active_devices[@]} )); then
        hca_csv=""
        for hca in "${matched_hcas[@]}"; do
            if [[ -n "$hca_csv" ]]; then
                hca_csv="${hca_csv},${hca}"
            else
                hca_csv="=${hca}"
            fi
        done
        printf '%s\n' "$hca_csv" > "$hca_list_file"
    else
        : > "$hca_list_file"
    fi
fi

echo "gid: selected RoCE v2 index $chosen across ${#matched_hcas[@]}/${#active_devices[@]} RDMA devices" >&2
printf '%s\n' "$chosen"
