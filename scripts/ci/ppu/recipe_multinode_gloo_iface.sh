#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derive the non-loopback network interface facing the rendezvous host for GLOO_SOCKET_IFNAME.
#
# Usage: recipe_multinode_gloo_iface.sh <host:port>
set -euo pipefail

address="${1:-${PPU_CI_DIST_INIT_ADDR:-}}"
if [[ -z "$address" ]]; then
    echo "usage: recipe_multinode_gloo_iface.sh <host:port>" >&2
    exit 2
fi

RECIPE_GLOO_TARGET="$address" python3 -I -B - <<'PYEOF'
import fcntl
import os
import socket
import struct
import sys

SIOCGIFADDR = 0x8915

target = os.environ["RECIPE_GLOO_TARGET"]
host, _, port_str = target.rpartition(":")
host = host.strip("[]")
if not host:
    print(f"gloo: {target!r} is not a <host>:<port> address", file=sys.stderr)
    raise SystemExit(1)
try:
    port = int(port_str)
except ValueError:
    port = 29500


def local_address_towards(destination: str, destination_port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((destination, destination_port))
        return sock.getsockname()[0]


def interface_addresses() -> dict[str, str]:
    found: dict[str, str] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                packed = fcntl.ioctl(
                    sock.fileno(),
                    SIOCGIFADDR,
                    struct.pack("256s", name.encode("utf-8")[:15]),
                )
            except OSError:
                continue
            found[name] = socket.inet_ntoa(packed[20:24])
    return found


try:
    source = local_address_towards(host, port)
except OSError as error:
    print(f"gloo: no route from this node to {host}: {error}", file=sys.stderr)
    raise SystemExit(1)

addresses = interface_addresses()
for name, addr in sorted(addresses.items()):
    if addr == source and name != "lo":
        print(f"gloo: binding to {name} ({addr}) towards {host}", file=sys.stderr)
        print(name)
        raise SystemExit(0)

inventory = ", ".join(f"{name}={addr}" for name, addr in sorted(addresses.items()))
print(
    f"gloo: source address towards {host} is {source}, matching no non-loopback interface "
    f"(interfaces: {inventory or 'none'})",
    file=sys.stderr,
)
raise SystemExit(1)
PYEOF
