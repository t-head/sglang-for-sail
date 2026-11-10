"""Manual, model-free probe of the PD source-buffer drain/abort protocol.

Run with the FIXED SGLang checkout on PYTHONPATH and Mooncake installed:

    python test/manual/disaggregation/test_mooncake_abort_lifetime.py \
        --protocol rdma --source-gpu 0 --destination-gpu 1

The source and destination are separate processes with independently registered
GPU buffers. Optional --protocol tcp explicitly switches BOTH buffers to CPU;
there is no automatic RDMA-to-TCP fallback. --ib-device accepts the wrapper's
ordinary HCA name or per-GPU JSON mapping. The same-node RDMA configuration does
not establish which physical link Mooncake selects internally.

This is fault injection at the transport-call boundary: a lease is admitted,
then a thread Event pauses immediately before the real synchronous native write
and again after its successful return, before production chunk completion. It
checks the production admission tracker, Mooncake ABORT/ACK/completion methods,
and decode generation ACK accounting over real ZMQ. It does NOT exercise a full
Scheduler/model/cache allocator, measure online incidence, or prove that a
native timeout/error cancels RDMA. Native failure deliberately never releases
the lease. A supervisor bounds the entire process group, including native hangs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from types import SimpleNamespace


def _configure(args):
    os.environ["MOONCAKE_PROTOCOL"] = args.protocol
    if args.protocol == "tcp":
        os.environ["MC_FORCE_TCP"] = "1"
    else:
        os.environ.pop("MC_FORCE_TCP", None)
    os.environ["SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT"] = str(
        max(1, int(args.timeout / 3))
    )


def _recv(pipe, timeout, expected):
    if not pipe.poll(timeout):
        raise TimeoutError(f"Timed out waiting for {expected}")
    message = pipe.recv()
    if message.get("event") != expected:
        raise RuntimeError(f"Expected {expected}, received {message}")
    return message


def _tensor(args, gpu, marker):
    import torch

    if args.protocol == "rdma":
        torch.cuda.set_device(gpu)
        value = torch.full(
            (args.bytes,), marker, dtype=torch.uint8, device=f"cuda:{gpu}"
        )
        torch.cuda.synchronize(gpu)
    else:
        value = torch.full((args.bytes,), marker, dtype=torch.uint8)
    return value


def _source(args, pipe):
    _configure(args)
    import zmq

    from sglang.srt.disaggregation.base import KVPoll
    from sglang.srt.disaggregation.common.transfer_lifetime import (
        TransferLifetimeTracker,
    )
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )

    source = _tensor(args, args.source_gpu, 0xA5)
    engine = MooncakeTransferEngine(args.hostname, args.source_gpu, args.ib_device)
    engine.register(source.data_ptr(), source.numel())
    context = zmq.Context()
    abort_socket = context.socket(zmq.PULL)
    abort_socket.setsockopt(zmq.LINGER, 0)
    port = abort_socket.bind_to_random_port("tcp://127.0.0.1")
    # Construct only the production control/ownership component; no model pools
    # or scheduler threads are needed for these methods.
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._transfer_lifetime = TransferLifetimeTracker()
    manager._abort_drain_lock = threading.RLock()
    manager.enable_deferred_decode_kv_release = True
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = defaultdict(int)
    manager.request_status = {args.room: KVPoll.Transferring}
    manager._zmq_ctx = context
    manager._socket_cache = {}
    manager._monitor_cache = {}
    manager._socket_send_locks = {}
    manager._socket_lock = threading.Lock()
    manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
    manager.pp_size = manager.attn_cp_size = 1
    assert manager._transfer_lifetime.open(args.room)
    pipe.send(
        {
            "event": "ready",
            "abort_port": port,
            "source_session": engine.get_session_id(),
        }
    )
    target = _recv(pipe, args.timeout, "start")
    assert manager._transfer_lifetime.try_acquire(args.room)
    before = threading.Event()
    allow_write = threading.Event()
    returned = threading.Event()
    allow_complete = threading.Event()
    completed = threading.Event()
    outcome = {}
    chunk = SimpleNamespace(room=args.room, lease_held=True)

    def write():
        try:
            if args.protocol == "rdma":
                import torch

                torch.cuda.set_device(args.source_gpu)
            before.set()
            if not allow_write.wait(args.timeout):
                raise TimeoutError("Pre-write injection gate was not released")
            outcome["rc"] = engine.transfer_sync(
                target["session"],
                source.data_ptr(),
                target["destination"],
                source.numel(),
            )
            returned.set()
            if not allow_complete.wait(args.timeout):
                raise TimeoutError("Post-write injection gate was not released")
            if outcome["rc"] != 0:
                raise RuntimeError(
                    f"Native write failed: {outcome['rc']}; lease retained"
                )
            # What transfer_worker does after a chunk's I/O has ended.
            manager._release_transfer_lease(chunk)
            manager._maybe_ack_drained_abort(chunk.room)
            completed.set()
        except BaseException:
            outcome["error"] = traceback.format_exc()
            returned.set()

    worker = threading.Thread(target=write, daemon=True)
    worker.start()
    if not before.wait(args.timeout):
        raise TimeoutError("Worker did not reach pre-write gate")
    pipe.send({"event": "before_write"})
    return_reported = completion_reported = False
    try:
        while True:
            if abort_socket.poll(10):
                manager._handle_abort_notification(abort_socket.recv_multipart())
                closed = manager._transfer_lifetime.is_closed(args.room)
                rejected = not manager._transfer_lifetime.try_acquire(args.room)
                pipe.send(
                    {
                        "event": "cancelled",
                        "closed": closed,
                        "new_admission_rejected": rejected,
                        "drained": manager._transfer_lifetime.is_drained(args.room),
                    }
                )
            if returned.is_set() and not return_reported:
                pipe.send({"event": "native_returned", **outcome})
                return_reported = True
            if completed.is_set() and not completion_reported:
                pipe.send(
                    {
                        "event": "completed",
                        "drained": manager._transfer_lifetime.is_drained(args.room),
                    }
                )
                completion_reported = True
            if pipe.poll():
                command = pipe.recv()["event"]
                if command == "release_write":
                    allow_write.set()
                elif command == "release_complete":
                    allow_complete.set()
                elif command == "stop":
                    break
                else:
                    raise ValueError(command)
    finally:
        allow_write.set()
        allow_complete.set()
        worker.join(min(args.timeout, 5))
        # Never deregister while a native call still owns the source. The outer
        # process-group supervisor handles an unresponsive native operation.
        if worker.is_alive():
            raise RuntimeError("Native write still running; buffer remains registered")
        engine.deregister(source.data_ptr())
        for socket in (
            *manager._socket_cache.values(),
            *manager._monitor_cache.values(),
        ):
            socket.close(linger=0)
        abort_socket.close(linger=0)
        context.term()


def _probe(args):
    _configure(args)
    import torch
    import zmq

    from sglang.srt.disaggregation.common.conn import (
        AbortAck,
        AbortNotification,
        CommonKVManager,
    )
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )
    from sglang.srt.utils.network import get_local_ip_auto

    if args.hostname is None:
        args.hostname = get_local_ip_auto()
    destination = _tensor(args, args.destination_gpu, 0x11)
    engine = MooncakeTransferEngine(args.hostname, args.destination_gpu, args.ib_device)
    engine.register(destination.data_ptr(), destination.numel())
    manager = CommonKVManager.__new__(CommonKVManager)
    manager.requires_transfer_drain = True
    manager._deferred_abort_ack_tracker = {}
    manager._deferred_abort_generation = 0
    generation = manager.arm_deferred_abort_room(args.room, None)
    context = zmq.Context()
    ack = context.socket(zmq.PULL)
    ack.setsockopt(zmq.LINGER, 0)
    ack_port = ack.bind_to_random_port("tcp://127.0.0.1")
    abort = context.socket(zmq.PUSH)
    abort.setsockopt(zmq.LINGER, 0)
    abort.setsockopt(zmq.SNDTIMEO, int(args.timeout * 1000))
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_source, args=(args, child))
    process.start()
    child.close()
    events = []

    def observe(event, **data):
        record = {"event": event, "time_monotonic": time.monotonic(), **data}
        events.append(record)
        print(json.dumps(record), flush=True)

    def verify_marker(marker):
        if destination.is_cuda:
            torch.cuda.synchronize(args.destination_gpu)
        values = destination.cpu()
        assert bool(torch.all(values == marker)), (
            f"Destination does not contain 0x{marker:02x}"
        )

    def assert_held(stage):
        assert not ack.poll(args.quiet_ms), f"Premature abort ACK at {stage}"
        assert not manager.is_abort_release_safe(args.room, 1, generation=generation), (
            f"Premature reuse at {stage}"
        )
        observe(stage, ack=False, reuse_allowed=False)

    try:
        ready = _recv(parent, args.timeout, "ready")
        abort.connect(f"tcp://127.0.0.1:{ready['abort_port']}")
        parent.send(
            {
                "event": "start",
                "session": engine.get_session_id(),
                "destination": destination.data_ptr(),
            }
        )
        _recv(parent, args.timeout, "before_write")
        message = AbortNotification(
            room=args.room,
            decode_ip="127.0.0.1",
            decode_port=ack_port,
            generation=generation,
        ).to_zmq()
        abort.send_multipart(message)
        cancelled = _recv(parent, args.timeout, "cancelled")
        assert cancelled["closed"] and cancelled["new_admission_rejected"]
        assert not cancelled["drained"]
        assert_held("cancelled_before_native_write")
        verify_marker(0x11)

        parent.send({"event": "release_write"})
        written = _recv(parent, args.timeout, "native_returned")
        assert "error" not in written and written["rc"] == 0, written
        verify_marker(0xA5)
        assert_held("native_write_returned_before_chunk_completion")
        parent.send({"event": "release_complete"})
        assert ack.poll(int(args.timeout * 1000)), (
            "No ACK after production chunk completion"
        )
        parts = ack.recv_multipart()
        assert parts == AbortAck(args.room, 0, generation).to_zmq(), parts
        manager.note_abort_ack(args.room, 0, generation)
        assert manager.is_abort_release_safe(args.room, 1, generation=generation)
        assert _recv(parent, args.timeout, "completed")["drained"]
        observe("drained_ack_received", reuse_allowed=True)

        destination.fill_(0x5A)
        if destination.is_cuda:
            torch.cuda.synchronize(args.destination_gpu)
        # Duplicate cancellation must not reopen producer admission or lose the
        # already recorded ACK; it must leave B's reused buffer intact.
        abort.send_multipart(message)
        duplicate = _recv(parent, args.timeout, "cancelled")
        assert duplicate["new_admission_rejected"] and duplicate["drained"]
        assert ack.poll(int(args.timeout * 1000)), (
            "Duplicate abort was not acknowledged"
        )
        assert ack.recv_multipart() == parts
        time.sleep(args.quiet_ms / 1000)
        verify_marker(0x5A)
        observe("reused_as_B", marker="0x5a", unchanged=True)
    finally:
        if process.is_alive():
            parent.send({"event": "stop"})
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        # Destination registration survives until the source process exits.
        engine.deregister(destination.data_ptr())
        abort.close(linger=0)
        ack.close(linger=0)
        context.term()
        parent.close()
    if process.exitcode != 0:
        raise RuntimeError(f"Source process failed with exit code {process.exitcode}")
    print(
        json.dumps(
            {
                "result": "PASS",
                "protocol": args.protocol,
                "buffers": "GPU"
                if args.protocol == "rdma"
                else "CPU (explicit fallback)",
                "source_gpu": args.source_gpu,
                "destination_gpu": args.destination_gpu,
                "bytes": args.bytes,
                "boundary_fault_injection": True,
                "native_error_cancellation_proven": False,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("rdma", "tcp"), default="rdma")
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--destination-gpu", type=int, default=1)
    parser.add_argument("--ib-device")
    parser.add_argument("--hostname")
    parser.add_argument("--bytes", type=int, default=1024 * 1024)
    parser.add_argument("--room", type=int, default=734921)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--quiet-ms", type=int, default=200)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.bytes <= 0 or args.timeout <= 0 or not 1 <= args.quiet_ms <= 5000:
        parser.error("bytes/timeout must be positive; quiet-ms must be in [1, 5000]")
    if args.protocol == "rdma" and args.source_gpu == args.destination_gpu:
        parser.error("RDMA GPU probe requires two different visible GPU indices")
    if args._worker:
        _probe(args)
        return
    # Hard deadline covers native initialization/transfer/deregistration and
    # kills only this probe's isolated process group on timeout.
    command = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--_worker"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        raise SystemExit(process.wait(timeout=args.timeout))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise SystemExit(
            "Probe timed out/interrupted; its process group was terminated"
        )


if __name__ == "__main__":
    main()
