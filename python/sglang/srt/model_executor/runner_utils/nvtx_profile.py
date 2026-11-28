"""Optional shape-rich NVTX ranges and model_prof iteration bookkeeping."""

from contextlib import contextmanager, nullcontext
from functools import wraps

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

_ENABLED = envs.SGLANG_PROFILE_NVTX.get()
_prof_iter = None
if _ENABLED:
    try:
        from torch.cuda.nvtx import range_pop, range_push
    except ImportError:
        _ENABLED = False
    if _ENABLED:
        try:
            from model_prof import prof_iter as _prof_iter
        except ImportError:
            pass


@contextmanager
def _range(label):
    range_push(label)
    try:
        yield
    finally:
        range_pop()


def _profile_iteration(iteration):
    if _prof_iter is not None and not get_is_capture_mode():
        # model_prof swaps the top range; keep it within a range we own.
        range_push("model_prof_iteration")
        try:
            _prof_iter(iteration)
        finally:
            range_pop()


def profile_mtp_stage(name, *, draft_decode=False):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not _ENABLED:
                return fn(self, *args, **kwargs)
            profiler = self.target_worker.model_runner.nvtx_profiler
            iteration = profiler.iteration - (not draft_decode)
            with profiler.range(name):
                if get_is_capture_mode() or _prof_iter is None:
                    return fn(self, *args, **kwargs)
                with profiler.range(f"[prof_range]: iter {iteration}"):
                    if draft_decode:
                        _profile_iteration(iteration)
                    return fn(self, *args, **kwargs)

        return wrapped

    return decorate


class RunnerNvtxProfiler:
    def __init__(self, *, is_draft_worker):
        self.is_draft_worker = is_draft_worker
        self.iteration = 0

    def range(self, label):
        return _range(label) if _ENABLED else nullcontext()

    def forward_range(self, forward_batch, forward_pass_id):
        if not _ENABLED:
            return nullcontext()
        prefill_bs = (
            len(forward_batch.extend_seq_lens_cpu)
            if forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.extend_seq_lens_cpu is not None
            else 0
        )
        return self.range(
            f"total bs={forward_batch.batch_size}, P bs={prefill_bs}, "
            f"forward_pass_id={forward_pass_id}"
        )

    def before_forward(self):
        if (
            _ENABLED
            and _prof_iter is not None
            and not self.is_draft_worker
            and not get_is_capture_mode()
        ):
            _profile_iteration(self.iteration)
            self.iteration += 1

    def mode_range(self, mode):
        if not _ENABLED:
            return nullcontext()
        if mode.is_decode():
            name = "forward_decode"
        elif mode.is_idle():
            name = "forward_idle"
        else:
            name = "forward_extend"
        return self.range(name)

    def sample_range(self, logits):
        if not _ENABLED:
            return nullcontext()
        return self.range(f"[FW_NATIVE] op:sample,logits_output:{logits.shape}")
