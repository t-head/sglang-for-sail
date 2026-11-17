"""Check reproducibility of synthetic Prefill warmup requests."""

import asyncio
import hashlib
import unittest

import numpy as np

from sglang.srt.entrypoints import warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillShapesWarmup(CustomTestCase):
    def test_synthetic_inputs_do_not_depend_on_global_seed(self):
        class Recorder:
            def __init__(self):
                self.requests = []

            async def generate_request(self, request, _):
                ids = np.asarray(request.input_ids, dtype="<u4")
                self.requests.append(
                    (ids.size, hashlib.sha256(ids.tobytes()).hexdigest())
                )
                yield None

        saved_rng = np.random.get_state()
        try:
            for mode in ("null", "prefill"):
                with self.subTest(mode=mode):
                    runs = []
                    for seed in (42, 1337):
                        np.random.seed(seed)
                        expected = np.random.randint(2**16, size=16)
                        np.random.seed(seed)
                        recorder = Recorder()
                        asyncio.run(warmup.prefill_shapes(mode, recorder))
                        self.assertTrue(
                            np.array_equal(np.random.randint(2**16, size=16), expected)
                        )
                        runs.append(recorder.requests)
                    self.assertEqual(len(runs[0]), 18)
                    self.assertEqual(runs[0], runs[1])
        finally:
            np.random.set_state(saved_rng)


if __name__ == "__main__":
    unittest.main()
