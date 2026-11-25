import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenExpContextParallelRows(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.arange(8).reshape(4, 2)
        self.positions = torch.arange(4)
        self.batch = SimpleNamespace(mm_input_embeds=None)
        self.model = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=False),
            has_ple=False,
            start_layer=0,
            end_layer=0,
        )
        self.enterContext(
            patch.object(qwen4_exp, "_shards_rows_over_cp", return_value=True)
        )
        self.shard = self.enterContext(
            patch.object(qwen4_exp, "cp_shard_hidden_states", side_effect=self._shard)
        )
        self.shard_positions = self.enterContext(
            patch.object(qwen4_exp, "cp_shard_position_ids", side_effect=self._shard)
        )
        residual = self.enterContext(patch.object(qwen4_exp, "residual_batch"))
        residual.to_pp.side_effect = lambda rows, batch: {"hidden_states": rows}

    @staticmethod
    def _shard(rows, batch):
        return rows[::2]

    def forward(self, **kwargs):
        return qwen4_exp.Qwen4ExpModel.forward(
            self.model, self.positions, self.positions, self.batch, **kwargs
        )["hidden_states"]

    def test_later_pipeline_stage_keeps_proxy_rows(self):
        self.model.pp_group.is_first_rank = False
        local_rows = self._shard(self.hidden, self.batch)
        from_pp = Mock(return_value=local_rows)
        self.model.layers = [
            SimpleNamespace(attn_boundary=SimpleNamespace(from_pp=from_pp))
        ]
        proxy = SimpleNamespace(tensors={"hidden_states": local_rows})

        result = self.forward(pp_proxy_tensors=proxy)

        from_pp.assert_called_once_with(proxy, self.batch)
        self.shard.assert_not_called()
        self.shard_positions.assert_called_once_with(self.positions, self.batch)
        self.assertIs(result, local_rows)


if __name__ == "__main__":
    unittest.main()
