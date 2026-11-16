"""Detection contract for the Step-5 tokenizer reload.

Step-5 ships a byte-level BPE ``tokenizer.json`` but declares the Llama
tokenizer class. Transformers v5 rebuilds the BPE from that class and drops the
space prefix, so ``" capital"`` encodes as ``"capital"``. ``get_tokenizer`` and
``get_processor`` therefore load Step-5 checkpoints straight from the
serialized tokenizer, which first requires recognising the checkpoint.

That recognition is what the cases below pin down: it alone decides which
loader runs, and it can be checked without a checkpoint. The reload itself
needs a real Step-5 ``tokenizer.json``; verify it on a device by encoding
``" capital"`` and asserting the first token still carries the leading space,
e.g. ``tokenizer.tokenize(" capital")[0].startswith("Ġ")`` for byte-level BPE.

Run standalone::

    python test/ppu/unit/test_step5_tokenizer_detect.py
"""

import json
import sys
import tempfile
from pathlib import Path

from sglang.srt.hf_transformers.tokenizer import _is_step5_checkpoint


def _write_config(directory: str, text: str) -> str:
    (Path(directory) / "config.json").write_text(text)
    return directory


def run_detect_step5_checkpoint():
    """``model_type`` on either the config or its text sub-config marks Step-5.

    ``startswith`` is what makes this work for every shipped spelling:
    ``step5``, ``step5_mtp`` and ``step5_vl``.
    """
    cases = [
        ("text model", {"model_type": "step5"}, True),
        ("mtp model", {"model_type": "step5_mtp"}, True),
        (
            "vl top level",
            {"model_type": "step5_vl", "text_config": {"model_type": "step5"}},
            True,
        ),
        # A wrapper config hiding the real model type in its text sub-config.
        (
            "vl text config only",
            {"model_type": "step3_vl", "text_config": {"model_type": "step5"}},
            True,
        ),
        (
            "other model",
            {"model_type": "qwen3_vl", "text_config": {"model_type": "qwen3"}},
            False,
        ),
        ("empty model type", {"model_type": ""}, False),
        (
            "text config without model type",
            {"model_type": "step3_vl", "text_config": {}},
            False,
        ),
    ]
    for tag, config, expected in cases:
        with tempfile.TemporaryDirectory() as tmp:
            got = _is_step5_checkpoint(_write_config(tmp, json.dumps(config)), None)
            assert got is expected, f"{tag}: expected {expected}, got {got}"


def run_unreadable_config_is_not_step5():
    """An absent or malformed config must leave every other loader in charge."""
    with tempfile.TemporaryDirectory() as tmp:
        assert _is_step5_checkpoint(tmp, None) is False, "missing config.json"
    with tempfile.TemporaryDirectory() as tmp:
        _write_config(tmp, "{not json")
        assert _is_step5_checkpoint(tmp, None) is False, "malformed config.json"


def main():
    run_detect_step5_checkpoint()
    run_unreadable_config_is_not_step5()
    print("all Step5 tokenizer detection cases passed")


if __name__ == "__main__":
    sys.exit(main())
