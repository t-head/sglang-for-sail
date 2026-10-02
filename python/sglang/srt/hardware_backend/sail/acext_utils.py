"""SAIL ACEXT initialization, isolated from common runtime utilities."""

import logging
import os

from sglang.srt.environ import envs
from sglang.srt.utils.device import get_device_sm

logger = logging.getLogger(__name__)


def set_acext_token_limit(acext_num_tokens: int = 6144):
    if "ACEXT_NUM_TOKENS_LIMIT" not in os.environ:
        os.environ["ACEXT_NUM_TOKENS_LIMIT"] = str(acext_num_tokens)
    logger.warning("acext token limit: %s", os.environ["ACEXT_NUM_TOKENS_LIMIT"])


def check_acext_version_compatibility(acext_version: int = 1050100):
    if get_device_sm() >= 89:
        envs.SGLANG_SAIL_USE_ACEXT_CUDA.set(False)
        return
    from acext import get_version as acext_get_version

    current_ver = acext_get_version()
    if current_ver < acext_version:
        logger.warning(
            "acext version %s is below minimum %s; falling back to Triton",
            current_ver,
            acext_version,
        )
        envs.SGLANG_SAIL_USE_ACEXT_CUDA.set(False)
    else:
        logger.warning("Current acext version: %s", current_ver)
