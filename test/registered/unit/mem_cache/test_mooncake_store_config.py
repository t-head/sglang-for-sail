"""Unit tests for MooncakeStoreConfig local_hostname resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeStoreConfigLocalHostname(CustomTestCase):
    _BASE_EXTRA_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_EXTRA_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_mooncake_env_when_omitted(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.2"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.2")

    def test_load_from_extra_config_uses_local_hostname_env_when_omitted(self):
        with temp_set_env(LOCAL_HOSTNAME="10.0.0.3"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.3")

    def test_load_from_extra_config_prefers_mooncake_env_over_local_hostname(self):
        with temp_set_env(
            MOONCAKE_LOCAL_HOSTNAME="10.0.0.7", LOCAL_HOSTNAME="10.0.0.8"
        ):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.7")

    def test_load_from_extra_config_defaults_to_localhost_without_env(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME=None, LOCAL_HOSTNAME=None):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "localhost")

    def test_load_from_extra_config_uses_explicit_override_without_env(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME=None, LOCAL_HOSTNAME=None):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.9"})
        self.assertEqual(cfg.local_hostname, "10.0.0.9")

    def test_load_from_extra_config_prefers_env_over_broadcast_override(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.4"):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.1"})
        self.assertEqual(cfg.local_hostname, "10.0.0.4")

    def test_load_from_env_uses_mooncake_env(self):
        with temp_set_env(
            MOONCAKE_LOCAL_HOSTNAME="10.0.0.5", MOONCAKE_MASTER="127.0.0.1:50051"
        ):
            cfg = MooncakeStoreConfig.load_from_env()
        self.assertEqual(cfg.local_hostname, "10.0.0.5")

    def test_load_from_env_uses_local_hostname_env(self):
        with temp_set_env(
            LOCAL_HOSTNAME="10.0.0.10", MOONCAKE_MASTER="127.0.0.1:50051"
        ):
            cfg = MooncakeStoreConfig.load_from_env()
        self.assertEqual(cfg.local_hostname, "10.0.0.10")

    def test_from_file_uses_process_env_when_local_hostname_omitted(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as fin:
            json.dump(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "metadata_server": "P2PHANDSHAKE",
                },
                fin,
            )
            config_path = fin.name

        try:
            with temp_set_env(
                allow_sglang=True,
                SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=config_path,
                MOONCAKE_LOCAL_HOSTNAME="10.0.0.6",
            ):
                cfg = MooncakeStoreConfig.from_file()
            self.assertEqual(cfg.local_hostname, "10.0.0.6")
        finally:
            os.unlink(config_path)

    def test_from_file_prefers_env_over_file_local_hostname(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as fin:
            json.dump(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "metadata_server": "P2PHANDSHAKE",
                    "local_hostname": "10.0.0.1",
                },
                fin,
            )
            config_path = fin.name

        try:
            with temp_set_env(
                allow_sglang=True,
                SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=config_path,
                MOONCAKE_LOCAL_HOSTNAME="10.0.0.11",
            ):
                cfg = MooncakeStoreConfig.from_file()
            self.assertEqual(cfg.local_hostname, "10.0.0.11")
        finally:
            os.unlink(config_path)


class TestMooncakeConfigSourceSelection(CustomTestCase):
    def test_standalone_follows_the_selected_source(self):
        """Extra config wins only when it names a server address, else the file,
        else env; no field merge."""
        # (extra_config, file_config, env standalone_storage, expected standalone)
        cases = (
            (
                {"master_server_address": "extra", "standalone_storage": False},
                {"standalone_storage": True},
                True,
                False,
            ),
            (
                {"master_server_address": "extra"},
                {"standalone_storage": True},
                True,
                False,
            ),
            (
                {"client_server_address": "extra", "standalone_storage": True},
                {"standalone_storage": False},
                False,
                True,
            ),
            ({"standalone_storage": False}, {"standalone_storage": True}, False, True),
            ({"standalone_storage": True}, None, False, False),
            (
                {"master_server_address": None, "standalone_storage": False},
                {"standalone_storage": True},
                False,
                True,
            ),
            (None, {}, True, False),
            (None, None, True, True),
            (None, None, False, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mooncake.json")
            for extra, file_config, env_standalone, expected in cases:
                with (
                    self.subTest(extra=extra, file=file_config, env=env_standalone),
                    envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(path),
                    envs.MOONCAKE_MASTER.override("environment"),
                    envs.MOONCAKE_STANDALONE_STORAGE.override(env_standalone),
                ):
                    if file_config is None:
                        envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.clear()
                    else:
                        with open(path, "w") as handle:
                            json.dump(
                                {"master_server_address": "file"} | file_config, handle
                            )
                    self.assertEqual(
                        MooncakeStoreConfig.load(extra).standalone_storage, expected
                    )

    def test_invalid_selected_file_does_not_fall_back(self):
        """An unreadable configured file must raise instead of falling back to env."""
        with (
            tempfile.TemporaryDirectory() as directory,
            envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                os.path.join(directory, "missing.json")
            ),
            envs.MOONCAKE_MASTER.override("environment"),
            envs.MOONCAKE_STANDALONE_STORAGE.override(False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to load config"):
                MooncakeStoreConfig.load({"standalone_storage": False})


if __name__ == "__main__":
    unittest.main()
