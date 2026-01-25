from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.config import Config
from app.model_catalog import DEFAULT_MODEL_CONFIG_BASENAME


class ConfigTests(unittest.TestCase):
    def test_from_env_infers_gemini_default_model_alias_when_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "GEMINI_API_KEY=gemini-key",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.default_model_alias, "g")

    def test_from_env_infers_claude_default_model_alias_when_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "ANTHROPIC_API_KEY=anthropic-key",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.default_model_alias, "c")

    def test_from_env_bootstraps_model_catalog_in_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_DATA_DIR=runtime-data",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)
                self.assertEqual(config.model_catalog_path, root_dir / "runtime-data" / DEFAULT_MODEL_CONFIG_BASENAME)
                self.assertTrue(config.model_catalog_path.is_file())
                self.assertEqual(config.model_catalog["openai"][0].model_id, "gpt-5.4")

    def test_from_env_reads_custom_model_catalog_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_dir = root_dir / "config"
            custom_catalog_dir.mkdir()
            custom_catalog_path = custom_catalog_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.openai.models]]",
                        'model_id = "gpt-5.5"',
                        'aliases = ["o"]',
                        "supports_images = true",
                        "supports_reasoning = true",
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=config/models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)
                self.assertEqual(config.model_catalog_path, custom_catalog_path)
                self.assertEqual(config.model_catalog["openai"][0].model_id, "gpt-5.5")

    def test_from_env_rejects_missing_custom_model_catalog_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            missing_catalog_path = root_dir / "config" / "typo-models.toml"
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=config/typo-models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "BOT_MODEL_CONFIG_PATH does not exist"):
                    Config.from_env(root_dir)

            self.assertFalse(missing_catalog_path.exists())
            self.assertFalse(missing_catalog_path.parent.exists())

    def test_from_env_rejects_boolean_thinking_budget_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.claude.models]]",
                        'model_id = "claude-sonnet-4-6"',
                        'aliases = ["c"]',
                        "supports_reasoning = true",
                        'thinking_mode = "enabled"',
                        "thinking_budget_tokens = true",
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "thinking_budget_tokens must be a positive integer"):
                    Config.from_env(root_dir)

    def test_from_env_rejects_boolean_max_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.claude.models]]",
                        'model_id = "claude-sonnet-4-6"',
                        'aliases = ["c"]',
                        "supports_reasoning = true",
                        "max_output_tokens = true",
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "max_output_tokens must be a positive integer"):
                    Config.from_env(root_dir)

    def test_from_env_rejects_claude_thinking_budget_above_max_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.claude.models]]",
                        'model_id = "claude-sonnet-4-6"',
                        'aliases = ["c"]',
                        "supports_reasoning = true",
                        'thinking_mode = "enabled"',
                        "thinking_budget_tokens = 2048",
                        "max_output_tokens = 1024",
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "thinking_budget_tokens=2048 greater than max_output_tokens=1024"):
                    Config.from_env(root_dir)

    def test_from_env_rejects_invalid_claude_output_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.claude.models]]",
                        'model_id = "claude-sonnet-4-6"',
                        'aliases = ["c"]',
                        "supports_reasoning = true",
                        'output_effort = "turbo"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "output_effort must be one of"):
                    Config.from_env(root_dir)

    def test_from_env_parses_openai_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.openai.models]]",
                        'model_id = "gpt-5.4"',
                        'aliases = ["o"]',
                        "supports_reasoning = true",
                        'reasoning_effort = "high"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.model_catalog["openai"][0].reasoning_effort, "high")

    def test_from_env_parses_openai_minimal_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.openai.models]]",
                        'model_id = "gpt-5.4"',
                        'aliases = ["o"]',
                        "supports_reasoning = true",
                        'reasoning_effort = "minimal"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.model_catalog["openai"][0].reasoning_effort, "minimal")

    def test_from_env_rejects_invalid_openai_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.openai.models]]",
                        'model_id = "gpt-5.4"',
                        'aliases = ["o"]',
                        "supports_reasoning = true",
                        'reasoning_effort = "turbo"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "reasoning_effort must be one of"):
                    Config.from_env(root_dir)

    def test_from_env_rejects_invalid_gemini_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.gemini.models]]",
                        'model_id = "gemini-3.1-pro-preview"',
                        'aliases = ["g"]',
                        "supports_reasoning = true",
                        'reasoning_effort = "xhigh"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "reasoning_effort must be one of"):
                    Config.from_env(root_dir)

    def test_from_env_parses_gemini_zero_thinking_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.gemini.models]]",
                        'model_id = "gemini-3.1-pro-preview"',
                        'aliases = ["g"]',
                        "supports_reasoning = true",
                        "thinking_budget_tokens = 0",
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.model_catalog["gemini"][0].thinking_budget_tokens, 0)

    def test_from_env_rejects_claude_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            custom_catalog_path = root_dir / "models.toml"
            custom_catalog_path.write_text(
                "\n".join(
                    [
                        "[[providers.claude.models]]",
                        'model_id = "claude-sonnet-4-6"',
                        'aliases = ["c"]',
                        "supports_reasoning = true",
                        'reasoning_effort = "high"',
                    ]
                )
            )
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "BOT_MODEL_CONFIG_PATH=models.toml",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "reasoning_effort is not supported"):
                    Config.from_env(root_dir)

    def test_from_env_reads_safety_identifier_salt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "OPENAI_API_KEY=test-openai-key",
                        "SAFETY_IDENTIFIER_SALT=0123456789abcdef",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(root_dir)

        self.assertEqual(config.safety_identifier_salt, "0123456789abcdef")

    def test_from_env_rejects_short_safety_identifier_salt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = Path(tempdir)
            (root_dir / ".env").write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=test-token",
                        "OPENAI_API_KEY=test-openai-key",
                        "SAFETY_IDENTIFIER_SALT=too-short",
                    ]
                )
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "SAFETY_IDENTIFIER_SALT"):
                    Config.from_env(root_dir)


if __name__ == "__main__":
    unittest.main()
