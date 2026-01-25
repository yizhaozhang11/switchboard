from __future__ import annotations

import asyncio
import io
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    class AsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def aclose(self) -> None:
            return None

    httpx_stub.HTTPError = HTTPError
    httpx_stub.AsyncClient = AsyncClient
    sys.modules["httpx"] = httpx_stub

from app.config import Config
from app.main import StartupError, async_main, build_registry, configure_logging, main, validate_persisted_model_aliases
from app.model_catalog import default_model_catalog
from app.storage import Storage
from app.types import ModelSpec


class MainTests(unittest.TestCase):
    def _make_config(
        self,
        *,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        default_model_alias: str = "o",
        model_catalog: dict[str, tuple[ModelSpec, ...]] | None = None,
    ) -> Config:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        data_dir = Path(tempdir.name)
        return Config(
            telegram_bot_token="token",
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            owner_user_ids=(),
            default_model_alias=default_model_alias,
            default_reply_mode="auto",
            skip_prefix="//",
            conversation_timeout_seconds=300,
            data_dir=data_dir,
            db_path=data_dir / "bot.sqlite3",
            model_catalog_path=data_dir / "models.toml",
            model_catalog=model_catalog or default_model_catalog(),
            system_prompt="system",
        )

    def test_build_registry_adds_gemini_models_when_configured(self) -> None:
        registry = build_registry(self._make_config(gemini_api_key="gemini-key", default_model_alias="g"))
        resolved = registry.resolve("g")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.model.provider, "gemini")
        self.assertEqual(resolved.model.model_id, "gemini-3.1-pro-preview")

    def test_build_registry_adds_openai_models_when_configured(self) -> None:
        registry = build_registry(self._make_config(openai_api_key="openai-key", default_model_alias="o"))
        resolved = registry.resolve("o")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.model.provider, "openai")
        self.assertEqual(resolved.model.model_id, "gpt-5.4")

    def test_build_registry_accepts_default_model_alias_with_search_suffix(self) -> None:
        registry = build_registry(self._make_config(openai_api_key="openai-key", default_model_alias="o-s"))
        selection = registry.resolve_selection("o-s")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.provider, "openai")
        self.assertEqual(selection.requested_tools, ("search",))

    def test_build_registry_uses_model_catalog_from_config(self) -> None:
        model_catalog = default_model_catalog()
        model_catalog["openai"] = (
            ModelSpec(
                provider="openai",
                model_id="gpt-5.5",
                aliases=("o",),
                supports_images=True,
                supports_reasoning=True,
            ),
        )
        registry = build_registry(
            self._make_config(
                openai_api_key="openai-key",
                default_model_alias="o",
                model_catalog=model_catalog,
            )
        )
        resolved = registry.resolve("o")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.model.model_id, "gpt-5.5")

    def test_build_registry_requires_at_least_one_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No providers are configured"):
            build_registry(self._make_config())

    def test_build_registry_adds_claude_models_when_configured(self) -> None:
        registry = build_registry(self._make_config(anthropic_api_key="anthropic-key", default_model_alias="c"))
        resolved = registry.resolve("c")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.model.provider, "claude")
        self.assertEqual(resolved.model.model_id, "claude-opus-4-6")

    def test_async_main_wraps_startup_configuration_errors(self) -> None:
        with mock.patch("app.main.Config.from_env", side_effect=RuntimeError("TELEGRAM_BOT_TOKEN is required")):
            with self.assertRaisesRegex(StartupError, "TELEGRAM_BOT_TOKEN is required"):
                asyncio.run(async_main())

    def test_main_prints_startup_error_without_traceback(self) -> None:
        async def fail_startup() -> None:
            raise StartupError("TELEGRAM_BOT_TOKEN is required")

        stderr = io.StringIO()
        with mock.patch("app.main.async_main", fail_startup):
            with mock.patch("sys.stderr", stderr):
                with self.assertRaises(SystemExit) as context:
                    main()

        self.assertEqual(context.exception.code, 1)
        self.assertEqual(stderr.getvalue(), "Startup error: TELEGRAM_BOT_TOKEN is required\n")

    def test_configure_logging_suppresses_http_client_info_logs(self) -> None:
        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")
        original_httpx_level = httpx_logger.level
        original_httpcore_level = httpcore_logger.level
        self.addCleanup(httpx_logger.setLevel, original_httpx_level)
        self.addCleanup(httpcore_logger.setLevel, original_httpcore_level)

        httpx_logger.setLevel(logging.NOTSET)
        httpcore_logger.setLevel(logging.NOTSET)

        configure_logging()

        self.assertEqual(httpx_logger.level, logging.WARNING)
        self.assertEqual(httpcore_logger.level, logging.WARNING)

    def test_validate_persisted_model_aliases_rejects_removed_aliases(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        data_dir = Path(tempdir.name)
        storage = Storage(
            data_dir / "bot.sqlite3",
            default_model_alias="o",
            default_reply_mode="auto",
            default_skip_prefix="//",
        )
        self.addCleanup(storage.close)
        storage.settings.get_chat_settings(100)
        storage.settings.set_default_model_alias(100, "legacy-chat")
        storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="legacy-conversation")

        model_catalog = default_model_catalog()
        model_catalog["openai"] = (
            ModelSpec(
                provider="openai",
                model_id="gpt-5.5",
                aliases=("new",),
                supports_images=True,
                supports_reasoning=True,
            ),
        )
        registry = build_registry(
            self._make_config(
                openai_api_key="openai-key",
                default_model_alias="new",
                model_catalog=model_catalog,
            )
        )

        with self.assertRaises(RuntimeError) as context:
            validate_persisted_model_aliases(
                storage=storage,
                registry=registry,
                model_catalog_path=data_dir / "models.toml",
            )

        message = str(context.exception)
        self.assertIn("legacy-chat", message)
        self.assertIn("legacy-conversation", message)
        self.assertIn("models.toml", message)

if __name__ == "__main__":
    unittest.main()
