from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from app.chat_service import ChatService
from app.config import Config
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider
from app.providers.registry import ProviderRegistry
from app.storage import Storage
from app.telegram_api import TelegramBotAPI
from app.telegram_app import TelegramApp
from app.types import ModelSpec


HTTP_CLIENT_LOGGERS = ("httpx", "httpcore")


class StartupError(RuntimeError):
    pass


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for logger_name in HTTP_CLIENT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _provider_models(config: Config, provider_name: str) -> list[ModelSpec]:
    models = list(config.model_catalog.get(provider_name, ()))
    if models:
        return models
    raise RuntimeError(
        f"{provider_name} is configured but no models are defined in {config.model_catalog_path}."
    )


def validate_persisted_model_aliases(
    *,
    storage: Storage,
    registry: ProviderRegistry,
    model_catalog_path: Path,
) -> None:
    missing_aliases = [alias for alias in storage.list_referenced_model_aliases() if registry.resolve(alias) is None]
    if not missing_aliases:
        return
    alias_list = ", ".join(missing_aliases)
    raise RuntimeError(
        "Persisted model aliases are no longer available: "
        f"{alias_list}. Keep alias updates additive in {model_catalog_path} "
        "or migrate stored chat settings and conversations before removing those aliases."
    )


def build_registry(config: Config) -> ProviderRegistry:
    providers = []
    if config.openai_api_key:
        providers.append(OpenAIProvider(config.openai_api_key, _provider_models(config, "openai")))

    if config.gemini_api_key:
        providers.append(GeminiProvider(config.gemini_api_key, _provider_models(config, "gemini")))

    if config.anthropic_api_key:
        providers.append(ClaudeProvider(config.anthropic_api_key, _provider_models(config, "claude")))

    if not providers:
        raise RuntimeError(
            "No providers are configured. Set OPENAI_API_KEY, GEMINI_API_KEY, and/or ANTHROPIC_API_KEY."
        )

    registry = ProviderRegistry(providers)
    if registry.resolve(config.default_model_alias) is None:
        raise RuntimeError(
            f"Default model alias '{config.default_model_alias}' is not available. "
            "Check provider configuration."
        )
    return registry


async def async_main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    configure_logging()

    storage: Storage | None = None
    try:
        config = Config.from_env(root_dir)
        storage = Storage(
            config.db_path,
            default_model_alias=config.default_model_alias,
            default_reply_mode=config.default_reply_mode,
            default_skip_prefix=config.skip_prefix,
        )
        registry = build_registry(config)
        validate_persisted_model_aliases(
            storage=storage,
            registry=registry,
            model_catalog_path=config.model_catalog_path,
        )
    except (RuntimeError, ValueError) as exc:
        if storage is not None:
            storage.close()
        raise StartupError(str(exc)) from exc

    try:
        service = ChatService(
            storage=storage,
            registry=registry,
            system_prompt=config.system_prompt,
            owner_user_ids=config.owner_user_ids,
            conversation_timeout_seconds=config.conversation_timeout_seconds,
            render_limit=config.telegram_message_limit,
            render_edit_interval_seconds=config.render_edit_interval_seconds,
            safety_identifier_salt=config.safety_identifier_salt,
        )

        async with TelegramBotAPI(
            config.telegram_bot_token,
            request_timeout_seconds=config.poll_timeout_seconds + 10,
        ) as api:
            app = TelegramApp(config=config, storage=storage, service=service, api=api)
            await app.run()
    finally:
        storage.close()


def main() -> None:
    try:
        asyncio.run(async_main())
    except StartupError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
