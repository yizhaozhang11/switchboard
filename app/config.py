from __future__ import annotations

from dataclasses import dataclass

from app.model_catalog import default_model_catalog, ModelCatalog
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider
from app.providers.registry import ProviderRegistry
from app.types import ModelSpec

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a Telegram chat. "
    "Answer clearly and briefly, keep context local to the active conversation, "
    "and avoid assuming you should answer messages that explicitly opt out."
)

VALID_REPLY_MODES = {"auto", "mention", "off"}


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str | None
    gemini_api_key: str | None
    anthropic_api_key: str | None
    owner_user_ids: tuple[int, ...]
    default_model_alias: str
    default_reply_mode: str
    skip_prefix: str
    conversation_timeout_seconds: int
    system_prompt: str
    safety_identifier_salt: str | None
    telegram_message_limit: int
    render_edit_interval_seconds: float
    model_catalog: ModelCatalog


def parse_owner_ids(raw_value: str) -> tuple[int, ...]:
    if not raw_value.strip():
        return ()
    owner_ids = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        owner_ids.append(int(item))
    return tuple(sorted(set(owner_ids)))


def infer_default_model_alias(
    *,
    openai_api_key: str | None,
    gemini_api_key: str | None,
    anthropic_api_key: str | None,
    model_catalog: ModelCatalog,
) -> str:
    provider_keys = (
        ("openai", openai_api_key),
        ("gemini", gemini_api_key),
        ("claude", anthropic_api_key),
    )
    for provider_name, api_key in provider_keys:
        if not api_key:
            continue
        for model in model_catalog.get(provider_name, ()):
            if model.aliases:
                return model.aliases[0]
    return "o"


def _provider_models(model_catalog: ModelCatalog, provider_name: str) -> list[ModelSpec]:
    models = list(model_catalog.get(provider_name, ()))
    if models:
        return models
    raise RuntimeError(
        f"{provider_name} is configured but no models are defined in the model catalog."
    )


def build_config() -> Config:
    """Build Config from Workers environment bindings."""
    from workers import env

    telegram_bot_token = str(env.TELEGRAM_BOT_TOKEN).strip()
    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    default_reply_mode = str(getattr(env, "BOT_DEFAULT_REPLY_MODE", "auto") or "auto").strip() or "auto"
    if default_reply_mode not in VALID_REPLY_MODES:
        raise RuntimeError(f"BOT_DEFAULT_REPLY_MODE must be one of {sorted(VALID_REPLY_MODES)}")

    openai_api_key = str(env.OPENAI_API_KEY).strip() if getattr(env, "OPENAI_API_KEY", None) else None
    gemini_api_key = str(env.GEMINI_API_KEY).strip() if getattr(env, "GEMINI_API_KEY", None) else None
    anthropic_api_key = str(env.ANTHROPIC_API_KEY).strip() if getattr(env, "ANTHROPIC_API_KEY", None) else None

    model_catalog = default_model_catalog()

    raw_default_model_alias = str(getattr(env, "BOT_DEFAULT_MODEL_ALIAS", "") or "").strip()
    default_model_alias = raw_default_model_alias or infer_default_model_alias(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key,
        model_catalog=model_catalog,
    )

    conversation_timeout_seconds = int(getattr(env, "BOT_CONVERSATION_TIMEOUT_SECONDS", "300") or "300")
    if conversation_timeout_seconds <= 0:
        raise RuntimeError("BOT_CONVERSATION_TIMEOUT_SECONDS must be greater than 0")

    safety_identifier_salt = (
        str(env.SAFETY_IDENTIFIER_SALT).strip()
        if getattr(env, "SAFETY_IDENTIFIER_SALT", None)
        else None
    )
    if safety_identifier_salt is not None and len(safety_identifier_salt) < 16:
        raise RuntimeError("SAFETY_IDENTIFIER_SALT must be at least 16 characters when set")

    return Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key,
        owner_user_ids=parse_owner_ids(str(getattr(env, "BOT_OWNER_USER_IDS", "") or "")),
        default_model_alias=default_model_alias,
        default_reply_mode=default_reply_mode,
        skip_prefix=str(getattr(env, "BOT_SKIP_PREFIX", "//") or "//"),
        conversation_timeout_seconds=conversation_timeout_seconds,
        system_prompt=str(getattr(env, "BOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT),
        safety_identifier_salt=safety_identifier_salt,
        telegram_message_limit=int(getattr(env, "BOT_TELEGRAM_MESSAGE_LIMIT", "3900") or "3900"),
        render_edit_interval_seconds=float(getattr(env, "BOT_RENDER_EDIT_INTERVAL_SECONDS", "1.0") or "1.0"),
        model_catalog=model_catalog,
    )


def build_registry(config: Config) -> ProviderRegistry:
    providers = []
    if config.openai_api_key:
        providers.append(OpenAIProvider(config.openai_api_key, _provider_models(config.model_catalog, "openai")))

    if config.gemini_api_key:
        providers.append(GeminiProvider(config.gemini_api_key, _provider_models(config.model_catalog, "gemini")))

    if config.anthropic_api_key:
        providers.append(ClaudeProvider(config.anthropic_api_key, _provider_models(config.model_catalog, "claude")))

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
