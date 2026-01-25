from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.model_catalog import (
    DEFAULT_MODEL_CONFIG_BASENAME,
    ModelCatalog,
    ensure_model_catalog,
    load_model_catalog,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a Telegram chat. "
    "Answer clearly and briefly, keep context local to the active conversation, "
    "and avoid assuming you should answer messages that explicitly opt out."
)

VALID_REPLY_MODES = {"auto", "mention", "off"}


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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
    data_dir: Path
    db_path: Path
    model_catalog_path: Path
    model_catalog: ModelCatalog
    system_prompt: str
    safety_identifier_salt: str | None = None
    poll_timeout_seconds: int = 30
    render_edit_interval_seconds: float = 1.0
    telegram_message_limit: int = 3900

    @classmethod
    def from_env(cls, root_dir: Path) -> "Config":
        load_dotenv(root_dir / ".env")

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

        default_reply_mode = os.getenv("BOT_DEFAULT_REPLY_MODE", "auto").strip() or "auto"
        if default_reply_mode not in VALID_REPLY_MODES:
            raise RuntimeError(f"BOT_DEFAULT_REPLY_MODE must be one of {sorted(VALID_REPLY_MODES)}")

        data_dir = Path(os.getenv("BOT_DATA_DIR", "./data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = (root_dir / data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        db_path = data_dir / "bot.sqlite3"
        raw_model_catalog_path = os.getenv("BOT_MODEL_CONFIG_PATH", "").strip()
        if raw_model_catalog_path:
            model_catalog_path = Path(raw_model_catalog_path).expanduser()
            if not model_catalog_path.is_absolute():
                model_catalog_path = (root_dir / model_catalog_path).resolve()
            if not model_catalog_path.is_file():
                raise RuntimeError(f"BOT_MODEL_CONFIG_PATH does not exist: {model_catalog_path}")
        else:
            model_catalog_path = data_dir / DEFAULT_MODEL_CONFIG_BASENAME
            ensure_model_catalog(model_catalog_path)
        model_catalog = load_model_catalog(model_catalog_path)
        openai_api_key = os.getenv("OPENAI_API_KEY") or None
        gemini_api_key = os.getenv("GEMINI_API_KEY") or None
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or None
        raw_default_model_alias = os.getenv("BOT_DEFAULT_MODEL_ALIAS", "").strip()
        default_model_alias = raw_default_model_alias or infer_default_model_alias(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            model_catalog=model_catalog,
        )
        conversation_timeout_seconds = int(os.getenv("BOT_CONVERSATION_TIMEOUT_SECONDS", "300"))
        if conversation_timeout_seconds <= 0:
            raise RuntimeError("BOT_CONVERSATION_TIMEOUT_SECONDS must be greater than 0")
        safety_identifier_salt = os.getenv("SAFETY_IDENTIFIER_SALT", "").strip() or None
        if safety_identifier_salt is not None and len(safety_identifier_salt) < 16:
            raise RuntimeError("SAFETY_IDENTIFIER_SALT must be at least 16 characters when set")

        return cls(
            telegram_bot_token=telegram_bot_token,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            owner_user_ids=parse_owner_ids(os.getenv("BOT_OWNER_USER_IDS", "")),
            default_model_alias=default_model_alias,
            default_reply_mode=default_reply_mode,
            skip_prefix=os.getenv("BOT_SKIP_PREFIX", "//"),
            conversation_timeout_seconds=conversation_timeout_seconds,
            data_dir=data_dir,
            db_path=db_path,
            model_catalog_path=model_catalog_path,
            model_catalog=model_catalog,
            system_prompt=os.getenv("BOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            safety_identifier_salt=safety_identifier_salt,
        )
