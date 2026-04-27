from __future__ import annotations

import tomllib
from pathlib import Path

from app.providers.claude import (
    DEFAULT_MAX_TOKENS as CLAUDE_DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_THINKING_BUDGET_TOKENS as CLAUDE_DEFAULT_THINKING_BUDGET_TOKENS,
)
from app.types import ModelSpec


ModelCatalog = dict[str, tuple[ModelSpec, ...]]

DEFAULT_MODEL_CONFIG_BASENAME = "models.toml"
KNOWN_PROVIDERS = ("openai", "gemini", "claude")
OPENAI_REASONING_EFFORT_VALUES = frozenset({"minimal", "none", "low", "medium", "high", "xhigh"})
GEMINI_REASONING_EFFORT_VALUES = frozenset({"minimal", "low", "medium", "high"})
CLAUDE_OUTPUT_EFFORT_VALUES = frozenset({"low", "medium", "high", "xhigh", "max"})
DEFAULT_MODEL_CATALOG_TEXT = """# Edit this file to change model ids or aliases without changing bot code.
# Restart the bot after updates.

[[providers.openai.models]]
model_id = "gpt-5.4"
aliases = ["o"]
supports_images = true
supports_tools = true
supports_reasoning = true
reasoning_effort = "high"

[[providers.openai.models]]
model_id = "gpt-5.4-mini"
aliases = ["om"]
supports_images = true
supports_tools = true
supports_reasoning = true

[[providers.openai.models]]
model_id = "gpt-5.4-nano"
aliases = ["on"]
supports_images = true
supports_tools = true
supports_reasoning = true

[[providers.gemini.models]]
model_id = "gemini-3.1-pro-preview"
aliases = ["g"]
supports_images = true
supports_tools = true
supports_reasoning = true
reasoning_effort = "high"

[[providers.gemini.models]]
model_id = "gemini-3-flash-preview"
aliases = ["gf"]
supports_images = true
supports_tools = true
supports_reasoning = true

[[providers.gemini.models]]
model_id = "gemini-3.1-flash-lite-preview"
aliases = ["gl"]
supports_images = true
supports_tools = true
supports_reasoning = true

[[providers.claude.models]]
model_id = "claude-opus-4-6"
aliases = ["c"]
supports_images = true
supports_tools = true
supports_reasoning = true
thinking_mode = "adaptive"
output_effort = "high"

[[providers.claude.models]]
model_id = "claude-sonnet-4-6"
aliases = ["cs"]
supports_images = true
supports_tools = true
supports_reasoning = true
thinking_mode = "adaptive"
output_effort = "high"

[[providers.claude.models]]
model_id = "claude-haiku-4-5-20251001"
aliases = ["ch"]
supports_images = true
supports_tools = true
supports_reasoning = true
thinking_mode = "enabled"
thinking_budget_tokens = 2048
"""


def ensure_model_catalog(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_MODEL_CATALOG_TEXT, encoding="utf-8")


def default_model_catalog() -> ModelCatalog:
    return _parse_model_catalog(tomllib.loads(DEFAULT_MODEL_CATALOG_TEXT))


def load_model_catalog(path: Path) -> ModelCatalog:
    try:
        with path.open("rb") as handle:
            raw_catalog = tomllib.load(handle)
    except OSError as exc:
        raise RuntimeError(f"Unable to read model catalog from {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"Invalid model catalog in {path}: {exc}") from exc
    return _parse_model_catalog(raw_catalog)


def _parse_model_catalog(raw_catalog: object) -> ModelCatalog:
    if not isinstance(raw_catalog, dict):
        raise RuntimeError("Model catalog root must be a TOML table")

    providers = raw_catalog.get("providers")
    if not isinstance(providers, dict):
        raise RuntimeError("Model catalog must define a [providers] table")

    catalog: ModelCatalog = {provider: () for provider in KNOWN_PROVIDERS}
    for provider_name, raw_provider in providers.items():
        if provider_name not in KNOWN_PROVIDERS:
            known = ", ".join(KNOWN_PROVIDERS)
            raise RuntimeError(f"Unknown provider '{provider_name}' in model catalog; expected one of {known}")
        if not isinstance(raw_provider, dict):
            raise RuntimeError(f"Provider '{provider_name}' config must be a TOML table")
        raw_models = raw_provider.get("models", [])
        if not isinstance(raw_models, list):
            raise RuntimeError(f"Provider '{provider_name}' models must be a TOML array")
        models = tuple(
            _parse_model_spec(
                provider_name=provider_name,
                raw_model=raw_model,
                path_label=f"providers.{provider_name}.models[{index}]",
            )
            for index, raw_model in enumerate(raw_models)
        )
        catalog[provider_name] = models
    return catalog


def _parse_model_spec(*, provider_name: str, raw_model: object, path_label: str) -> ModelSpec:
    if not isinstance(raw_model, dict):
        raise RuntimeError(f"{path_label} must be a TOML table")

    model_id = _require_str(raw_model, "model_id", path_label)
    aliases = _require_str_tuple(raw_model, "aliases", path_label)
    supports_images = _optional_bool(raw_model, "supports_images", path_label, default=False)
    supports_files = _optional_bool(raw_model, "supports_files", path_label, default=False)
    supports_tools = _optional_bool(raw_model, "supports_tools", path_label, default=False)
    supports_reasoning = _optional_bool(raw_model, "supports_reasoning", path_label, default=False)
    supports_streaming = _optional_bool(raw_model, "supports_streaming", path_label, default=True)
    reasoning_effort = _optional_str(raw_model, "reasoning_effort", path_label)
    thinking_mode = _optional_str(raw_model, "thinking_mode", path_label)
    if thinking_mode is not None and thinking_mode not in {"adaptive", "enabled"}:
        raise RuntimeError(f"{path_label}.thinking_mode must be 'adaptive' or 'enabled'")
    thinking_budget_tokens = (
        _optional_nonnegative_int(raw_model, "thinking_budget_tokens", path_label)
        if provider_name == "gemini"
        else _optional_positive_int(raw_model, "thinking_budget_tokens", path_label)
    )
    output_effort = _optional_str(raw_model, "output_effort", path_label)
    max_output_tokens = _optional_positive_int(raw_model, "max_output_tokens", path_label)
    _validate_provider_model_constraints(
        provider_name=provider_name,
        path_label=path_label,
        supports_reasoning=supports_reasoning,
        reasoning_effort=reasoning_effort,
        thinking_mode=thinking_mode,
        thinking_budget_tokens=thinking_budget_tokens,
        output_effort=output_effort,
        max_output_tokens=max_output_tokens,
    )

    return ModelSpec(
        provider=provider_name,
        model_id=model_id,
        aliases=aliases,
        supports_images=supports_images,
        supports_files=supports_files,
        supports_tools=supports_tools,
        supports_reasoning=supports_reasoning,
        supports_streaming=supports_streaming,
        reasoning_effort=reasoning_effort,
        thinking_mode=thinking_mode,
        thinking_budget_tokens=thinking_budget_tokens,
        output_effort=output_effort,
        max_output_tokens=max_output_tokens,
    )


def _validate_provider_model_constraints(
    *,
    provider_name: str,
    path_label: str,
    supports_reasoning: bool,
    reasoning_effort: str | None,
    thinking_mode: str | None,
    thinking_budget_tokens: int | None,
    output_effort: str | None,
    max_output_tokens: int | None,
) -> None:
    if reasoning_effort is not None and not supports_reasoning:
        raise RuntimeError(f"{path_label}.reasoning_effort requires supports_reasoning = true")

    if provider_name == "openai":
        if reasoning_effort is not None and reasoning_effort not in OPENAI_REASONING_EFFORT_VALUES:
            allowed_values = ", ".join(sorted(OPENAI_REASONING_EFFORT_VALUES))
            raise RuntimeError(f"{path_label}.reasoning_effort must be one of: {allowed_values}")
        return

    if provider_name == "gemini":
        if reasoning_effort is not None and reasoning_effort not in GEMINI_REASONING_EFFORT_VALUES:
            allowed_values = ", ".join(sorted(GEMINI_REASONING_EFFORT_VALUES))
            raise RuntimeError(f"{path_label}.reasoning_effort must be one of: {allowed_values}")
        return

    if provider_name != "claude":
        return

    if reasoning_effort is not None:
        raise RuntimeError(f"{path_label}.reasoning_effort is not supported for provider '{provider_name}'")

    if not supports_reasoning:
        return

    if output_effort is not None and output_effort not in CLAUDE_OUTPUT_EFFORT_VALUES:
        allowed_values = ", ".join(sorted(CLAUDE_OUTPUT_EFFORT_VALUES))
        raise RuntimeError(f"{path_label}.output_effort must be one of: {allowed_values}")

    effective_thinking_mode = "adaptive" if thinking_mode == "adaptive" else "enabled"
    if effective_thinking_mode != "enabled":
        return

    effective_thinking_budget_tokens = thinking_budget_tokens or CLAUDE_DEFAULT_THINKING_BUDGET_TOKENS
    effective_max_output_tokens = max_output_tokens or CLAUDE_DEFAULT_MAX_OUTPUT_TOKENS
    if effective_thinking_budget_tokens > effective_max_output_tokens:
        raise RuntimeError(
            f"{path_label} has thinking_budget_tokens={effective_thinking_budget_tokens} "
            f"greater than max_output_tokens={effective_max_output_tokens}"
        )


def _require_str(raw_model: dict[str, object], key: str, path_label: str) -> str:
    value = raw_model.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{path_label}.{key} must be a non-empty string")
    return value.strip()


def _require_str_tuple(raw_model: dict[str, object], key: str, path_label: str) -> tuple[str, ...]:
    value = raw_model.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{path_label}.{key} must be a non-empty array")
    items: list[str] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise RuntimeError(f"{path_label}.{key}[{index}] must be a non-empty string")
        items.append(entry.strip())
    return tuple(items)


def _optional_str(raw_model: dict[str, object], key: str, path_label: str) -> str | None:
    value = raw_model.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{path_label}.{key} must be a non-empty string when set")
    return value.strip()


def _optional_bool(raw_model: dict[str, object], key: str, path_label: str, *, default: bool) -> bool:
    value = raw_model.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RuntimeError(f"{path_label}.{key} must be a boolean")
    return value


def _optional_positive_int(raw_model: dict[str, object], key: str, path_label: str) -> int | None:
    value = raw_model.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{path_label}.{key} must be a positive integer")
    return value


def _optional_nonnegative_int(raw_model: dict[str, object], key: str, path_label: str) -> int | None:
    value = raw_model.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{path_label}.{key} must be a non-negative integer")
    return value
