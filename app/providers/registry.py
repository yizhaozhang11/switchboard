from __future__ import annotations

from dataclasses import dataclass

from app.providers.base import Provider
from app.types import ModelSpec, RequestedTool


MODEL_ALIAS_TOOL_SEPARATOR = "-"
DEFAULT_REQUESTED_TOOL_ALIASES: dict[str, RequestedTool] = {"s": "search"}
PROVIDER_REQUESTED_TOOL_ALIASES: dict[str, dict[str, RequestedTool]] = {
    "openai": DEFAULT_REQUESTED_TOOL_ALIASES,
    "gemini": {"s": "search", "u": "fetch"},
    "claude": {"s": "search", "u": "fetch"},
}


def supported_tool_aliases_for_provider(provider_name: str) -> tuple[str, ...]:
    return tuple(PROVIDER_REQUESTED_TOOL_ALIASES.get(provider_name, DEFAULT_REQUESTED_TOOL_ALIASES))


@dataclass(frozen=True)
class ResolvedModel:
    provider: Provider
    model: ModelSpec


@dataclass(frozen=True)
class ResolvedModelSelection:
    provider: Provider
    model: ModelSpec
    requested_tools: tuple[RequestedTool, ...] = ()


class ProviderRegistry:
    def __init__(self, providers: list[Provider]) -> None:
        self._providers = providers
        self._models: list[ResolvedModel] = []
        self._alias_map: dict[str, ResolvedModel] = {}

        for provider in providers:
            for model in provider.get_models():
                resolved = ResolvedModel(provider=provider, model=model)
                self._models.append(resolved)
                for alias in model.aliases:
                    normalized = alias.casefold()
                    if normalized in self._alias_map:
                        raise ValueError(f"Duplicate model alias: {alias}")
                    self._alias_map[normalized] = resolved

    def list_models(self) -> list[ResolvedModel]:
        return list(self._models)

    def resolve(self, alias: str) -> ResolvedModel | None:
        selection = self.resolve_selection(alias)
        if selection is None:
            return None
        return ResolvedModel(provider=selection.provider, model=selection.model)

    def resolve_selection(self, alias: str) -> ResolvedModelSelection | None:
        normalized = alias.casefold().strip()
        if not normalized:
            return None

        resolved = self._alias_map.get(normalized)
        if resolved is not None:
            return ResolvedModelSelection(provider=resolved.provider, model=resolved.model)

        parts = normalized.split(MODEL_ALIAS_TOOL_SEPARATOR)
        if len(parts) < 2:
            return None

        for split_index in range(len(parts) - 1, 0, -1):
            base_alias = MODEL_ALIAS_TOOL_SEPARATOR.join(parts[:split_index])
            requested_tool_aliases = tuple("".join(parts[split_index:]))
            resolved = self._alias_map.get(base_alias)
            if resolved is None:
                continue
            provider_tool_aliases = PROVIDER_REQUESTED_TOOL_ALIASES.get(
                resolved.model.provider,
                DEFAULT_REQUESTED_TOOL_ALIASES,
            )
            try:
                requested_tools = tuple(provider_tool_aliases[tool] for tool in requested_tool_aliases)
            except KeyError:
                continue
            if len(set(requested_tools)) != len(requested_tools):
                return None
            if requested_tools and not resolved.model.supports_tools:
                return None
            return ResolvedModelSelection(
                provider=resolved.provider,
                model=resolved.model,
                requested_tools=requested_tools,
            )
        return None
