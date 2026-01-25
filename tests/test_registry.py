from __future__ import annotations

import unittest

from app.providers.registry import ProviderRegistry
from app.types import ChatRequest, ModelSpec, StreamEvent


class DummyProvider:
    def __init__(self, name: str, model: ModelSpec) -> None:
        self.name = name
        self._model = model

    def get_models(self) -> list[ModelSpec]:
        return [self._model]

    async def stream_reply(self, request: ChatRequest):
        yield StreamEvent(kind="done", text="ok")


class RegistryTests(unittest.TestCase):
    def test_resolve_alias(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="dummy", model_id="model-a", aliases=("a", "alias")),
        )
        registry = ProviderRegistry([provider])
        resolved = registry.resolve("alias")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.model.model_id, "model-a")

    def test_duplicate_alias_raises(self) -> None:
        provider_a = DummyProvider("a", ModelSpec(provider="a", model_id="m1", aliases=("dup",)))
        provider_b = DummyProvider("b", ModelSpec(provider="b", model_id="m2", aliases=("dup",)))
        with self.assertRaises(ValueError):
            ProviderRegistry([provider_a, provider_b])

    def test_resolve_selection_parses_search_suffix(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="openai", model_id="model-a", aliases=("alias",), supports_tools=True),
        )
        registry = ProviderRegistry([provider])
        selection = registry.resolve_selection("alias-s")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.model_id, "model-a")
        self.assertEqual(selection.requested_tools, ("search",))

    def test_resolve_selection_parses_fetch_suffix_for_supported_provider(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="gemini", model_id="model-a", aliases=("alias",), supports_tools=True),
        )
        registry = ProviderRegistry([provider])
        selection = registry.resolve_selection("alias-u")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.model_id, "model-a")
        self.assertEqual(selection.requested_tools, ("fetch",))

    def test_resolve_selection_parses_combined_suffix_characters(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="gemini", model_id="model-a", aliases=("alias",), supports_tools=True),
        )
        registry = ProviderRegistry([provider])
        selection = registry.resolve_selection("alias-su")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.model_id, "model-a")
        self.assertEqual(selection.requested_tools, ("search", "fetch"))

    def test_resolve_selection_parses_combined_suffix_characters_across_separators(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="gemini", model_id="model-a", aliases=("alias",), supports_tools=True),
        )
        registry = ProviderRegistry([provider])
        selection = registry.resolve_selection("alias-s-u")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.model_id, "model-a")
        self.assertEqual(selection.requested_tools, ("search", "fetch"))

    def test_resolve_selection_prefers_exact_alias_over_tool_suffix(self) -> None:
        provider_a = DummyProvider(
            "a",
            ModelSpec(provider="openai", model_id="base", aliases=("alias",), supports_tools=True),
        )
        provider_b = DummyProvider(
            "b",
            ModelSpec(provider="b", model_id="exact", aliases=("alias-s",)),
        )
        registry = ProviderRegistry([provider_a, provider_b])
        selection = registry.resolve_selection("alias-s")
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.model.model_id, "exact")
        self.assertEqual(selection.requested_tools, ())

    def test_resolve_selection_rejects_search_suffix_for_models_without_tool_support(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="openai", model_id="model-a", aliases=("alias",), supports_tools=False),
        )
        registry = ProviderRegistry([provider])
        self.assertIsNone(registry.resolve_selection("alias-s"))

    def test_resolve_selection_rejects_fetch_suffix_for_provider_without_fetch_support(self) -> None:
        provider = DummyProvider(
            "dummy",
            ModelSpec(provider="openai", model_id="model-a", aliases=("alias",), supports_tools=True),
        )
        registry = ProviderRegistry([provider])
        self.assertIsNone(registry.resolve_selection("alias-u"))


if __name__ == "__main__":
    unittest.main()
