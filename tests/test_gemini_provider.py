from __future__ import annotations

import sys
import types
import unittest

from app.providers.gemini import GeminiProvider
from app.types import ChatRequest, ContentPart, ConversationMessage, ImageRef, ModelSpec


class FakePart:
    def __init__(
        self,
        *,
        text: str | None = None,
        data: bytes | None = None,
        mime_type: str | None = None,
        thought: bool | None = None,
    ) -> None:
        self.text = text
        self.data = data
        self.mime_type = mime_type
        self.thought = thought

    @classmethod
    def from_text(cls, *, text: str) -> "FakePart":
        return cls(text=text)

    @classmethod
    def from_bytes(cls, *, data: bytes, mime_type: str) -> "FakePart":
        return cls(data=data, mime_type=mime_type)


class FakeContent:
    def __init__(self, *, role: str, parts: list[FakePart]) -> None:
        self.role = role
        self.parts = parts


class FakeGenerateContentConfig:
    def __init__(
        self,
        *,
        system_instruction: str | None = None,
        thinking_config: object | None = None,
        tools: list[object] | None = None,
    ) -> None:
        self.system_instruction = system_instruction
        self.thinking_config = thinking_config
        self.tools = tools


class FakeThinkingConfig:
    def __init__(
        self,
        *,
        include_thoughts: bool,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        self.include_thoughts = include_thoughts
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget


class FakeGoogleSearch:
    pass


class FakeTool:
    def __init__(self, *, google_search: object | None = None) -> None:
        self.google_search = google_search


class FakeModelsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_chunks: list[object] | None = None

    async def generate_content_stream(self, *, model: str, contents: list[FakeContent], config: object):
        self.calls.append({"model": model, "contents": contents, "config": config})

        async def iterator():
            chunks = self.next_chunks
            if chunks is None:
                chunks = [
                    types.SimpleNamespace(text="Hello ", usage_metadata=None, prompt_feedback=None, candidates=[]),
                    types.SimpleNamespace(
                        text="Gemini",
                        usage_metadata=types.SimpleNamespace(prompt_token_count=11, candidates_token_count=7),
                        prompt_feedback=None,
                        candidates=[types.SimpleNamespace(finish_reason="STOP")],
                    ),
                ]
            for chunk in chunks:
                yield chunk

        return iterator()


class FakeClient:
    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.aio = types.SimpleNamespace(models=FakeModelsAPI())


class GeminiProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ("google", "google.genai", "google.genai.types")}

        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.genai")
        genai_types_module = types.ModuleType("google.genai.types")
        genai_module.Client = FakeClient
        genai_types_module.Part = FakePart
        genai_types_module.Content = FakeContent
        genai_types_module.GenerateContentConfig = FakeGenerateContentConfig
        genai_types_module.ThinkingConfig = FakeThinkingConfig
        genai_types_module.GoogleSearch = FakeGoogleSearch
        genai_types_module.Tool = FakeTool
        genai_module.types = genai_types_module
        google_module.genai = genai_module

        sys.modules["google"] = google_module
        sys.modules["google.genai"] = genai_module
        sys.modules["google.genai.types"] = genai_types_module

    def tearDown(self) -> None:
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _make_provider(self) -> GeminiProvider:
        return GeminiProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="gemini",
                    model_id="gemini-3.1-pro-preview",
                    aliases=("g",),
                    supports_images=True,
                    supports_reasoning=True,
                )
            ],
        )

    async def test_stream_reply_builds_gemini_request_and_streams_text(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[
                ConversationMessage(role="user", content="Hi"),
                ConversationMessage(role="assistant", content="Hello there"),
                ConversationMessage(role="user", content="How are you?"),
            ],
            system_prompt="Answer briefly.",
        )

        events = [event async for event in provider.stream_reply(request)]

        self.assertEqual(
            [(event.kind, event.text) for event in events],
            [("text_delta", "Hello "), ("text_delta", "Gemini"), ("done", "Hello Gemini")],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 11, "output_tokens": 7})

        client = provider._client
        assert client is not None
        call = client.aio.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-pro-preview")
        self.assertEqual([content.role for content in call["contents"]], ["user", "model", "user"])
        self.assertEqual(
            [[part.text for part in content.parts] for content in call["contents"]],
            [["Hi"], ["Hello there"], ["How are you?"]],
        )
        config = call["config"]
        assert isinstance(config, FakeGenerateContentConfig)
        self.assertEqual(config.system_instruction, "Answer briefly.")
        assert isinstance(config.thinking_config, FakeThinkingConfig)
        self.assertTrue(config.thinking_config.include_thoughts)
        self.assertIsNone(config.thinking_config.thinking_level)
        self.assertIsNone(config.thinking_config.thinking_budget)
        self.assertIsNone(config.tools)

    async def test_stream_reply_passes_reasoning_effort_and_budget(self) -> None:
        provider = GeminiProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="gemini",
                    model_id="gemini-3.1-pro-preview",
                    aliases=("g",),
                    supports_images=True,
                    supports_reasoning=True,
                    reasoning_effort="high",
                    thinking_budget_tokens=4096,
                )
            ],
        )
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Hi")],
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        config = client.aio.models.calls[0]["config"]
        assert isinstance(config, FakeGenerateContentConfig)
        assert isinstance(config.thinking_config, FakeThinkingConfig)
        self.assertEqual(config.thinking_config.thinking_level, "high")
        self.assertEqual(config.thinking_config.thinking_budget, 4096)

    async def test_stream_reply_includes_image_parts(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[
                ConversationMessage(
                    role="user",
                    content="look",
                    images=(ImageRef(mime_type="image/png", data=b"png-bytes"),),
                ),
            ],
            system_prompt="Describe images.",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        content = client.aio.models.calls[0]["contents"][0]
        self.assertEqual(content.role, "user")
        self.assertEqual([part.text for part in content.parts if part.text], ["look"])
        self.assertEqual([part.data for part in content.parts if part.data], [b"png-bytes"])
        self.assertEqual([part.mime_type for part in content.parts if part.mime_type], ["image/png"])

    async def test_stream_reply_preserves_interleaved_parts(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        image_one = ImageRef(mime_type="image/png", data=b"one")
        image_two = ImageRef(mime_type="image/jpeg", data=b"two")
        request = ChatRequest(
            model=model,
            conversation=[
                ConversationMessage(
                    role="user",
                    content="summary",
                    images=(image_one, image_two),
                    parts=(
                        ContentPart(kind="text", text="1.\nfirst queued"),
                        ContentPart(kind="image", image=image_one),
                        ContentPart(kind="text", text="2.\nsecond queued"),
                        ContentPart(kind="image", image=image_two),
                    ),
                ),
            ],
            system_prompt="Describe images.",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        content = client.aio.models.calls[0]["contents"][0]
        self.assertEqual(
            [(part.text, part.data, part.mime_type) for part in content.parts],
            [
                ("1.\nfirst queued", None, None),
                (None, b"one", "image/png"),
                ("2.\nsecond queued", None, None),
                (None, b"two", "image/jpeg"),
            ],
        )

    async def test_stream_reply_emits_reasoning_summary_events(self) -> None:
        provider = self._make_provider()
        client = provider._get_sdk()[0]
        client.aio.models.next_chunks = [
            types.SimpleNamespace(
                text="",
                usage_metadata=None,
                prompt_feedback=None,
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(parts=[FakePart(text="Step 1", thought=True)]),
                        finish_reason=None,
                    )
                ],
            ),
            types.SimpleNamespace(
                text="",
                usage_metadata=types.SimpleNamespace(prompt_token_count=9, candidates_token_count=5),
                prompt_feedback=None,
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(parts=[FakePart(text="Gemini answer", thought=False)]),
                        finish_reason="STOP",
                    )
                ],
            ),
        ]
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Hi")],
            system_prompt="Answer briefly.",
        )

        events = [event async for event in provider.stream_reply(request)]

        self.assertEqual(
            [(event.kind, event.text) for event in events],
            [
                ("reasoning_delta", "Step 1"),
                ("reasoning_delimiter", ""),
                ("text_delta", "Gemini answer"),
                ("done", "Gemini answer"),
            ],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 9, "output_tokens": 5})

    async def test_stream_reply_enables_google_search_when_requested(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="latest news")],
            requested_tools=("search",),
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        config = client.aio.models.calls[0]["config"]
        assert isinstance(config, FakeGenerateContentConfig)
        assert config.tools is not None
        self.assertEqual(len(config.tools), 1)
        tool = config.tools[0]
        assert isinstance(tool, FakeTool)
        self.assertIsInstance(tool.google_search, FakeGoogleSearch)

    async def test_stream_reply_enables_url_context_when_requested(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Summarize https://example.com")],
            requested_tools=("fetch",),
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        config = client.aio.models.calls[0]["config"]
        assert isinstance(config, FakeGenerateContentConfig)
        assert config.tools is not None
        self.assertEqual(config.tools, [{"url_context": {}}])


if __name__ == "__main__":
    unittest.main()
