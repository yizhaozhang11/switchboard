from __future__ import annotations

import sys
import types
import unittest

from app.providers.openai import OpenAIProvider
from app.types import ChatRequest, ContentPart, ConversationMessage, ImageRef, ModelSpec


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_events: list[object] | None = None

    async def create(
        self,
        *,
        model: str,
        input: list[dict[str, object]],
        instructions: str | None,
        stream: bool,
        reasoning: dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
        safety_identifier: str | None = None,
    ):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "instructions": instructions,
                "stream": stream,
                "reasoning": reasoning,
                "tools": tools,
                "safety_identifier": safety_identifier,
            }
        )

        async def iterator():
            events = self.next_events
            if events is None:
                events = [
                    types.SimpleNamespace(type="response.output_text.delta", delta="Hello "),
                    types.SimpleNamespace(type="response.output_text.delta", delta="OpenAI"),
                    types.SimpleNamespace(
                        type="response.completed",
                        response=types.SimpleNamespace(
                            usage=types.SimpleNamespace(input_tokens=17, output_tokens=9),
                        ),
                    ),
                ]
            for event in events:
                yield event

        return iterator()


class FakeAsyncOpenAI:
    def __init__(self, *, api_key: str, max_retries: int, timeout: int) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.responses = FakeResponsesAPI()


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved_module = sys.modules.get("openai")
        openai_module = types.ModuleType("openai")
        openai_module.AsyncOpenAI = FakeAsyncOpenAI
        sys.modules["openai"] = openai_module

    def tearDown(self) -> None:
        if self._saved_module is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = self._saved_module

    def _make_provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="openai",
                    model_id="gpt-5.4",
                    aliases=("o",),
                    supports_images=True,
                    supports_reasoning=True,
                )
            ],
        )

    async def test_stream_reply_builds_openai_request_and_streams_text(self) -> None:
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
            [("text_delta", "Hello "), ("text_delta", "OpenAI"), ("done", "Hello OpenAI")],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 17, "output_tokens": 9})

        client = provider._client
        assert client is not None
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5.4")
        self.assertEqual(call["instructions"], "Answer briefly.")
        self.assertTrue(call["stream"])
        self.assertEqual(call["reasoning"], {"summary": "detailed"})
        self.assertIsNone(call["safety_identifier"])
        self.assertEqual(
            call["input"],
            [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": [{"type": "input_text", "text": "How are you?"}]},
            ],
        )

    async def test_stream_reply_passes_reasoning_effort_when_configured(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="openai",
                    model_id="gpt-5.4",
                    aliases=("o",),
                    supports_images=True,
                    supports_reasoning=True,
                    reasoning_effort="high",
                )
            ],
        )
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Hi")],
            system_prompt="Answer briefly.",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        self.assertEqual(client.responses.calls[0]["reasoning"], {"summary": "detailed", "effort": "high"})

    async def test_stream_reply_includes_data_url_images(self) -> None:
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
        user_message = client.responses.calls[0]["input"][0]
        self.assertEqual(user_message["role"], "user")
        content = user_message["content"]
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "input_text", "text": "look"})
        self.assertEqual(
            content[1],
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,cG5nLWJ5dGVz",
                "detail": "high",
            },
        )

    async def test_stream_reply_preserves_interleaved_text_and_image_parts(self) -> None:
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
        content = client.responses.calls[0]["input"][0]["content"]
        self.assertEqual(
            content,
            [
                {"type": "input_text", "text": "1.\nfirst queued"},
                {"type": "input_image", "image_url": "data:image/png;base64,b25l", "detail": "high"},
                {"type": "input_text", "text": "2.\nsecond queued"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,dHdv", "detail": "high"},
            ],
        )

    async def test_stream_reply_emits_reasoning_summary_events(self) -> None:
        provider = self._make_provider()
        client = provider._get_client()
        client.responses.next_events = [
            types.SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Step 1"),
            types.SimpleNamespace(type="response.reasoning_summary_part.done"),
            types.SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Step 2"),
            types.SimpleNamespace(type="response.output_text.delta", delta="Answer"),
            types.SimpleNamespace(
                type="response.completed",
                response=types.SimpleNamespace(
                    usage=types.SimpleNamespace(input_tokens=12, output_tokens=4),
                ),
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
                ("reasoning_delta", "Step 2"),
                ("text_delta", "Answer"),
                ("done", "Answer"),
            ],
        )

    async def test_stream_reply_passes_safety_identifier_when_present(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Hi")],
            system_prompt="Answer briefly.",
            safety_identifier="hashed-user-id",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        self.assertEqual(client.responses.calls[0]["safety_identifier"], "hashed-user-id")

    async def test_stream_reply_enables_web_search_tool_when_requested(self) -> None:
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
        self.assertEqual(client.responses.calls[0]["tools"], [{"type": "web_search"}])


if __name__ == "__main__":
    unittest.main()
