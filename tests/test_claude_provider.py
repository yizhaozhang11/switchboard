from __future__ import annotations

import sys
import types
import unittest

from app.providers.claude import ClaudeProvider
from app.types import ChatRequest, ContentPart, ConversationMessage, ImageRef, ModelSpec


class FakeMessagesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_events: list[object] | None = None

    async def create(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict[str, object]],
        max_tokens: int,
        stream: bool,
        thinking: dict[str, object] | None = None,
        output_config: dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ):
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
                "thinking": thinking,
                "output_config": output_config,
                "tools": tools,
            }
        )

        async def iterator():
            events = self.next_events
            if events is None:
                events = [
                    types.SimpleNamespace(
                        type="message_start",
                        message=types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=19)),
                    ),
                    types.SimpleNamespace(
                        type="content_block_delta",
                        delta=types.SimpleNamespace(type="text_delta", text="Hello "),
                    ),
                    types.SimpleNamespace(
                        type="content_block_delta",
                        delta=types.SimpleNamespace(type="text_delta", text="Claude"),
                    ),
                    types.SimpleNamespace(
                        type="message_delta",
                        usage=types.SimpleNamespace(output_tokens=8),
                    ),
                    types.SimpleNamespace(type="message_stop"),
                ]
            for event in events:
                yield event

        return iterator()


class FakeAsyncAnthropic:
    def __init__(self, *, api_key: str, max_retries: int, timeout: int) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.messages = FakeMessagesAPI()


class ClaudeProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved_module = sys.modules.get("anthropic")
        anthropic_module = types.ModuleType("anthropic")
        anthropic_module.AsyncAnthropic = FakeAsyncAnthropic
        sys.modules["anthropic"] = anthropic_module

    def tearDown(self) -> None:
        if self._saved_module is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = self._saved_module

    def _make_provider(self) -> ClaudeProvider:
        return ClaudeProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="claude",
                    model_id="claude-sonnet-4-6",
                    aliases=("cs",),
                    supports_images=True,
                    supports_reasoning=True,
                    thinking_mode="adaptive",
                    output_effort="high",
                ),
                ModelSpec(
                    provider="claude",
                    model_id="claude-opus-4-6",
                    aliases=("c",),
                    supports_images=True,
                    supports_reasoning=True,
                    thinking_mode="adaptive",
                    output_effort="high",
                ),
                ModelSpec(
                    provider="claude",
                    model_id="claude-haiku-4-5-20251001",
                    aliases=("ch",),
                    supports_images=True,
                    supports_reasoning=True,
                    thinking_mode="enabled",
                    thinking_budget_tokens=2048,
                ),
            ],
        )

    async def test_stream_reply_builds_claude_request_and_streams_text(self) -> None:
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
            [("text_delta", "Hello "), ("text_delta", "Claude"), ("done", "Hello Claude")],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 19, "output_tokens": 8})

        client = provider._client
        assert client is not None
        call = client.messages.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-4-6")
        self.assertEqual(call["system"], "Answer briefly.")
        self.assertEqual(call["max_tokens"], 8192)
        self.assertTrue(call["stream"])
        self.assertEqual(call["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(call["output_config"], {"effort": "high"})
        self.assertIsNone(call["tools"])
        self.assertEqual(
            call["messages"],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "How are you?"},
            ],
        )

    async def test_stream_reply_includes_base64_images(self) -> None:
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
        message = client.messages.calls[0]["messages"][0]
        self.assertEqual(message["role"], "user")
        content = message["content"]
        assert isinstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "look"})
        self.assertEqual(
            content[1],
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "cG5nLWJ5dGVz",
                },
            },
        )

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
        content = client.messages.calls[0]["messages"][0]["content"]
        self.assertEqual(
            content,
            [
                {"type": "text", "text": "1.\nfirst queued"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "b25l",
                    },
                },
                {"type": "text", "text": "2.\nsecond queued"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "dHdv",
                    },
                },
            ],
        )

    async def test_stream_reply_preserves_text_only_merged_prompt_spacing(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        merged_prompt = (
            "Additional user messages sent while you were replying:\n\n"
            "1.\n"
            "second\n\n"
            "2.\n"
            "third"
        )
        request = ChatRequest(
            model=model,
            conversation=[
                ConversationMessage(
                    role="user",
                    content=merged_prompt,
                    parts=(
                        ContentPart(
                            kind="text",
                            text="Additional user messages sent while you were replying:",
                        ),
                        ContentPart(kind="text", text="1.\nsecond"),
                        ContentPart(kind="text", text="2.\nthird"),
                    ),
                ),
            ],
            system_prompt="Answer briefly.",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        self.assertEqual(
            client.messages.calls[0]["messages"][0],
            {"role": "user", "content": merged_prompt},
        )

    async def test_stream_reply_uses_manual_thinking_for_haiku(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[2]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Hi")],
            system_prompt="Answer briefly.",
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        call = client.messages.calls[0]
        self.assertEqual(call["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(
            call["thinking"],
            {"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
        )
        self.assertIsNone(call["output_config"])

    async def test_stream_reply_emits_reasoning_summary_events(self) -> None:
        provider = self._make_provider()
        client = provider._get_client()
        client.messages.next_events = [
            types.SimpleNamespace(
                type="message_start",
                message=types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=13)),
            ),
            types.SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=types.SimpleNamespace(type="thinking"),
            ),
            types.SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=types.SimpleNamespace(type="thinking_delta", thinking="Step 1"),
            ),
            types.SimpleNamespace(type="content_block_stop", index=0),
            types.SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=types.SimpleNamespace(type="text"),
            ),
            types.SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=types.SimpleNamespace(type="text_delta", text="Claude answer"),
            ),
            types.SimpleNamespace(
                type="message_delta",
                usage=types.SimpleNamespace(output_tokens=6),
            ),
            types.SimpleNamespace(type="content_block_stop", index=1),
            types.SimpleNamespace(type="message_stop"),
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
                ("text_delta", "Claude answer"),
                ("done", "Claude answer"),
            ],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 13, "output_tokens": 6})

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
        self.assertEqual(
            client.messages.calls[0]["tools"],
            [{"type": "web_search_20250305", "name": "web_search"}],
        )

    async def test_stream_reply_enables_web_fetch_tool_when_requested(self) -> None:
        provider = self._make_provider()
        model = provider.get_models()[0]
        request = ChatRequest(
            model=model,
            conversation=[ConversationMessage(role="user", content="Analyze https://example.com")],
            requested_tools=("fetch",),
        )

        _ = [event async for event in provider.stream_reply(request)]

        client = provider._client
        assert client is not None
        self.assertEqual(
            client.messages.calls[0]["tools"],
            [{"type": "web_fetch_20250910", "name": "web_fetch"}],
        )


if __name__ == "__main__":
    unittest.main()
