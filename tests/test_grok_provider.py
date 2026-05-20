from __future__ import annotations

import sys
import types
import unittest

from app.providers.grok import GrokProvider, XAI_BASE_URL
from app.types import ChatRequest, ContentPart, ConversationMessage, ImageRef, ModelSpec


class FakeCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_chunks: list[object] | None = None

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool,
        reasoning_effort: str | None = None,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "stream": stream,
                "reasoning_effort": reasoning_effort,
            }
        )

        async def iterator():
            chunks = self.next_chunks
            if chunks is None:
                chunks = [
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(role="assistant", content=None, model_extra={}),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content=None, model_extra={"reasoning_content": "Step 1"}),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content="Hello ", model_extra={}),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content="Grok", model_extra={}),
                                finish_reason="stop",
                            )
                        ],
                        usage=types.SimpleNamespace(prompt_tokens=17, completion_tokens=9),
                    ),
                ]
            for chunk in chunks:
                yield chunk

        return iterator()


class FakeAsyncOpenAI:
    def __init__(self, *, api_key: str, base_url: str, max_retries: int, timeout: int) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.chat = types.SimpleNamespace(completions=FakeCompletionsAPI())


class GrokProviderTests(unittest.IsolatedAsyncioTestCase):
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

    def _make_provider(self) -> GrokProvider:
        return GrokProvider(
            api_key="test-key",
            models=[
                ModelSpec(
                    provider="grok",
                    model_id="grok-4.3",
                    aliases=("x",),
                    supports_images=True,
                    supports_reasoning=True,
                    reasoning_effort="high",
                )
            ],
        )

    async def test_stream_reply_builds_xai_request_and_streams_text(self) -> None:
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
            [
                ("reasoning_delta", "Step 1"),
                ("reasoning_delimiter", ""),
                ("text_delta", "Hello "),
                ("text_delta", "Grok"),
                ("done", "Hello Grok"),
            ],
        )
        self.assertEqual(events[-1].usage, {"input_tokens": 17, "output_tokens": 9})

        client = provider._client
        assert client is not None
        self.assertEqual(client.base_url, XAI_BASE_URL)
        self.assertEqual(client.api_key, "test-key")
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "grok-4.3")
        self.assertTrue(call["stream"])
        self.assertEqual(call["reasoning_effort"], "high")
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "How are you?"},
            ],
        )

    async def test_stream_reply_includes_data_url_images(self) -> None:
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
        content = client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertEqual(
            content,
            [
                {"type": "text", "text": "1.\nfirst queued"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,b25l", "detail": "high"},
                },
                {"type": "text", "text": "2.\nsecond queued"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,dHdv", "detail": "high"},
                },
            ],
        )

    async def test_stream_reply_reports_length_finish_reason(self) -> None:
        provider = self._make_provider()
        client = provider._get_client()
        client.chat.completions.next_chunks = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(content="partial", model_extra={}),
                        finish_reason="length",
                    )
                ],
                usage=None,
            )
        ]
        model = provider.get_models()[0]
        request = ChatRequest(model=model, conversation=[ConversationMessage(role="user", content="Hi")])

        events = [event async for event in provider.stream_reply(request)]

        self.assertEqual(
            [(event.kind, event.text) for event in events],
            [
                ("text_delta", "partial"),
                ("error", "Output truncated due to length limit"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
