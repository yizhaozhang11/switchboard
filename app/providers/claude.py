from __future__ import annotations

import base64
from typing import Any, AsyncIterator

from app.types import ChatRequest, ConversationMessage, ModelSpec, StreamEvent, build_content_parts

DEFAULT_MAX_TOKENS = 8192
DEFAULT_THINKING_BUDGET_TOKENS = 2048
CLAUDE_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
CLAUDE_WEB_FETCH_TOOL_TYPE = "web_fetch_20250910"


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str, models: list[ModelSpec]) -> None:
        self.api_key = api_key
        self._models = list(models)
        self._client = None

    def get_models(self) -> list[ModelSpec]:
        return list(self._models)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - covered by packaging/bootstrap
                raise RuntimeError("anthropic is not installed") from exc

            self._client = AsyncAnthropic(api_key=self.api_key, max_retries=0, timeout=120)
        return self._client

    def _build_messages(self, conversation: list[ConversationMessage]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for message in conversation:
            role = "assistant" if message.role == "assistant" else "user"
            message_parts = message.parts or build_content_parts(message.content, message.images)
            if all(part.kind == "text" for part in message_parts):
                content = message.content if message.content else "".join(part.text for part in message_parts)
                messages.append({"role": role, "content": content})
                continue

            content_blocks: list[dict[str, object]] = []
            for part in message_parts:
                if part.kind == "text":
                    if part.text:
                        content_blocks.append({"type": "text", "text": part.text})
                    continue
                if part.image is None:
                    continue
                image_base64 = base64.b64encode(part.image.data).decode("ascii")
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.image.mime_type,
                            "data": image_base64,
                        },
                    }
                )
            messages.append({"role": role, "content": content_blocks})
        return messages

    def _build_thinking_config(self, request: ChatRequest) -> dict[str, object] | None:
        if not request.model.supports_reasoning:
            return None
        if request.model.thinking_mode == "adaptive":
            return {"type": "adaptive", "display": "summarized"}
        budget_tokens = request.model.thinking_budget_tokens or DEFAULT_THINKING_BUDGET_TOKENS
        return {
            "type": "enabled",
            "budget_tokens": budget_tokens,
            "display": "summarized",
        }

    def _build_output_config(self, request: ChatRequest) -> dict[str, object] | None:
        if not request.model.supports_reasoning:
            return None
        if request.model.output_effort:
            return {"effort": request.model.output_effort}
        return None

    async def stream_reply(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        usage: dict[str, int] | None = None
        try:
            create_kwargs: dict[str, object] = {
                "model": request.model.model_id,
                "system": request.system_prompt or None,
                "messages": self._build_messages(request.conversation),
                "max_tokens": request.model.max_output_tokens or DEFAULT_MAX_TOKENS,
                "stream": True,
            }
            thinking = self._build_thinking_config(request)
            if thinking is not None:
                create_kwargs["thinking"] = thinking
            output_config = self._build_output_config(request)
            if output_config is not None:
                create_kwargs["output_config"] = output_config
            tools: list[dict[str, object]] = []
            if "search" in request.requested_tools:
                tools.append({"type": CLAUDE_WEB_SEARCH_TOOL_TYPE, "name": "web_search"})
            if "fetch" in request.requested_tools:
                tools.append({"type": CLAUDE_WEB_FETCH_TOOL_TYPE, "name": "web_fetch"})
            if tools:
                create_kwargs["tools"] = tools
            stream = await self._get_client().messages.create(
                **create_kwargs,
            )
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        chunks: list[str] = []
        thinking_blocks: dict[int, str] = {}
        try:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "content_block_start":
                    index = getattr(event, "index", None)
                    content_block = getattr(event, "content_block", None)
                    block_type = getattr(content_block, "type", None)
                    if isinstance(index, int) and isinstance(block_type, str):
                        thinking_blocks[index] = block_type
                    if block_type == "redacted_thinking":
                        yield StreamEvent(kind="reasoning_delta", text="[redacted_thinking]")
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            chunks.append(text)
                            yield StreamEvent(kind="text_delta", text=text)
                    elif delta_type == "thinking_delta":
                        text = getattr(delta, "thinking", "") or ""
                        if text:
                            yield StreamEvent(kind="reasoning_delta", text=text)
                    elif delta_type == "signature_delta":
                        continue
                elif event_type == "content_block_stop":
                    index = getattr(event, "index", None)
                    if isinstance(index, int):
                        block_type = thinking_blocks.pop(index, None)
                        if block_type in {"thinking", "redacted_thinking"}:
                            yield StreamEvent(kind="reasoning_delimiter")
                elif event_type == "message_start":
                    message = getattr(event, "message", None)
                    event_usage = getattr(message, "usage", None)
                    if event_usage is not None:
                        usage = {"input_tokens": getattr(event_usage, "input_tokens", 0)}
                elif event_type == "message_delta":
                    event_usage = getattr(event, "usage", None)
                    if event_usage is not None:
                        usage = usage or {}
                        output_tokens = getattr(event_usage, "output_tokens", None)
                        if output_tokens is not None:
                            usage["output_tokens"] = output_tokens
                elif event_type == "error":
                    error = getattr(event, "error", None)
                    if error is None:
                        yield StreamEvent(kind="error", text="Unknown Claude streaming error")
                    else:
                        error_type = getattr(error, "type", "error")
                        message = getattr(error, "message", str(error))
                        yield StreamEvent(kind="error", text=f"{error_type}: {message}")
                    return
                else:
                    continue
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        yield StreamEvent(kind="done", text="".join(chunks), usage=usage)
