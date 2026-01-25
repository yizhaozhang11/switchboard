from __future__ import annotations

import base64
from typing import AsyncIterator

from app.types import ChatRequest, ModelSpec, StreamEvent, build_content_parts

IGNORED_EVENT_TYPES = {
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.done",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.output_text.annotation.added",
    "keepalive",
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, models: list[ModelSpec]) -> None:
        self.api_key = api_key
        self._models = list(models)
        self._client = None

    def get_models(self) -> list[ModelSpec]:
        return list(self._models)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, max_retries=0, timeout=120)
        return self._client

    def _build_reasoning_config(self, request: ChatRequest) -> dict[str, object] | None:
        if not request.model.supports_reasoning:
            return None
        reasoning: dict[str, object] = {"summary": "detailed"}
        if request.model.reasoning_effort:
            reasoning["effort"] = request.model.reasoning_effort
        return reasoning

    async def stream_reply(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        input_messages = []
        for message in request.conversation:
            if message.role == "assistant":
                input_messages.append({"role": "assistant", "content": message.content})
            else:
                content = []
                message_parts = message.parts or build_content_parts(message.content, message.images)
                for part in message_parts:
                    if part.kind == "text":
                        if part.text:
                            content.append({"type": "input_text", "text": part.text})
                        continue

                    if part.image is None:
                        continue
                    image_base64 = base64.b64encode(part.image.data).decode("ascii")
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{part.image.mime_type};base64,{image_base64}",
                            "detail": "high",
                        }
                    )
                input_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )

        chunks: list[str] = []
        seen_done = False
        try:
            request_kwargs: dict[str, object] = {
                "model": request.model.model_id,
                "input": input_messages,
                "instructions": request.system_prompt,
                "stream": True,
            }
            reasoning = self._build_reasoning_config(request)
            if reasoning is not None:
                request_kwargs["reasoning"] = reasoning
            if "search" in request.requested_tools:
                request_kwargs["tools"] = [{"type": "web_search"}]
            if request.safety_identifier:
                request_kwargs["safety_identifier"] = request.safety_identifier
            stream = await client.responses.create(
                **request_kwargs,
            )
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    chunks.append(event.delta)
                    yield StreamEvent(kind="text_delta", text=event.delta)
                elif event_type == "response.reasoning_summary_text.delta":
                    yield StreamEvent(kind="reasoning_delta", text=event.delta)
                elif event_type == "response.reasoning_summary_part.done":
                    yield StreamEvent(kind="reasoning_delimiter")
                elif event_type == "response.completed":
                    seen_done = True
                    usage = None
                    if getattr(event.response, "usage", None) is not None:
                        usage = {
                            "input_tokens": event.response.usage.input_tokens,
                            "output_tokens": event.response.usage.output_tokens,
                        }
                    yield StreamEvent(kind="done", text="".join(chunks), usage=usage)
                elif event_type == "response.failed":
                    error = event.response.error
                    yield StreamEvent(kind="error", text=f"{error.code}: {error.message}")
                    return
                elif event_type == "error":
                    yield StreamEvent(kind="error", text=str(event))
                    return
                elif event_type in IGNORED_EVENT_TYPES:
                    continue
                else:
                    continue
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        if not seen_done:
            yield StreamEvent(kind="done", text="".join(chunks))
