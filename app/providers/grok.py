from __future__ import annotations

import base64
from typing import Any, AsyncIterator

from app.types import ChatRequest, ConversationMessage, ModelSpec, StreamEvent, build_content_parts

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokProvider:
    name = "grok"

    def __init__(self, api_key: str, models: list[ModelSpec]) -> None:
        self.api_key = api_key
        self._models = list(models)
        self._client = None

    def get_models(self) -> list[ModelSpec]:
        return list(self._models)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - covered by packaging/bootstrap
                raise RuntimeError("openai is not installed") from exc

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=XAI_BASE_URL,
                max_retries=0,
                timeout=120,
            )
        return self._client

    def _build_messages(self, request: ChatRequest) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for message in request.conversation:
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
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part.image.mime_type};base64,{image_base64}",
                            "detail": "high",
                        },
                    }
                )
            messages.append({"role": role, "content": content_blocks})

        return messages

    def _extract_delta_text(self, delta: Any, name: str) -> str:
        text = getattr(delta, name, None)
        if text is not None:
            return text
        model_extra = getattr(delta, "model_extra", None)
        if isinstance(model_extra, dict):
            extra_text = model_extra.get(name)
            if isinstance(extra_text, str):
                return extra_text
        return ""

    def _extract_usage(self, usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None
        extracted: dict[str, int] = {}
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        else:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is not None:
            extracted["input_tokens"] = input_tokens
        if output_tokens is not None:
            extracted["output_tokens"] = output_tokens
        return extracted or None

    async def stream_reply(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        try:
            create_kwargs: dict[str, object] = {
                "model": request.model.model_id,
                "messages": self._build_messages(request),
                "stream": True,
            }
            if request.model.reasoning_effort:
                create_kwargs["reasoning_effort"] = request.model.reasoning_effort

            stream = await self._get_client().chat.completions.create(**create_kwargs)
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        chunks: list[str] = []
        usage: dict[str, int] | None = None
        saw_reasoning = False
        closed_reasoning = False
        try:
            async for response in stream:
                response_usage = self._extract_usage(getattr(response, "usage", None))
                if response_usage is not None:
                    usage = response_usage

                choices = getattr(response, "choices", None) or []
                if not choices:
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    role = getattr(delta, "role", None)
                    if role is not None and role != "assistant":
                        yield StreamEvent(kind="error", text=f"Unexpected delta role: {role}")
                        return

                    reasoning_text = self._extract_delta_text(delta, "reasoning_content")
                    if reasoning_text:
                        saw_reasoning = True
                        yield StreamEvent(kind="reasoning_delta", text=reasoning_text)

                    text = self._extract_delta_text(delta, "content")
                    if text:
                        if saw_reasoning and not closed_reasoning:
                            closed_reasoning = True
                            yield StreamEvent(kind="reasoning_delimiter")
                        chunks.append(text)
                        yield StreamEvent(kind="text_delta", text=text)

                finish_reason = getattr(choice, "finish_reason", None)
                finish_details = getattr(choice, "finish_details", None)
                if finish_reason is None and finish_details is not None:
                    if isinstance(finish_details, dict):
                        finish_reason = finish_details.get("type")
                    else:
                        finish_reason = getattr(finish_details, "type", None)
                if finish_reason is None:
                    model_extra = getattr(choice, "model_extra", None)
                    if isinstance(model_extra, dict):
                        finish_details = model_extra.get("finish_details")
                        if isinstance(finish_details, dict):
                            finish_reason = finish_details.get("type")
                if finish_reason is None:
                    continue
                if finish_reason == "stop":
                    continue
                if finish_reason == "length":
                    yield StreamEvent(kind="error", text="Output truncated due to length limit")
                    return
                if finish_details is not None and finish_reason != getattr(choice, "finish_reason", None):
                    yield StreamEvent(kind="error", text=f"finish_details={finish_details}")
                    return
                yield StreamEvent(kind="error", text=f"finish_reason={finish_reason}")
                return
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        yield StreamEvent(kind="done", text="".join(chunks), usage=usage)
