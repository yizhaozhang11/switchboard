from __future__ import annotations

from typing import Any, AsyncIterator

from app.types import ChatRequest, ConversationMessage, ModelSpec, StreamEvent, build_content_parts


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, models: list[ModelSpec]) -> None:
        self.api_key = api_key
        self._models = list(models)
        self._client = None
        self._types = None

    def get_models(self) -> list[ModelSpec]:
        return list(self._models)

    def _get_sdk(self) -> tuple[Any, Any]:
        if self._client is None or self._types is None:
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError as exc:  # pragma: no cover - covered by packaging/bootstrap
                raise RuntimeError("google-genai is not installed") from exc

            self._client = genai.Client(api_key=self.api_key)
            self._types = genai_types
        return self._client, self._types

    def _build_contents(self, conversation: list[ConversationMessage], genai_types: Any) -> list[Any]:
        contents = []
        for message in conversation:
            role = "model" if message.role == "assistant" else "user"
            parts = []
            message_parts = message.parts or build_content_parts(message.content, message.images)
            for part in message_parts:
                if part.kind == "text":
                    if part.text:
                        parts.append(genai_types.Part.from_text(text=part.text))
                    continue
                if part.image is not None:
                    parts.append(genai_types.Part.from_bytes(data=part.image.data, mime_type=part.image.mime_type))
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=parts,
                )
            )
        return contents

    def _build_config(self, request: ChatRequest, genai_types: Any) -> Any | None:
        config_kwargs: dict[str, Any] = {}
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.model.supports_reasoning:
            thinking_kwargs: dict[str, Any] = {"include_thoughts": True}
            if request.model.reasoning_effort:
                thinking_kwargs["thinking_level"] = request.model.reasoning_effort
            if request.model.thinking_budget_tokens is not None:
                thinking_kwargs["thinking_budget"] = request.model.thinking_budget_tokens
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(**thinking_kwargs)
        tools: list[object] = []
        if "search" in request.requested_tools:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
        if "fetch" in request.requested_tools:
            tools.append({"url_context": {}})
        if tools:
            config_kwargs["tools"] = tools
        if not config_kwargs:
            return None
        return genai_types.GenerateContentConfig(**config_kwargs)

    def _extract_usage(self, usage_metadata: Any) -> dict[str, int] | None:
        usage: dict[str, int] = {}
        input_tokens = getattr(usage_metadata, "prompt_token_count", None)
        output_tokens = getattr(usage_metadata, "candidates_token_count", None)
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        return usage or None

    async def stream_reply(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        try:
            client, genai_types = self._get_sdk()
            stream = await client.aio.models.generate_content_stream(
                model=request.model.model_id,
                contents=self._build_contents(request.conversation, genai_types),
                config=self._build_config(request, genai_types),
            )
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        chunks: list[str] = []
        usage: dict[str, int] | None = None
        saw_reasoning = False
        closed_reasoning = False
        try:
            async for chunk in stream:
                prompt_feedback = getattr(chunk, "prompt_feedback", None)
                if prompt_feedback is not None:
                    block_reason = getattr(prompt_feedback, "block_reason", None)
                    if block_reason:
                        yield StreamEvent(kind="error", text=f"Prompt blocked: {block_reason}")
                        return

                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata is not None:
                    usage = self._extract_usage(usage_metadata)

                candidates = getattr(chunk, "candidates", None) or []
                handled_parts = False
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    candidate_parts = getattr(content, "parts", None) or []
                    for part in candidate_parts:
                        part_text = getattr(part, "text", "") or ""
                        if not part_text:
                            continue
                        handled_parts = True
                        if getattr(part, "thought", False):
                            saw_reasoning = True
                            yield StreamEvent(kind="reasoning_delta", text=part_text)
                        else:
                            if saw_reasoning and not closed_reasoning:
                                closed_reasoning = True
                                yield StreamEvent(kind="reasoning_delimiter")
                            chunks.append(part_text)
                            yield StreamEvent(kind="text_delta", text=part_text)
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason and finish_reason not in {"STOP", "MAX_TOKENS"}:
                        yield StreamEvent(kind="error", text=f"finish_reason={finish_reason}")
                        return

                if not handled_parts:
                    chunk_text = getattr(chunk, "text", "") or ""
                    if chunk_text:
                        if saw_reasoning and not closed_reasoning:
                            closed_reasoning = True
                            yield StreamEvent(kind="reasoning_delimiter")
                        chunks.append(chunk_text)
                        yield StreamEvent(kind="text_delta", text=chunk_text)
        except Exception as exc:
            yield StreamEvent(kind="error", text=f"{type(exc).__name__}: {exc}")
            return

        yield StreamEvent(kind="done", text="".join(chunks), usage=usage)
