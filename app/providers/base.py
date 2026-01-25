from __future__ import annotations

from typing import AsyncIterator, Protocol

from app.types import ChatRequest, ModelSpec, StreamEvent


class Provider(Protocol):
    name: str

    def get_models(self) -> list[ModelSpec]:
        ...

    async def stream_reply(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        ...
