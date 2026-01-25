from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ImageRefKind = Literal["telegram", "stored", "loaded"]
RequestedTool = Literal["search", "fetch"]


@dataclass(frozen=True)
class ImageRef:
    mime_type: str
    kind: ImageRefKind | str = ""
    file_id: str | None = None
    file_size: int | None = None
    blob_path: str | None = None
    sha256: str | None = None
    size: int | None = None
    data: bytes | None = None

    def __post_init__(self) -> None:
        inferred_kind = self.kind or self._infer_kind()
        if inferred_kind not in {"telegram", "stored", "loaded"}:
            raise ValueError(f"Unsupported image ref kind: {inferred_kind}")
        object.__setattr__(self, "kind", inferred_kind)

        if inferred_kind == "telegram" and not self.file_id:
            raise ValueError("Telegram image refs require file_id")
        if inferred_kind == "stored" and (not self.blob_path or not self.sha256 or self.size is None):
            raise ValueError("Stored image refs require blob_path, sha256, and size")
        if inferred_kind == "loaded" and self.data is None:
            raise ValueError("Loaded image refs require data")

    def _infer_kind(self) -> ImageRefKind:
        if self.data is not None:
            return "loaded"
        if self.blob_path is not None:
            return "stored"
        if self.file_id is not None:
            return "telegram"
        raise ValueError("Unable to infer image ref kind")

    @classmethod
    def telegram(cls, *, file_id: str, mime_type: str, file_size: int | None = None) -> "ImageRef":
        return cls(
            kind="telegram",
            file_id=file_id,
            mime_type=mime_type,
            file_size=file_size,
        )

    @classmethod
    def stored(
        cls,
        *,
        mime_type: str,
        blob_path: str,
        sha256: str,
        size: int,
    ) -> "ImageRef":
        return cls(
            kind="stored",
            mime_type=mime_type,
            blob_path=blob_path,
            sha256=sha256,
            size=size,
        )

    @classmethod
    def loaded(cls, *, mime_type: str, data: bytes) -> "ImageRef":
        return cls(
            kind="loaded",
            mime_type=mime_type,
            data=data,
        )


@dataclass(frozen=True)
class ContentPart:
    kind: Literal["text", "image"]
    text: str = ""
    image: ImageRef | None = None


def build_content_parts(content: str, images: tuple[ImageRef, ...]) -> tuple[ContentPart, ...]:
    parts: list[ContentPart] = []
    if content:
        parts.append(ContentPart(kind="text", text=content))
    for image in images:
        parts.append(ContentPart(kind="image", image=image))
    return tuple(parts)


@dataclass(frozen=True)
class ChatSettings:
    chat_id: int
    enabled: bool
    reply_mode: str
    default_model_alias: str
    skip_prefix: str


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    chat_id: int
    message_id: int
    user_id: int
    chat_type: str
    text: str
    from_bot: bool
    mentions_bot: bool
    source_message_ids: tuple[int, ...] = ()
    reply_to_message_id: int | None = None
    reply_to_user_id: int | None = None
    reply_to_bot: bool = False
    reply_to_text: str | None = None
    images: tuple[ImageRef, ...] = ()
    parts: tuple[ContentPart, ...] = ()
    media_group_id: str | None = None


@dataclass(frozen=True)
class IgnoreAction:
    reason: str


@dataclass(frozen=True)
class CommandAction:
    name: str
    argument: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class ChatAction:
    content: str
    intent: Literal["plain", "new", "choose_model", "set_system_prompt"] = "plain"
    model_alias: str | None = None
    system_prompt: str | None = None
    images: tuple[ImageRef, ...] = ()
    parts: tuple[ContentPart, ...] = ()


RouteAction = IgnoreAction | CommandAction | ChatAction


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    aliases: tuple[str, ...]
    supports_images: bool = False
    supports_files: bool = False
    supports_tools: bool = False
    supports_reasoning: bool = False
    supports_streaming: bool = True
    reasoning_effort: str | None = None
    thinking_mode: str | None = None
    thinking_budget_tokens: int | None = None
    output_effort: str | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    images: tuple[ImageRef, ...] = ()
    parts: tuple[ContentPart, ...] = ()


@dataclass(frozen=True)
class ChatRequest:
    model: ModelSpec
    conversation: list[ConversationMessage]
    system_prompt: str | None = None
    safety_identifier: str | None = None
    requested_tools: tuple[RequestedTool, ...] = ()


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class StoredMessage:
    id: int
    conversation_id: int
    chat_id: int
    telegram_message_id: int | None
    message_type: Literal["seed", "user", "assistant"]
    parent_message_id: int | None
    provider: str | None
    model_id: str | None
    model_alias: str | None
    content: str
    status: str
    created_at: str
    raw_content: dict[str, Any] = field(default_factory=dict)
    images: tuple[ImageRef, ...] = ()
    parts: tuple[ContentPart, ...] = ()


@dataclass(frozen=True)
class StoredConversation:
    id: int
    chat_id: int
    user_id: int
    model_alias: str
    system_prompt_override: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class PendingMessage:
    id: int
    conversation_id: int
    telegram_message_id: int
    content: str
    created_at: str
    source_telegram_message_ids: tuple[int, ...] = ()
    raw_content: dict[str, Any] = field(default_factory=dict)
    images: tuple[ImageRef, ...] = ()
    parts: tuple[ContentPart, ...] = ()


@dataclass(frozen=True)
class AllowlistEntry:
    kind: Literal["chat", "user"]
    target_id: int
    created_at: str
