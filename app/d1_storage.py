from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from app.types import (
    AllowlistEntry,
    ChatSettings,
    ContentPart,
    ImageRef,
    IncomingMessage,
    PendingMessage,
    StoredConversation,
    StoredMessage,
    build_content_parts,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id INTEGER PRIMARY KEY,
    enabled INTEGER NOT NULL,
    reply_mode TEXT NOT NULL,
    default_model_alias TEXT NOT NULL,
    skip_prefix TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    model_alias TEXT NOT NULL,
    system_prompt_override TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_chat_user
ON conversations (chat_id, user_id, id DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    telegram_message_id INTEGER,
    message_type TEXT NOT NULL,
    parent_message_id INTEGER,
    provider TEXT,
    model_id TEXT,
    model_alias TEXT,
    content_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON messages (conversation_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_messages_chat_created
ON messages (chat_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_messages_chat_telegram
ON messages (chat_id, telegram_message_id);

CREATE TABLE IF NOT EXISTS telegram_message_links (
    chat_id INTEGER NOT NULL,
    telegram_message_id INTEGER NOT NULL,
    logical_message_id INTEGER NOT NULL,
    part_index INTEGER NOT NULL,
    PRIMARY KEY (chat_id, telegram_message_id)
);

CREATE TABLE IF NOT EXISTS pending_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    telegram_message_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_messages_conversation
ON pending_messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS allowlist_entries (
    kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, target_id)
);

CREATE TABLE IF NOT EXISTS assistant_render_states (
    assistant_message_id INTEGER PRIMARY KEY,
    phase TEXT,
    final_status TEXT,
    reply_text TEXT,
    reasoning_json TEXT,
    render_markdown INTEGER
);

CREATE TABLE IF NOT EXISTS blobs (
    sha256 TEXT PRIMARY KEY,
    mime_type TEXT NOT NULL,
    data BLOB NOT NULL,
    size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

@dataclass(frozen=True)
class InboxUpdate:
    update_id: int
    chat_id: int
    user_id: int
    message_id: int
    media_group_key: str | None
    state: str
    received_at: str
    claimed_at: str | None
    completed_at: str | None
    reply_started_at: str | None
    reply_sent_at: str | None
    realized_user_message_id: int | None
    realized_assistant_message_id: int | None
    assistant_render_phase: str | None
    assistant_render_final_status: str | None
    assistant_render_reply_text: str | None
    assistant_render_reasoning_blocks: tuple[str, ...]
    assistant_render_markdown: bool | None
    message: IncomingMessage


@dataclass(frozen=True)
class AssistantRenderState:
    assistant_message_id: int
    phase: str | None
    final_status: str | None
    reply_text: str | None
    reasoning_blocks: tuple[str, ...]
    render_markdown: bool | None


class D1SettingsStore:
    def __init__(
        self,
        db,
        *,
        default_model_alias: str,
        default_reply_mode: str,
        default_skip_prefix: str,
    ) -> None:
        self._db = db
        self.default_model_alias = default_model_alias
        self.default_reply_mode = default_reply_mode
        self.default_skip_prefix = default_skip_prefix

    async def get_chat_settings(self, chat_id: int) -> ChatSettings:
        result = await self._db.prepare(
            "SELECT chat_id, enabled, reply_mode, default_model_alias, skip_prefix FROM chat_settings WHERE chat_id = ?"
        ).bind(chat_id).first()
        if result is None:
            now = utcnow()
            await self._db.prepare(
                "INSERT INTO chat_settings (chat_id, enabled, reply_mode, default_model_alias, skip_prefix, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ).bind(chat_id, 1, self.default_reply_mode, self.default_model_alias, self.default_skip_prefix, now, now).run()
            return ChatSettings(
                chat_id=chat_id,
                enabled=True,
                reply_mode=self.default_reply_mode,
                default_model_alias=self.default_model_alias,
                skip_prefix=self.default_skip_prefix,
            )
        return ChatSettings(
            chat_id=result["chat_id"],
            enabled=bool(result["enabled"]),
            reply_mode=result["reply_mode"],
            default_model_alias=result["default_model_alias"],
            skip_prefix=result["skip_prefix"],
        )

    async def set_default_model_alias(self, chat_id: int, alias: str) -> None:
        await self.get_chat_settings(chat_id)
        await self._db.prepare(
            "UPDATE chat_settings SET default_model_alias = ?, updated_at = ? WHERE chat_id = ?"
        ).bind(alias, utcnow(), chat_id).run()

    async def set_reply_mode(self, chat_id: int, reply_mode: str) -> None:
        await self.get_chat_settings(chat_id)
        await self._db.prepare(
            "UPDATE chat_settings SET reply_mode = ?, updated_at = ? WHERE chat_id = ?"
        ).bind(reply_mode, utcnow(), chat_id).run()

    async def toggle_allowlist_entry(self, *, kind: str, target_id: int) -> bool:
        self._validate_allowlist_kind(kind)
        result = await self._db.prepare(
            "SELECT 1 FROM allowlist_entries WHERE kind = ? AND target_id = ?"
        ).bind(kind, target_id).first()
        if result is None:
            await self._db.prepare(
                "INSERT INTO allowlist_entries (kind, target_id, created_at) VALUES (?, ?, ?)"
            ).bind(kind, target_id, utcnow()).run()
            return True
        await self._db.prepare(
            "DELETE FROM allowlist_entries WHERE kind = ? AND target_id = ?"
        ).bind(kind, target_id).run()
        return False

    async def list_allowlist_entries(self) -> list[AllowlistEntry]:
        result = await self._db.prepare(
            "SELECT kind, target_id, created_at FROM allowlist_entries ORDER BY kind ASC, target_id ASC"
        ).all()
        return [
            AllowlistEntry(kind=row["kind"], target_id=row["target_id"], created_at=row["created_at"])
            for row in result.results
        ]

    async def is_reply_allowed(self, *, chat_id: int, user_id: int) -> bool:
        result = await self._db.prepare(
            "SELECT 1 FROM allowlist_entries WHERE (kind = 'chat' AND target_id = ?) OR (kind = 'user' AND target_id = ?) LIMIT 1"
        ).bind(chat_id, user_id).first()
        return result is not None

    def _validate_allowlist_kind(self, kind: str) -> None:
        if kind not in {"chat", "user"}:
            raise ValueError(f"Invalid allowlist kind: {kind}")


class D1ConversationStore:
    def __init__(self, db) -> None:
        self._db = db

    async def create_conversation(
        self,
        *,
        chat_id: int,
        user_id: int,
        model_alias: str,
        system_prompt_override: str | None = None,
    ) -> StoredConversation:
        now = utcnow()
        result = await self._db.prepare(
            "INSERT INTO conversations (chat_id, user_id, model_alias, system_prompt_override, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) RETURNING id"
        ).bind(chat_id, user_id, model_alias, system_prompt_override, now, now).first()
        conversation_id = result["id"]
        conversation = await self.get_conversation(conversation_id)
        assert conversation is not None
        return conversation

    async def get_conversation(self, conversation_id: int) -> StoredConversation | None:
        result = await self._db.prepare(
            "SELECT * FROM conversations WHERE id = ?"
        ).bind(conversation_id).first()
        if result is None:
            return None
        return self._row_to_conversation(result)

    async def delete_conversation_if_empty(self, conversation_id: int) -> None:
        await self._db.prepare(
            "DELETE FROM conversations WHERE id = ? AND NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = ?) AND NOT EXISTS (SELECT 1 FROM pending_messages WHERE conversation_id = ?)"
        ).bind(conversation_id, conversation_id, conversation_id).run()

    async def find_recent_state_message(self, *, chat_id: int, user_id: int, not_before: str) -> StoredMessage | None:
        result = await self._db.prepare(
            """SELECT m.*
            FROM conversations c
            JOIN messages m ON m.id = (
                SELECT tip.id
                FROM messages tip
                WHERE tip.conversation_id = c.id
                ORDER BY tip.id DESC
                LIMIT 1
            )
            WHERE c.chat_id = ?
              AND c.user_id = ?
              AND (
                  c.updated_at >= ?
                  OR (m.message_type = 'assistant' AND m.status = 'streaming')
              )
            ORDER BY c.updated_at DESC, m.id DESC
            LIMIT 1"""
        ).bind(chat_id, user_id, not_before).first()
        if result is None:
            return None
        return self._row_to_message(result)

    async def create_message(
        self,
        *,
        conversation_id: int,
        chat_id: int,
        telegram_message_id: int | None,
        message_type: str,
        parent_message_id: int | None,
        provider: str | None,
        model_id: str | None,
        model_alias: str | None,
        content: str,
        status: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] | None = None,
        created_at: str | None = None,
    ) -> int:
        timestamp = created_at or utcnow()
        result = await self._db.prepare(
            "INSERT INTO messages (conversation_id, chat_id, telegram_message_id, message_type, parent_message_id, provider, model_id, model_alias, content_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id"
        ).bind(
            conversation_id,
            chat_id,
            telegram_message_id,
            message_type,
            parent_message_id,
            provider,
            model_id,
            model_alias,
            json.dumps(self._serialize_content_payload(content=content, images=images, parts=parts)),
            status,
            timestamp,
        ).first()
        await self._touch_conversation(conversation_id, timestamp=timestamp)
        return result["id"]

    async def update_message(self, message_id: int, *, content: str, status: str) -> None:
        current = await self.get_message(message_id)
        preserved_images = current.images if current is not None else ()
        preserved_parts = current.parts if current is not None else None
        await self._db.prepare(
            "UPDATE messages SET content_json = ?, status = ? WHERE id = ?"
        ).bind(
            json.dumps(self._serialize_content_payload(content=content, images=preserved_images, parts=preserved_parts)),
            status,
            message_id,
        ).run()
        if current is not None:
            await self._touch_conversation(current.conversation_id)

    async def link_telegram_message(
        self,
        *,
        chat_id: int,
        telegram_message_id: int,
        logical_message_id: int,
        part_index: int,
    ) -> None:
        await self._db.prepare(
            "INSERT OR REPLACE INTO telegram_message_links (chat_id, telegram_message_id, logical_message_id, part_index) VALUES (?, ?, ?, ?)"
        ).bind(chat_id, telegram_message_id, logical_message_id, part_index).run()

    async def list_linked_telegram_message_ids(self, *, logical_message_id: int) -> list[int]:
        result = await self._db.prepare(
            "SELECT telegram_message_id FROM telegram_message_links WHERE logical_message_id = ? ORDER BY part_index ASC, telegram_message_id ASC"
        ).bind(logical_message_id).all()
        return [int(row["telegram_message_id"]) for row in result.results]

    async def get_message_by_telegram(self, *, chat_id: int, telegram_message_id: int) -> StoredMessage | None:
        result = await self._db.prepare(
            "SELECT * FROM messages WHERE chat_id = ? AND telegram_message_id = ? ORDER BY id DESC LIMIT 1"
        ).bind(chat_id, telegram_message_id).first()
        if result is not None:
            return self._row_to_message(result)
        result = await self._db.prepare(
            """SELECT m.*
            FROM telegram_message_links l
            JOIN messages m ON m.id = l.logical_message_id
            WHERE l.chat_id = ? AND l.telegram_message_id = ?"""
        ).bind(chat_id, telegram_message_id).first()
        if result is not None:
            return self._row_to_message(result)
        return None

    async def get_message(self, message_id: int) -> StoredMessage | None:
        result = await self._db.prepare(
            "SELECT * FROM messages WHERE id = ?"
        ).bind(message_id).first()
        if result is None:
            return None
        return self._row_to_message(result)

    async def get_latest_message(self, conversation_id: int) -> StoredMessage | None:
        result = await self._db.prepare(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 1"
        ).bind(conversation_id).first()
        if result is None:
            return None
        return self._row_to_message(result)

    async def list_streaming_assistant_messages(self) -> list[StoredMessage]:
        result = await self._db.prepare(
            "SELECT * FROM messages WHERE message_type = 'assistant' AND status = 'streaming' ORDER BY id ASC"
        ).all()
        return [self._row_to_message(row) for row in result.results]

    async def get_conversation_tip_message(self, conversation_id: int) -> StoredMessage | None:
        return await self.get_latest_message(conversation_id)

    async def is_conversation_streaming(self, conversation_id: int) -> bool:
        tip = await self.get_conversation_tip_message(conversation_id)
        return tip is not None and tip.message_type == "assistant" and tip.status == "streaming"

    async def find_pending_message_by_telegram(self, *, chat_id: int, telegram_message_id: int) -> PendingMessage | None:
        result = await self._db.prepare(
            """SELECT p.*
            FROM pending_messages p
            JOIN conversations c ON c.id = p.conversation_id
            WHERE c.chat_id = ?
            ORDER BY p.id DESC"""
        ).bind(chat_id).all()
        for row in result.results:
            pending_message = self._row_to_pending_message(row)
            if pending_message.telegram_message_id == telegram_message_id:
                return pending_message
            if telegram_message_id in pending_message.source_telegram_message_ids:
                return pending_message
        return None

    async def list_pending_messages(self, *, conversation_id: int) -> list[PendingMessage]:
        result = await self._db.prepare(
            "SELECT * FROM pending_messages WHERE conversation_id = ? ORDER BY id ASC"
        ).bind(conversation_id).all()
        return [self._row_to_pending_message(row) for row in result.results]

    async def build_thread(self, message_id: int) -> list[StoredMessage]:
        thread: list[StoredMessage] = []
        current_id = message_id
        while current_id is not None:
            message = await self.get_message(current_id)
            if message is None:
                break
            thread.append(message)
            current_id = message.parent_message_id
        thread.reverse()
        return thread

    async def enqueue_pending_message(
        self,
        *,
        conversation_id: int,
        telegram_message_id: int,
        source_telegram_message_ids: tuple[int, ...] = (),
        content: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] | None = None,
    ) -> int:
        result = await self._db.prepare(
            "INSERT INTO pending_messages (conversation_id, telegram_message_id, content, created_at) VALUES (?, ?, ?, ?) RETURNING id"
        ).bind(
            conversation_id,
            telegram_message_id,
            json.dumps(self._serialize_content_payload(content=content, images=images, parts=parts, source_telegram_message_ids=source_telegram_message_ids)),
            utcnow(),
        ).first()
        await self._touch_conversation(conversation_id)
        return result["id"]

    async def update_pending_message(
        self,
        pending_message_id: int,
        *,
        telegram_message_id: int,
        source_telegram_message_ids: tuple[int, ...] = (),
        content: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] | None = None,
    ) -> None:
        await self._db.prepare(
            "UPDATE pending_messages SET telegram_message_id = ?, content = ? WHERE id = ?"
        ).bind(
            telegram_message_id,
            json.dumps(self._serialize_content_payload(content=content, images=images, parts=parts, source_telegram_message_ids=source_telegram_message_ids)),
            pending_message_id,
        ).run()
        result = await self._db.prepare(
            "SELECT conversation_id FROM pending_messages WHERE id = ?"
        ).bind(pending_message_id).first()
        if result is not None:
            await self._touch_conversation(result["conversation_id"])

    async def drain_pending_messages(self, *, conversation_id: int) -> list[PendingMessage]:
        result = await self._db.prepare(
            "SELECT * FROM pending_messages WHERE conversation_id = ? ORDER BY id ASC"
        ).bind(conversation_id).all()
        if result.results:
            await self._db.prepare(
                "DELETE FROM pending_messages WHERE conversation_id = ?"
            ).bind(conversation_id).run()
        return [self._row_to_pending_message(row) for row in result.results]

    async def delete_pending_messages(self, *, pending_message_ids: tuple[int, ...]) -> None:
        if not pending_message_ids:
            return
        placeholders = ", ".join("?" for _ in pending_message_ids)
        await self._db.prepare(
            f"DELETE FROM pending_messages WHERE id IN ({placeholders})"
        ).bind(*pending_message_ids).run()

    def _serialize_content_payload(
        self,
        *,
        content: str,
        images: tuple[ImageRef, ...],
        parts: tuple[ContentPart, ...] | None,
        source_telegram_message_ids: tuple[int, ...] = (),
    ) -> dict[str, object]:
        payload: dict[str, object] = {"text": content}
        if images:
            payload["images"] = [self._serialize_image(image) for image in images]
        if source_telegram_message_ids:
            payload["source_telegram_message_ids"] = [
                telegram_message_id
                for telegram_message_id in source_telegram_message_ids
                if telegram_message_id > 0
            ]
        effective_parts = parts if parts is not None else build_content_parts(content, images)
        if effective_parts:
            payload["parts"] = [self._serialize_part(part) for part in effective_parts]
        return payload

    async def _touch_conversation(self, conversation_id: int, *, timestamp: str | None = None) -> None:
        await self._db.prepare(
            "UPDATE conversations SET updated_at = ? WHERE id = ?"
        ).bind(timestamp or utcnow(), conversation_id).run()

    def _deserialize_content_payload(self, raw: str) -> dict[str, object]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Content payload must be a JSON object")
        return payload

    def _row_to_message(self, row) -> StoredMessage:
        payload = self._deserialize_content_payload(row["content_json"])
        return StoredMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            chat_id=row["chat_id"],
            telegram_message_id=row["telegram_message_id"],
            message_type=row["message_type"],
            parent_message_id=row["parent_message_id"],
            provider=row["provider"],
            model_id=row["model_id"],
            model_alias=row["model_alias"],
            content=payload.get("text", ""),
            status=row["status"],
            created_at=row["created_at"],
            raw_content=payload,
            images=self._images_from_payload(payload),
            parts=self._parts_from_payload(payload),
        )

    def _row_to_conversation(self, row) -> StoredConversation:
        return StoredConversation(
            id=row["id"],
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            model_alias=row["model_alias"],
            system_prompt_override=row["system_prompt_override"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_pending_message(self, row) -> PendingMessage:
        payload = self._deserialize_content_payload(row["content"])
        return PendingMessage(
            id=row["id"],
            conversation_id=row["conversation_id"],
            telegram_message_id=row["telegram_message_id"],
            content=payload.get("text", ""),
            created_at=row["created_at"],
            source_telegram_message_ids=self._source_telegram_message_ids_from_payload(payload),
            raw_content=payload,
            images=self._images_from_payload(payload),
            parts=self._parts_from_payload(payload),
        )

    def _images_from_payload(self, payload: dict[str, object]) -> tuple[ImageRef, ...]:
        return self._images_from_raw_list(payload.get("images"))

    def _parts_from_payload(self, payload: dict[str, object]) -> tuple[ContentPart, ...]:
        raw_parts = payload.get("parts")
        if not isinstance(raw_parts, list):
            return ()
        parts: list[ContentPart] = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            kind = raw_part.get("kind")
            if kind == "text":
                text = raw_part.get("text")
                if isinstance(text, str):
                    parts.append(ContentPart(kind="text", text=text))
            elif kind == "image":
                image = self._image_from_raw(raw_part)
                if image is not None:
                    parts.append(ContentPart(kind="image", image=image))
        return tuple(parts)

    def _serialize_image(self, image: ImageRef) -> dict[str, object]:
        if image.kind == "telegram":
            if image.file_id is None:
                raise ValueError("Telegram image refs require file_id")
            payload: dict[str, object] = {"mime_type": image.mime_type, "file_id": image.file_id}
            if image.file_size is not None:
                payload["file_size"] = image.file_size
            return payload
        if image.kind == "stored":
            assert image.blob_path is not None
            assert image.sha256 is not None
            assert image.size is not None
            return {"mime_type": image.mime_type, "blob_path": image.blob_path, "sha256": image.sha256, "size": image.size}
        raise ValueError("Loaded image refs cannot be serialized")

    def _serialize_part(self, part: ContentPart) -> dict[str, object]:
        if part.kind == "text":
            return {"kind": "text", "text": part.text}
        assert part.image is not None
        return {"kind": "image", **self._serialize_image(part.image)}

    def _images_from_raw_list(self, raw_images: object) -> tuple[ImageRef, ...]:
        if not isinstance(raw_images, list):
            return ()
        images: list[ImageRef] = []
        for raw_image in raw_images:
            image = self._image_from_raw(raw_image)
            if image is not None:
                images.append(image)
        return tuple(images)

    def _image_from_raw(self, raw_image: object) -> ImageRef | None:
        if not isinstance(raw_image, dict):
            return None
        file_id = raw_image.get("file_id")
        blob_path = raw_image.get("blob_path")
        mime_type = raw_image.get("mime_type")
        sha256 = raw_image.get("sha256")
        size = raw_image.get("size")
        if isinstance(file_id, str) and isinstance(mime_type, str):
            file_size = raw_image.get("file_size")
            return ImageRef.telegram(
                file_id=file_id,
                mime_type=mime_type,
                file_size=file_size if isinstance(file_size, int) else None,
            )
        if not isinstance(blob_path, str) or not isinstance(mime_type, str) or not isinstance(sha256, str) or not isinstance(size, int):
            return None
        return ImageRef.stored(mime_type=mime_type, blob_path=blob_path, sha256=sha256, size=size)

    def _source_telegram_message_ids_from_payload(self, payload: dict[str, object]) -> tuple[int, ...]:
        raw_ids = payload.get("source_telegram_message_ids")
        if not isinstance(raw_ids, list):
            return ()
        source_ids: list[int] = []
        for raw_id in raw_ids:
            if isinstance(raw_id, int) and raw_id > 0:
                source_ids.append(raw_id)
        return tuple(source_ids)


class D1InboxStore:
    def __init__(self, db) -> None:
        self._db = db

    async def enqueue_messages(self, *, messages: list[IncomingMessage]) -> None:
        if not messages:
            return
        for message in messages:
            await self._db.prepare(
                "INSERT OR IGNORE INTO inbox_updates (update_id, chat_id, user_id, message_id, media_group_key, state, payload_json, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ).bind(
                message.update_id,
                message.chat_id,
                message.user_id,
                message.message_id,
                self._media_group_key(message),
                "queued",
                json.dumps(self._serialize_incoming_message(message)),
                utcnow(),
            ).run()

    async def has_queued_updates(self) -> bool:
        result = await self._db.prepare(
            "SELECT 1 FROM inbox_updates WHERE state = 'queued' LIMIT 1"
        ).first()
        return result is not None

    async def get_update(self, *, update_id: int) -> InboxUpdate | None:
        result = await self._db.prepare(
            "SELECT * FROM inbox_updates WHERE update_id = ?"
        ).bind(update_id).first()
        if result is None:
            return None
        return self._row_to_inbox_update(result)

    async def list_updates(self, *, state: str | None = None) -> list[InboxUpdate]:
        if state is None:
            result = await self._db.prepare(
                "SELECT * FROM inbox_updates ORDER BY update_id ASC"
            ).all()
        else:
            result = await self._db.prepare(
                "SELECT * FROM inbox_updates WHERE state = ? ORDER BY update_id ASC"
            ).bind(state).all()
        return [self._row_to_inbox_update(row) for row in result.results]

    async def claim_next_ready(
        self,
        *,
        media_group_delay_seconds: float,
        media_group_boundary_grace_seconds: float = 0.0,
    ) -> list[InboxUpdate] | None:
        result = await self._db.prepare(
            "SELECT * FROM inbox_updates WHERE state = 'queued' ORDER BY update_id ASC"
        ).all()
        if not result.results:
            return None

        now = datetime.now(timezone.utc)
        seen_media_group_keys: set[str] = set()
        for row in result.results:
            entry = self._row_to_inbox_update(row)
            if entry.media_group_key is None:
                claimed_at = utcnow()
                update_result = await self._db.prepare(
                    "UPDATE inbox_updates SET state = 'claimed', claimed_at = ? WHERE update_id = ? AND state = 'queued'"
                ).bind(claimed_at, entry.update_id).run()
                if update_result.meta.changes > 0:
                    return [self._replace_state(entry, state="claimed", claimed_at=claimed_at)]
                continue

            if entry.media_group_key in seen_media_group_keys:
                continue
            seen_media_group_keys.add(entry.media_group_key)
            claimed_group_row = await self._db.prepare(
                "SELECT 1 FROM inbox_updates WHERE state = 'claimed' AND media_group_key = ? LIMIT 1"
            ).bind(entry.media_group_key).first()
            if claimed_group_row is not None:
                continue
            group_result = await self._db.prepare(
                "SELECT * FROM inbox_updates WHERE state = 'queued' AND media_group_key = ? ORDER BY update_id ASC"
            ).bind(entry.media_group_key).all()
            if not group_result.results:
                continue
            latest_received_at = max(self._parse_timestamp(group_row["received_at"]) for group_row in group_result.results)
            ready_after_seconds = media_group_delay_seconds + media_group_boundary_grace_seconds
            if (now - latest_received_at).total_seconds() < ready_after_seconds:
                continue

            claimed_at = utcnow()
            await self._db.prepare(
                "UPDATE inbox_updates SET state = 'claimed', claimed_at = ? WHERE state = 'queued' AND media_group_key = ?"
            ).bind(claimed_at, entry.media_group_key).run()
            return [
                self._replace_state(self._row_to_inbox_update(group_row), state="claimed", claimed_at=claimed_at)
                for group_row in group_result.results
            ]
        return None

    async def claim_queued_media_group_siblings(self, *, media_group_key: str) -> list[InboxUpdate]:
        result = await self._db.prepare(
            "SELECT * FROM inbox_updates WHERE state = 'queued' AND media_group_key = ? ORDER BY update_id ASC"
        ).bind(media_group_key).all()
        if not result.results:
            return []
        claimed_at = utcnow()
        await self._db.prepare(
            "UPDATE inbox_updates SET state = 'claimed', claimed_at = ? WHERE state = 'queued' AND media_group_key = ?"
        ).bind(claimed_at, media_group_key).run()
        return [
            self._replace_state(self._row_to_inbox_update(group_row), state="claimed", claimed_at=claimed_at)
            for group_row in result.results
        ]

    async def mark_updates_realized(
        self,
        *,
        update_ids: tuple[int, ...],
        user_message_id: int | None,
        assistant_message_id: int | None,
    ) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET realized_user_message_id = ?, realized_assistant_message_id = ? WHERE update_id IN ({placeholders})"
        ).bind(user_message_id, assistant_message_id, *update_ids).run()

    async def mark_reply_sent(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        reply_sent_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET reply_sent_at = COALESCE(reply_sent_at, ?) WHERE update_id IN ({placeholders})"
        ).bind(reply_sent_at, *update_ids).run()

    async def mark_reply_started(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        reply_started_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET reply_started_at = COALESCE(reply_started_at, ?) WHERE update_id IN ({placeholders})"
        ).bind(reply_started_at, *update_ids).run()

    async def clear_reply_sent(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET reply_sent_at = NULL WHERE update_id IN ({placeholders})"
        ).bind(*update_ids).run()

    async def clear_reply_started(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET reply_started_at = NULL WHERE update_id IN ({placeholders})"
        ).bind(*update_ids).run()

    async def set_assistant_render_state(
        self,
        *,
        assistant_message_id: int,
        phase: str | None,
        final_status: str | None,
        reply_text: str | None,
        reasoning_blocks: tuple[str, ...],
        render_markdown: bool | None,
    ) -> None:
        serialized_reasoning = json.dumps(list(reasoning_blocks)) if reasoning_blocks else None
        serialized_render_markdown = None if render_markdown is None else int(render_markdown)
        if (
            phase is None
            and final_status is None
            and reply_text is None
            and not reasoning_blocks
            and render_markdown is None
        ):
            await self._db.prepare(
                "DELETE FROM assistant_render_states WHERE assistant_message_id = ?"
            ).bind(assistant_message_id).run()
        else:
            await self._db.prepare(
                "INSERT INTO assistant_render_states (assistant_message_id, phase, final_status, reply_text, reasoning_json, render_markdown) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(assistant_message_id) DO UPDATE SET phase = excluded.phase, final_status = excluded.final_status, reply_text = excluded.reply_text, reasoning_json = excluded.reasoning_json, render_markdown = excluded.render_markdown"
            ).bind(assistant_message_id, phase, final_status, reply_text, serialized_reasoning, serialized_render_markdown).run()
        await self._db.prepare(
            "UPDATE inbox_updates SET assistant_render_phase = ?, assistant_render_final_status = ?, assistant_render_reply_text = ?, assistant_render_reasoning_json = ?, assistant_render_markdown = ? WHERE realized_assistant_message_id = ?"
        ).bind(phase, final_status, reply_text, serialized_reasoning, serialized_render_markdown, assistant_message_id).run()

    async def get_assistant_render_state(self, *, assistant_message_id: int) -> AssistantRenderState | None:
        result = await self._db.prepare(
            "SELECT * FROM assistant_render_states WHERE assistant_message_id = ?"
        ).bind(assistant_message_id).first()
        if result is None:
            return None
        reasoning_blocks: tuple[str, ...] = ()
        raw_reasoning = result["reasoning_json"]
        if isinstance(raw_reasoning, str) and raw_reasoning:
            parsed_reasoning = json.loads(raw_reasoning)
            if isinstance(parsed_reasoning, list):
                reasoning_blocks = tuple(str(item) for item in parsed_reasoning if isinstance(item, str))
        return AssistantRenderState(
            assistant_message_id=result["assistant_message_id"],
            phase=result["phase"],
            final_status=result["final_status"],
            reply_text=result["reply_text"],
            reasoning_blocks=reasoning_blocks,
            render_markdown=None if result["render_markdown"] is None else bool(result["render_markdown"]),
        )

    async def complete_updates(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        completed_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET state = 'completed', completed_at = ? WHERE update_id IN ({placeholders})"
        ).bind(completed_at, *update_ids).run()

    async def reset_updates_to_queued(self, *, update_ids: tuple[int, ...]) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        await self._db.prepare(
            f"UPDATE inbox_updates SET state = 'queued', claimed_at = NULL, completed_at = NULL, reply_started_at = NULL, reply_sent_at = NULL, realized_user_message_id = NULL, realized_assistant_message_id = NULL, assistant_render_phase = NULL, assistant_render_final_status = NULL, assistant_render_reply_text = NULL, assistant_render_reasoning_json = NULL, assistant_render_markdown = NULL WHERE update_id IN ({placeholders})"
        ).bind(*update_ids).run()

    @staticmethod
    def _media_group_key(message: IncomingMessage) -> str | None:
        if message.media_group_id is None:
            return None
        return f"{message.chat_id}:{message.user_id}:{message.media_group_id}"

    def _serialize_incoming_message(self, message: IncomingMessage) -> dict[str, object]:
        return {
            "update_id": message.update_id,
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "user_id": message.user_id,
            "chat_type": message.chat_type,
            "text": message.text,
            "from_bot": message.from_bot,
            "mentions_bot": message.mentions_bot,
            "source_message_ids": list(message.source_message_ids),
            "reply_to_message_id": message.reply_to_message_id,
            "reply_to_user_id": message.reply_to_user_id,
            "reply_to_bot": message.reply_to_bot,
            "reply_to_text": message.reply_to_text,
            "images": [self._serialize_image(image) for image in message.images],
            "parts": [self._serialize_part(part) for part in message.parts],
            "media_group_id": message.media_group_id,
        }

    def _row_to_inbox_update(self, row) -> InboxUpdate:
        payload = json.loads(row["payload_json"])
        reasoning_blocks: tuple[str, ...] = ()
        raw_reasoning = row["assistant_render_reasoning_json"]
        if isinstance(raw_reasoning, str) and raw_reasoning:
            parsed_reasoning = json.loads(raw_reasoning)
            if isinstance(parsed_reasoning, list):
                reasoning_blocks = tuple(str(item) for item in parsed_reasoning if isinstance(item, str))
        return InboxUpdate(
            update_id=row["update_id"],
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            message_id=row["message_id"],
            media_group_key=row["media_group_key"],
            state=row["state"],
            received_at=row["received_at"],
            claimed_at=row["claimed_at"],
            completed_at=row["completed_at"],
            reply_started_at=row["reply_started_at"],
            reply_sent_at=row["reply_sent_at"],
            realized_user_message_id=row["realized_user_message_id"],
            realized_assistant_message_id=row["realized_assistant_message_id"],
            assistant_render_phase=row["assistant_render_phase"],
            assistant_render_final_status=row["assistant_render_final_status"],
            assistant_render_reply_text=row["assistant_render_reply_text"],
            assistant_render_reasoning_blocks=reasoning_blocks,
            assistant_render_markdown=None if row["assistant_render_markdown"] is None else bool(row["assistant_render_markdown"]),
            message=self._deserialize_incoming_message(payload),
        )

    def _deserialize_incoming_message(self, payload: dict[str, object]) -> IncomingMessage:
        source_message_ids = payload.get("source_message_ids")
        images = payload.get("images")
        parts = payload.get("parts")
        return IncomingMessage(
            update_id=int(payload["update_id"]),
            chat_id=int(payload["chat_id"]),
            message_id=int(payload["message_id"]),
            user_id=int(payload["user_id"]),
            chat_type=str(payload.get("chat_type") or ""),
            text=str(payload.get("text") or ""),
            from_bot=bool(payload.get("from_bot")),
            mentions_bot=bool(payload.get("mentions_bot")),
            source_message_ids=tuple(int(item) for item in source_message_ids) if isinstance(source_message_ids, list) else (),
            reply_to_message_id=int(payload["reply_to_message_id"]) if payload.get("reply_to_message_id") is not None else None,
            reply_to_user_id=int(payload["reply_to_user_id"]) if payload.get("reply_to_user_id") is not None else None,
            reply_to_bot=bool(payload.get("reply_to_bot")),
            reply_to_text=str(payload["reply_to_text"]) if payload.get("reply_to_text") is not None else None,
            images=self._deserialize_images(images),
            parts=self._deserialize_parts(parts),
            media_group_id=str(payload["media_group_id"]) if payload.get("media_group_id") is not None else None,
        )

    def _deserialize_images(self, raw_images: object) -> tuple[ImageRef, ...]:
        if not isinstance(raw_images, list):
            return ()
        images: list[ImageRef] = []
        for raw_image in raw_images:
            image = self._deserialize_image(raw_image)
            if image is not None:
                images.append(image)
        return tuple(images)

    def _deserialize_parts(self, raw_parts: object) -> tuple[ContentPart, ...]:
        if not isinstance(raw_parts, list):
            return ()
        parts: list[ContentPart] = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            kind = raw_part.get("kind")
            if kind == "text":
                text = raw_part.get("text")
                if isinstance(text, str):
                    parts.append(ContentPart(kind="text", text=text))
                continue
            if kind != "image":
                continue
            image = self._deserialize_image(raw_part)
            if image is not None:
                parts.append(ContentPart(kind="image", image=image))
        return tuple(parts)

    def _serialize_part(self, part: ContentPart) -> dict[str, object]:
        if part.kind == "text":
            return {"kind": "text", "text": part.text}
        assert part.image is not None
        return {"kind": "image", **self._serialize_image(part.image)}

    def _serialize_image(self, image: ImageRef) -> dict[str, object]:
        if image.kind == "telegram":
            payload: dict[str, object] = {"mime_type": image.mime_type, "file_id": image.file_id}
            if image.file_size is not None:
                payload["file_size"] = image.file_size
            return payload
        if image.kind == "stored":
            return {"mime_type": image.mime_type, "blob_path": image.blob_path, "sha256": image.sha256, "size": image.size}
        raise ValueError("Inbox messages cannot serialize loaded image refs")

    def _deserialize_image(self, raw_image: object) -> ImageRef | None:
        if not isinstance(raw_image, dict):
            return None
        file_id = raw_image.get("file_id")
        mime_type = raw_image.get("mime_type")
        if isinstance(file_id, str) and isinstance(mime_type, str):
            file_size = raw_image.get("file_size")
            return ImageRef.telegram(file_id=file_id, mime_type=mime_type, file_size=file_size if isinstance(file_size, int) else None)
        blob_path = raw_image.get("blob_path")
        sha256 = raw_image.get("sha256")
        size = raw_image.get("size")
        if not isinstance(blob_path, str) or not isinstance(sha256, str) or not isinstance(mime_type, str) or not isinstance(size, int):
            return None
        return ImageRef.stored(mime_type=mime_type, blob_path=blob_path, sha256=sha256, size=size)

    @staticmethod
    def _parse_timestamp(raw_timestamp: str) -> datetime:
        return datetime.fromisoformat(raw_timestamp)

    @staticmethod
    def _replace_state(entry: InboxUpdate, *, state: str, claimed_at: str | None) -> InboxUpdate:
        return InboxUpdate(
            update_id=entry.update_id,
            chat_id=entry.chat_id,
            user_id=entry.user_id,
            message_id=entry.message_id,
            media_group_key=entry.media_group_key,
            state=state,
            received_at=entry.received_at,
            claimed_at=claimed_at,
            completed_at=entry.completed_at,
            reply_started_at=entry.reply_started_at,
            reply_sent_at=entry.reply_sent_at,
            realized_user_message_id=entry.realized_user_message_id,
            realized_assistant_message_id=entry.realized_assistant_message_id,
            assistant_render_phase=entry.assistant_render_phase,
            assistant_render_final_status=entry.assistant_render_final_status,
            assistant_render_reply_text=entry.assistant_render_reply_text,
            assistant_render_reasoning_blocks=entry.assistant_render_reasoning_blocks,
            assistant_render_markdown=entry.assistant_render_markdown,
            message=entry.message,
        )


class D1BlobStore:
    """Content-addressed blob storage via D1 (replaces filesystem BlobStore)."""
    def __init__(self, db) -> None:
        self._db = db

    async def store_image(self, *, mime_type: str, data: bytes) -> ImageRef:
        sha256 = hashlib.sha256(data).hexdigest()
        existing = await self._db.prepare(
            "SELECT 1 FROM blobs WHERE sha256 = ?"
        ).bind(sha256).first()
        if existing is None:
            await self._db.prepare(
                "INSERT INTO blobs (sha256, mime_type, data, size, created_at) VALUES (?, ?, ?, ?, ?)"
            ).bind(sha256, mime_type, data, len(data), utcnow()).run()
        return ImageRef.stored(
            mime_type=mime_type,
            blob_path=sha256,
            sha256=sha256,
            size=len(data),
        )

    async def load_image_bytes(self, image: ImageRef) -> bytes:
        if image.sha256 is None:
            raise ValueError("Expected stored image ref with sha256")
        row = await self._db.prepare(
            "SELECT data FROM blobs WHERE sha256 = ?"
        ).bind(image.sha256).first()
        if row is None:
            raise ValueError(f"Blob not found: {image.sha256}")
        return row["data"]


class D1Storage:
    @staticmethod
    async def init_schema(db) -> None:
        await db.exec(SCHEMA)

    def __init__(
        self,
        db,
        default_model_alias: str,
        default_reply_mode: str,
        default_skip_prefix: str,
    ) -> None:
        self.settings = D1SettingsStore(
            db,
            default_model_alias=default_model_alias,
            default_reply_mode=default_reply_mode,
            default_skip_prefix=default_skip_prefix,
        )
        self.conversations = D1ConversationStore(db)
        self.inbox = D1InboxStore(db)
        self.blobs = D1BlobStore(db)

    async def list_referenced_model_aliases(self) -> list[str]:
        result = await self.settings._db.prepare(
            "SELECT DISTINCT model_alias FROM conversations"
        ).run()
        return [row["model_alias"] for row in result.results]
