from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

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


# Switchboard starts its public schema history here. Pre-release databases are not
# auto-migrated or auto-reset; incompatible files must be handled explicitly.
SCHEMA_VERSION = 1
COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION}

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

CREATE TABLE IF NOT EXISTS inbox_updates (
    update_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    media_group_key TEXT,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    received_at TEXT NOT NULL,
    claimed_at TEXT,
    completed_at TEXT,
    reply_started_at TEXT,
    reply_sent_at TEXT,
    realized_user_message_id INTEGER,
    realized_assistant_message_id INTEGER,
    assistant_render_phase TEXT,
    assistant_render_final_status TEXT,
    assistant_render_reply_text TEXT,
    assistant_render_reasoning_json TEXT,
    assistant_render_markdown INTEGER
);

CREATE INDEX IF NOT EXISTS idx_inbox_updates_state_update
ON inbox_updates (state, update_id);

CREATE INDEX IF NOT EXISTS idx_inbox_updates_media_group_state
ON inbox_updates (media_group_key, state, update_id);

CREATE TABLE IF NOT EXISTS assistant_render_states (
    assistant_message_id INTEGER PRIMARY KEY,
    phase TEXT,
    final_status TEXT,
    reply_text TEXT,
    reasoning_json TEXT,
    render_markdown INTEGER
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class SettingsStore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        default_model_alias: str,
        default_reply_mode: str,
        default_skip_prefix: str,
    ) -> None:
        self._conn = conn
        self.default_model_alias = default_model_alias
        self.default_reply_mode = default_reply_mode
        self.default_skip_prefix = default_skip_prefix

    def get_chat_settings(self, chat_id: int, *, commit: bool = True) -> ChatSettings:
        row = self._conn.execute(
            "SELECT chat_id, enabled, reply_mode, default_model_alias, skip_prefix FROM chat_settings WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row is None:
            now = utcnow()
            self._conn.execute(
                """
                INSERT INTO chat_settings (
                    chat_id, enabled, reply_mode, default_model_alias, skip_prefix, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, 1, self.default_reply_mode, self.default_model_alias, self.default_skip_prefix, now, now),
            )
            if commit:
                self._conn.commit()
            return ChatSettings(
                chat_id=chat_id,
                enabled=True,
                reply_mode=self.default_reply_mode,
                default_model_alias=self.default_model_alias,
                skip_prefix=self.default_skip_prefix,
            )
        return ChatSettings(
            chat_id=row["chat_id"],
            enabled=bool(row["enabled"]),
            reply_mode=row["reply_mode"],
            default_model_alias=row["default_model_alias"],
            skip_prefix=row["skip_prefix"],
        )

    def set_default_model_alias(self, chat_id: int, alias: str, *, commit: bool = True) -> None:
        self.get_chat_settings(chat_id, commit=commit)
        self._conn.execute(
            "UPDATE chat_settings SET default_model_alias = ?, updated_at = ? WHERE chat_id = ?",
            (alias, utcnow(), chat_id),
        )
        if commit:
            self._conn.commit()

    def set_reply_mode(self, chat_id: int, reply_mode: str, *, commit: bool = True) -> None:
        self.get_chat_settings(chat_id, commit=commit)
        self._conn.execute(
            "UPDATE chat_settings SET reply_mode = ?, updated_at = ? WHERE chat_id = ?",
            (reply_mode, utcnow(), chat_id),
        )
        if commit:
            self._conn.commit()

    def toggle_allowlist_entry(self, *, kind: str, target_id: int, commit: bool = True) -> bool:
        self._validate_allowlist_kind(kind)
        row = self._conn.execute(
            "SELECT 1 FROM allowlist_entries WHERE kind = ? AND target_id = ?",
            (kind, target_id),
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO allowlist_entries (kind, target_id, created_at) VALUES (?, ?, ?)",
                (kind, target_id, utcnow()),
            )
            if commit:
                self._conn.commit()
            return True

        self._conn.execute(
            "DELETE FROM allowlist_entries WHERE kind = ? AND target_id = ?",
            (kind, target_id),
        )
        if commit:
            self._conn.commit()
        return False

    def list_allowlist_entries(self) -> list[AllowlistEntry]:
        rows = self._conn.execute(
            "SELECT kind, target_id, created_at FROM allowlist_entries ORDER BY kind ASC, target_id ASC",
        ).fetchall()
        return [
            AllowlistEntry(kind=row["kind"], target_id=row["target_id"], created_at=row["created_at"])
            for row in rows
        ]

    def is_reply_allowed(self, *, chat_id: int, user_id: int) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM allowlist_entries
            WHERE (kind = 'chat' AND target_id = ?)
               OR (kind = 'user' AND target_id = ?)
            LIMIT 1
            """,
            (chat_id, user_id),
        ).fetchone()
        return row is not None

    def _validate_allowlist_kind(self, kind: str) -> None:
        if kind not in {"chat", "user"}:
            raise ValueError(f"Invalid allowlist kind: {kind}")


class BlobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def store_image(self, *, mime_type: str, data: bytes) -> ImageRef:
        sha256 = hashlib.sha256(data).hexdigest()
        relative_path = Path("images") / sha256[:2] / sha256[2:4] / sha256
        absolute_path = self.data_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        if not absolute_path.exists():
            absolute_path.write_bytes(data)
        return ImageRef.stored(
            mime_type=mime_type,
            blob_path=str(relative_path),
            sha256=sha256,
            size=len(data),
        )

    def load_image_bytes(self, image: ImageRef) -> bytes:
        if image.blob_path is None:
            raise ValueError("Expected stored image ref")
        return (self.data_dir / image.blob_path).read_bytes()


class ConversationStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create_conversation(
        self,
        *,
        chat_id: int,
        user_id: int,
        model_alias: str,
        system_prompt_override: str | None = None,
    ) -> StoredConversation:
        now = utcnow()
        cursor = self._conn.execute(
            """
            INSERT INTO conversations (
                chat_id, user_id, model_alias, system_prompt_override, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, user_id, model_alias, system_prompt_override, now, now),
        )
        self._conn.commit()
        conversation = self.get_conversation(int(cursor.lastrowid))
        assert conversation is not None
        return conversation

    def get_conversation(self, conversation_id: int) -> StoredConversation | None:
        row = self._conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return None if row is None else self._row_to_conversation(row)

    def delete_conversation_if_empty(self, conversation_id: int) -> None:
        self._conn.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM messages WHERE conversation_id = ?
              )
              AND NOT EXISTS (
                  SELECT 1 FROM pending_messages WHERE conversation_id = ?
              )
            """,
            (conversation_id, conversation_id, conversation_id),
        )
        self._conn.commit()

    def find_recent_state_message(self, *, chat_id: int, user_id: int, not_before: str) -> StoredMessage | None:
        row = self._conn.execute(
            """
            SELECT m.*
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
            LIMIT 1
            """,
            (chat_id, user_id, not_before),
        ).fetchone()
        return None if row is None else self._row_to_message(row)

    def create_message(
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
        commit: bool = True,
    ) -> int:
        timestamp = created_at or utcnow()
        cursor = self._conn.execute(
            """
            INSERT INTO messages (
                conversation_id, chat_id, telegram_message_id, message_type, parent_message_id, provider, model_id, model_alias, content_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
            ),
        )
        self._touch_conversation(conversation_id, timestamp=timestamp)
        if commit:
            self._conn.commit()
        return int(cursor.lastrowid)

    def update_message(self, message_id: int, *, content: str, status: str, commit: bool = True) -> None:
        current = self.get_message(message_id)
        preserved_images = current.images if current is not None else ()
        preserved_parts = current.parts if current is not None else None
        self._conn.execute(
            "UPDATE messages SET content_json = ?, status = ? WHERE id = ?",
            (
                json.dumps(
                    self._serialize_content_payload(
                        content=content,
                        images=preserved_images,
                        parts=preserved_parts,
                    )
                ),
                status,
                message_id,
            ),
        )
        if current is not None:
            self._touch_conversation(current.conversation_id)
        if commit:
            self._conn.commit()

    def link_telegram_message(
        self,
        *,
        chat_id: int,
        telegram_message_id: int,
        logical_message_id: int,
        part_index: int,
        commit: bool = True,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO telegram_message_links (
                chat_id, telegram_message_id, logical_message_id, part_index
            ) VALUES (?, ?, ?, ?)
            """,
            (chat_id, telegram_message_id, logical_message_id, part_index),
        )
        if commit:
            self._conn.commit()

    def list_linked_telegram_message_ids(self, *, logical_message_id: int) -> list[int]:
        rows = self._conn.execute(
            """
            SELECT telegram_message_id
            FROM telegram_message_links
            WHERE logical_message_id = ?
            ORDER BY part_index ASC, telegram_message_id ASC
            """,
            (logical_message_id,),
        ).fetchall()
        return [int(row["telegram_message_id"]) for row in rows]

    def get_message_by_telegram(self, *, chat_id: int, telegram_message_id: int) -> StoredMessage | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM messages
            WHERE chat_id = ? AND telegram_message_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id, telegram_message_id),
        ).fetchone()
        if row is not None:
            return self._row_to_message(row)

        row = self._conn.execute(
            """
            SELECT m.*
            FROM telegram_message_links l
            JOIN messages m ON m.id = l.logical_message_id
            WHERE l.chat_id = ? AND l.telegram_message_id = ?
            """,
            (chat_id, telegram_message_id),
        ).fetchone()
        return None if row is None else self._row_to_message(row)

    def get_message(self, message_id: int) -> StoredMessage | None:
        row = self._conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return None if row is None else self._row_to_message(row)

    def get_latest_message(self, conversation_id: int) -> StoredMessage | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        return None if row is None else self._row_to_message(row)

    def list_streaming_assistant_messages(self) -> list[StoredMessage]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM messages
            WHERE message_type = 'assistant' AND status = 'streaming'
            ORDER BY id ASC
            """,
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_conversation_tip_message(self, conversation_id: int) -> StoredMessage | None:
        return self.get_latest_message(conversation_id)

    def is_conversation_streaming(self, conversation_id: int) -> bool:
        tip = self.get_conversation_tip_message(conversation_id)
        return tip is not None and tip.message_type == "assistant" and tip.status == "streaming"

    def find_pending_message_by_telegram(self, *, chat_id: int, telegram_message_id: int) -> PendingMessage | None:
        rows = self._conn.execute(
            """
            SELECT p.*
            FROM pending_messages p
            JOIN conversations c ON c.id = p.conversation_id
            WHERE c.chat_id = ?
            ORDER BY p.id DESC
            """,
            (chat_id,),
        ).fetchall()
        for row in rows:
            pending_message = self._row_to_pending_message(row)
            if pending_message.telegram_message_id == telegram_message_id:
                return pending_message
            if telegram_message_id in pending_message.source_telegram_message_ids:
                return pending_message
        return None

    def list_pending_messages(self, *, conversation_id: int) -> list[PendingMessage]:
        rows = self._conn.execute(
            "SELECT * FROM pending_messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_pending_message(row) for row in rows]

    def build_thread(self, message_id: int) -> list[StoredMessage]:
        thread: list[StoredMessage] = []
        current_id = message_id
        while current_id is not None:
            message = self.get_message(current_id)
            if message is None:
                break
            thread.append(message)
            current_id = message.parent_message_id
        thread.reverse()
        return thread

    def enqueue_pending_message(
        self,
        *,
        conversation_id: int,
        telegram_message_id: int,
        source_telegram_message_ids: tuple[int, ...] = (),
        content: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] | None = None,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO pending_messages (conversation_id, telegram_message_id, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                telegram_message_id,
                json.dumps(
                    self._serialize_content_payload(
                        content=content,
                        images=images,
                        parts=parts,
                        source_telegram_message_ids=source_telegram_message_ids,
                    )
                ),
                utcnow(),
            ),
        )
        self._touch_conversation(conversation_id)
        self._conn.commit()
        return int(cursor.lastrowid)

    def update_pending_message(
        self,
        pending_message_id: int,
        *,
        telegram_message_id: int,
        source_telegram_message_ids: tuple[int, ...] = (),
        content: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] | None = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE pending_messages
            SET telegram_message_id = ?, content = ?
            WHERE id = ?
            """,
            (
                telegram_message_id,
                json.dumps(
                    self._serialize_content_payload(
                        content=content,
                        images=images,
                        parts=parts,
                        source_telegram_message_ids=source_telegram_message_ids,
                    )
                ),
                pending_message_id,
            ),
        )
        row = self._conn.execute(
            "SELECT conversation_id FROM pending_messages WHERE id = ?",
            (pending_message_id,),
        ).fetchone()
        if row is not None:
            self._touch_conversation(int(row["conversation_id"]))
        self._conn.commit()

    def drain_pending_messages(self, *, conversation_id: int) -> list[PendingMessage]:
        rows = self._conn.execute(
            "SELECT * FROM pending_messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        if rows:
            self._conn.execute("DELETE FROM pending_messages WHERE conversation_id = ?", (conversation_id,))
            self._conn.commit()
        return [self._row_to_pending_message(row) for row in rows]

    def delete_pending_messages(self, *, pending_message_ids: tuple[int, ...], commit: bool = True) -> None:
        if not pending_message_ids:
            return
        placeholders = ", ".join("?" for _ in pending_message_ids)
        self._conn.execute(
            f"DELETE FROM pending_messages WHERE id IN ({placeholders})",
            pending_message_ids,
        )
        if commit:
            self._conn.commit()

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

    def _touch_conversation(self, conversation_id: int, *, timestamp: str | None = None) -> None:
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp or utcnow(), conversation_id),
        )

    def _deserialize_content_payload(self, raw: str) -> dict[str, object]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("SQLite content payload must be a JSON object")
        return payload

    def _row_to_message(self, row: sqlite3.Row) -> StoredMessage:
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

    def _row_to_conversation(self, row: sqlite3.Row) -> StoredConversation:
        return StoredConversation(
            id=row["id"],
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            model_alias=row["model_alias"],
            system_prompt_override=row["system_prompt_override"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_pending_message(self, row: sqlite3.Row) -> PendingMessage:
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
            payload: dict[str, object] = {
                "mime_type": image.mime_type,
                "file_id": image.file_id,
            }
            if image.file_size is not None:
                payload["file_size"] = image.file_size
            return payload
        if image.kind == "stored":
            assert image.blob_path is not None
            assert image.sha256 is not None
            assert image.size is not None
            return {
                "mime_type": image.mime_type,
                "blob_path": image.blob_path,
                "sha256": image.sha256,
                "size": image.size,
            }
        raise ValueError("Loaded image refs cannot be serialized to SQLite")

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
        return ImageRef.stored(
            mime_type=mime_type,
            blob_path=blob_path,
            sha256=sha256,
            size=size,
        )

    def _source_telegram_message_ids_from_payload(self, payload: dict[str, object]) -> tuple[int, ...]:
        raw_ids = payload.get("source_telegram_message_ids")
        if not isinstance(raw_ids, list):
            return ()
        source_ids: list[int] = []
        for raw_id in raw_ids:
            if isinstance(raw_id, int) and raw_id > 0:
                source_ids.append(raw_id)
        return tuple(source_ids)


class InboxStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def enqueue_messages(self, *, messages: list[IncomingMessage]) -> None:
        if not messages:
            return
        rows = [
            (
                message.update_id,
                message.chat_id,
                message.user_id,
                message.message_id,
                self._media_group_key(message),
                "queued",
                json.dumps(self._serialize_incoming_message(message)),
                utcnow(),
            )
            for message in messages
        ]
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO inbox_updates (
                update_id, chat_id, user_id, message_id, media_group_key, state, payload_json, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def has_queued_updates(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM inbox_updates WHERE state = 'queued' LIMIT 1",
        ).fetchone()
        return row is not None

    def get_update(self, *, update_id: int) -> InboxUpdate | None:
        row = self._conn.execute(
            "SELECT * FROM inbox_updates WHERE update_id = ?",
            (update_id,),
        ).fetchone()
        return None if row is None else self._row_to_inbox_update(row)

    def list_updates(self, *, state: str | None = None) -> list[InboxUpdate]:
        if state is None:
            rows = self._conn.execute(
                "SELECT * FROM inbox_updates ORDER BY update_id ASC",
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM inbox_updates WHERE state = ? ORDER BY update_id ASC",
                (state,),
            ).fetchall()
        return [self._row_to_inbox_update(row) for row in rows]

    def claim_next_ready(
        self,
        *,
        media_group_delay_seconds: float,
        media_group_boundary_grace_seconds: float = 0.0,
    ) -> list[InboxUpdate] | None:
        rows = self._conn.execute(
            "SELECT * FROM inbox_updates WHERE state = 'queued' ORDER BY update_id ASC",
        ).fetchall()
        if not rows:
            return None

        now = datetime.now(timezone.utc)
        seen_media_group_keys: set[str] = set()
        for row in rows:
            entry = self._row_to_inbox_update(row)
            if entry.media_group_key is None:
                claimed_at = utcnow()
                cursor = self._conn.execute(
                    """
                    UPDATE inbox_updates
                    SET state = 'claimed', claimed_at = ?
                    WHERE update_id = ? AND state = 'queued'
                    """,
                    (claimed_at, entry.update_id),
                )
                if cursor.rowcount:
                    self._conn.commit()
                    return [self._replace_state(entry, state="claimed", claimed_at=claimed_at)]
                continue

            if entry.media_group_key in seen_media_group_keys:
                continue
            seen_media_group_keys.add(entry.media_group_key)
            claimed_group_row = self._conn.execute(
                """
                SELECT 1
                FROM inbox_updates
                WHERE state = 'claimed' AND media_group_key = ?
                LIMIT 1
                """,
                (entry.media_group_key,),
            ).fetchone()
            if claimed_group_row is not None:
                continue
            group_rows = self._conn.execute(
                """
                SELECT *
                FROM inbox_updates
                WHERE state = 'queued' AND media_group_key = ?
                ORDER BY update_id ASC
                """,
                (entry.media_group_key,),
            ).fetchall()
            if not group_rows:
                continue
            latest_received_at = max(self._parse_timestamp(group_row["received_at"]) for group_row in group_rows)
            ready_after_seconds = media_group_delay_seconds + media_group_boundary_grace_seconds
            if (now - latest_received_at).total_seconds() < ready_after_seconds:
                continue

            claimed_at = utcnow()
            self._conn.execute(
                """
                UPDATE inbox_updates
                SET state = 'claimed', claimed_at = ?
                WHERE state = 'queued' AND media_group_key = ?
                """,
                (claimed_at, entry.media_group_key),
            )
            self._conn.commit()
            return [
                self._replace_state(self._row_to_inbox_update(group_row), state="claimed", claimed_at=claimed_at)
                for group_row in group_rows
            ]
        return None

    def claim_queued_media_group_siblings(self, *, media_group_key: str) -> list[InboxUpdate]:
        group_rows = self._conn.execute(
            """
            SELECT *
            FROM inbox_updates
            WHERE state = 'queued' AND media_group_key = ?
            ORDER BY update_id ASC
            """,
            (media_group_key,),
        ).fetchall()
        if not group_rows:
            return []

        claimed_at = utcnow()
        self._conn.execute(
            """
            UPDATE inbox_updates
            SET state = 'claimed', claimed_at = ?
            WHERE state = 'queued' AND media_group_key = ?
            """,
            (claimed_at, media_group_key),
        )
        self._conn.commit()
        return [
            self._replace_state(self._row_to_inbox_update(group_row), state="claimed", claimed_at=claimed_at)
            for group_row in group_rows
        ]

    def mark_updates_realized(
        self,
        *,
        update_ids: tuple[int, ...],
        user_message_id: int | None,
        assistant_message_id: int | None,
        commit: bool = True,
    ) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET realized_user_message_id = ?, realized_assistant_message_id = ?
            WHERE update_id IN ({placeholders})
            """,
            (user_message_id, assistant_message_id, *update_ids),
        )
        if commit:
            self._conn.commit()

    def mark_reply_sent(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        reply_sent_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET reply_sent_at = COALESCE(reply_sent_at, ?)
            WHERE update_id IN ({placeholders})
            """,
            (reply_sent_at, *update_ids),
        )
        if commit:
            self._conn.commit()

    def mark_reply_started(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        reply_started_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET reply_started_at = COALESCE(reply_started_at, ?)
            WHERE update_id IN ({placeholders})
            """,
            (reply_started_at, *update_ids),
        )
        if commit:
            self._conn.commit()

    def clear_reply_sent(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET reply_sent_at = NULL
            WHERE update_id IN ({placeholders})
            """,
            update_ids,
        )
        if commit:
            self._conn.commit()

    def clear_reply_started(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET reply_started_at = NULL
            WHERE update_id IN ({placeholders})
            """,
            update_ids,
        )
        if commit:
            self._conn.commit()

    def set_assistant_render_state(
        self,
        *,
        assistant_message_id: int,
        phase: str | None,
        final_status: str | None,
        reply_text: str | None,
        reasoning_blocks: tuple[str, ...],
        render_markdown: bool | None,
        commit: bool = True,
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
            self._conn.execute(
                "DELETE FROM assistant_render_states WHERE assistant_message_id = ?",
                (assistant_message_id,),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO assistant_render_states (
                    assistant_message_id, phase, final_status, reply_text, reasoning_json, render_markdown
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(assistant_message_id) DO UPDATE SET
                    phase = excluded.phase,
                    final_status = excluded.final_status,
                    reply_text = excluded.reply_text,
                    reasoning_json = excluded.reasoning_json,
                    render_markdown = excluded.render_markdown
                """,
                (
                    assistant_message_id,
                    phase,
                    final_status,
                    reply_text,
                    serialized_reasoning,
                    serialized_render_markdown,
                ),
            )
        self._conn.execute(
            """
            UPDATE inbox_updates
            SET
                assistant_render_phase = ?,
                assistant_render_final_status = ?,
                assistant_render_reply_text = ?,
                assistant_render_reasoning_json = ?,
                assistant_render_markdown = ?
            WHERE realized_assistant_message_id = ?
            """,
            (
                phase,
                final_status,
                reply_text,
                serialized_reasoning,
                serialized_render_markdown,
                assistant_message_id,
            ),
        )
        if commit:
            self._conn.commit()

    def get_assistant_render_state(self, *, assistant_message_id: int) -> AssistantRenderState | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM assistant_render_states
            WHERE assistant_message_id = ?
            """,
            (assistant_message_id,),
        ).fetchone()
        if row is None:
            return None
        reasoning_blocks: tuple[str, ...] = ()
        raw_reasoning = row["reasoning_json"]
        if isinstance(raw_reasoning, str) and raw_reasoning:
            parsed_reasoning = json.loads(raw_reasoning)
            if isinstance(parsed_reasoning, list):
                reasoning_blocks = tuple(str(item) for item in parsed_reasoning if isinstance(item, str))
        return AssistantRenderState(
            assistant_message_id=row["assistant_message_id"],
            phase=row["phase"],
            final_status=row["final_status"],
            reply_text=row["reply_text"],
            reasoning_blocks=reasoning_blocks,
            render_markdown=None if row["render_markdown"] is None else bool(row["render_markdown"]),
        )

    def complete_updates(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        completed_at = utcnow()
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET state = 'completed', completed_at = ?
            WHERE update_id IN ({placeholders})
            """,
            (completed_at, *update_ids),
        )
        if commit:
            self._conn.commit()

    def reset_updates_to_queued(self, *, update_ids: tuple[int, ...], commit: bool = True) -> None:
        if not update_ids:
            return
        placeholders = ", ".join("?" for _ in update_ids)
        self._conn.execute(
            f"""
            UPDATE inbox_updates
            SET
                state = 'queued',
                claimed_at = NULL,
                completed_at = NULL,
                reply_started_at = NULL,
                reply_sent_at = NULL,
                realized_user_message_id = NULL,
                realized_assistant_message_id = NULL,
                assistant_render_phase = NULL,
                assistant_render_final_status = NULL,
                assistant_render_reply_text = NULL,
                assistant_render_reasoning_json = NULL,
                assistant_render_markdown = NULL
            WHERE update_id IN ({placeholders})
            """,
            update_ids,
        )
        if commit:
            self._conn.commit()

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

    def _row_to_inbox_update(self, row: sqlite3.Row) -> InboxUpdate:
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
            assistant_render_markdown=(
                None if row["assistant_render_markdown"] is None else bool(row["assistant_render_markdown"])
            ),
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
            payload: dict[str, object] = {
                "mime_type": image.mime_type,
                "file_id": image.file_id,
            }
            if image.file_size is not None:
                payload["file_size"] = image.file_size
            return payload
        if image.kind == "stored":
            return {
                "mime_type": image.mime_type,
                "blob_path": image.blob_path,
                "sha256": image.sha256,
                "size": image.size,
            }
        raise ValueError("Inbox messages cannot serialize loaded image refs")

    def _deserialize_image(self, raw_image: object) -> ImageRef | None:
        if not isinstance(raw_image, dict):
            return None
        file_id = raw_image.get("file_id")
        mime_type = raw_image.get("mime_type")
        if isinstance(file_id, str) and isinstance(mime_type, str):
            file_size = raw_image.get("file_size")
            return ImageRef.telegram(
                file_id=file_id,
                mime_type=mime_type,
                file_size=file_size if isinstance(file_size, int) else None,
            )
        blob_path = raw_image.get("blob_path")
        sha256 = raw_image.get("sha256")
        size = raw_image.get("size")
        if not isinstance(blob_path, str) or not isinstance(sha256, str) or not isinstance(mime_type, str) or not isinstance(size, int):
            return None
        return ImageRef.stored(
            mime_type=mime_type,
            blob_path=blob_path,
            sha256=sha256,
            size=size,
        )

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


class Storage:
    def __init__(
        self,
        db_path: Path,
        default_model_alias: str,
        default_reply_mode: str,
        default_skip_prefix: str,
    ) -> None:
        self.db_path = Path(db_path)
        self.data_dir = self.db_path.parent
        self.default_model_alias = default_model_alias
        self.default_reply_mode = default_reply_mode
        self.default_skip_prefix = default_skip_prefix

        self._validate_schema_compatibility()

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._ensure_additive_schema()
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()
        self._savepoint_counter = 0

        self.settings = SettingsStore(
            self._conn,
            default_model_alias=default_model_alias,
            default_reply_mode=default_reply_mode,
            default_skip_prefix=default_skip_prefix,
        )
        self.conversations = ConversationStore(self._conn)
        self.inbox = InboxStore(self._conn)
        self.blobs = BlobStore(self.data_dir)

    def close(self) -> None:
        self._conn.close()

    def _ensure_additive_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_render_states (
                assistant_message_id INTEGER PRIMARY KEY,
                phase TEXT,
                final_status TEXT,
                reply_text TEXT,
                reasoning_json TEXT,
                render_markdown INTEGER
            )
            """
        )
        self._ensure_column("inbox_updates", "reply_started_at", "TEXT")
        self._ensure_column("inbox_updates", "reply_sent_at", "TEXT")
        self._ensure_column("inbox_updates", "assistant_render_phase", "TEXT")
        self._ensure_column("inbox_updates", "assistant_render_final_status", "TEXT")
        self._ensure_column("inbox_updates", "assistant_render_reply_text", "TEXT")
        self._ensure_column("inbox_updates", "assistant_render_reasoning_json", "TEXT")
        self._ensure_column("inbox_updates", "assistant_render_markdown", "INTEGER")

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        if column_name in existing_columns:
            return
        self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        savepoint_name = f"sp_{self._savepoint_counter}"
        self._savepoint_counter += 1
        self._conn.execute(f"SAVEPOINT {savepoint_name}")
        try:
            yield
        except Exception:
            self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            self._conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            raise
        else:
            self._conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")

    def list_referenced_model_aliases(self) -> tuple[str, ...]:
        rows = self._conn.execute(
            """
            SELECT alias
            FROM (
                SELECT default_model_alias AS alias FROM chat_settings
                UNION
                SELECT model_alias AS alias FROM conversations
            )
            WHERE alias IS NOT NULL AND alias != ''
            ORDER BY alias COLLATE NOCASE ASC, alias ASC
            """
        ).fetchall()
        return tuple(row["alias"] for row in rows)

    def _validate_schema_compatibility(self) -> None:
        if not self.db_path.exists():
            return
        conn = sqlite3.connect(str(self.db_path))
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            object_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM sqlite_master
                    WHERE type IN ('table', 'index', 'trigger', 'view')
                      AND name NOT LIKE 'sqlite_%'
                    """
                ).fetchone()[0]
            )
        finally:
            conn.close()
        if version == 0 and object_count == 0:
            return
        if version in COMPATIBLE_SCHEMA_VERSIONS:
            return
        compatible_versions = ", ".join(str(item) for item in sorted(COMPATIBLE_SCHEMA_VERSIONS))
        raise RuntimeError(
            f"Incompatible SQLite schema version {version} at {self.db_path}. "
            f"Expected one of: {compatible_versions}. "
            "Switchboard does not auto-reset or migrate incompatible databases; "
            "move or delete the existing database file if you want to start fresh."
        )
