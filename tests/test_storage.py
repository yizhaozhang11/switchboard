from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sqlite3

from app.storage import Storage
from app.types import ContentPart, ImageRef, IncomingMessage


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "bot.sqlite3"
        self.storage = Storage(
            self.db_path,
            default_model_alias="o",
            default_reply_mode="auto",
            default_skip_prefix="//",
        )

    def tearDown(self) -> None:
        self.storage.close()
        self.tempdir.cleanup()

    def _make_inbox_message(
        self,
        *,
        update_id: int,
        message_id: int,
        text: str = "album",
        media_group_id: str | None = None,
    ) -> IncomingMessage:
        return IncomingMessage(
            update_id=update_id,
            chat_id=100,
            message_id=message_id,
            user_id=200,
            chat_type="group",
            text=text,
            from_bot=False,
            mentions_bot=False,
            source_message_ids=(message_id,),
            images=(ImageRef.telegram(file_id=f"photo-{update_id}", mime_type="image/jpeg", file_size=20),)
            if media_group_id is not None
            else (),
            media_group_id=media_group_id,
        )

    def test_default_settings_are_created(self) -> None:
        settings = self.storage.settings.get_chat_settings(123)
        self.assertEqual(settings.chat_id, 123)
        self.assertEqual(settings.default_model_alias, "o")
        self.assertEqual(settings.reply_mode, "auto")

    def test_find_recent_state_message_by_chat_and_user(self) -> None:
        older_conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        older_message_id = self.storage.conversations.create_message(
            conversation_id=older_conversation.id,
            chat_id=1,
            telegram_message_id=11,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="older",
            status="complete",
            created_at="2026-04-14T10:00:00+00:00",
        )
        newer_conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        newer_message_id = self.storage.conversations.create_message(
            conversation_id=newer_conversation.id,
            chat_id=1,
            telegram_message_id=12,
            message_type="seed",
            parent_message_id=None,
            provider=None,
            model_id=None,
            model_alias="o",
            content="seed prompt",
            status="complete",
            created_at="2026-04-14T11:00:00+00:00",
        )
        other_user_conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=99, model_alias="o")
        other_user_message_id = self.storage.conversations.create_message(
            conversation_id=other_user_conversation.id,
            chat_id=1,
            telegram_message_id=13,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="other user",
            status="complete",
            created_at="2026-04-14T12:00:00+00:00",
        )
        resolved = self.storage.conversations.find_recent_state_message(
            chat_id=1,
            user_id=10,
            not_before="2026-04-14T09:00:00+00:00",
        )
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.id, newer_message_id)
        self.assertEqual(resolved.message_type, "seed")

    def test_build_thread_returns_ordered_messages(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=11,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hi",
            status="complete",
        )
        thread = self.storage.conversations.build_thread(assistant_message_id)
        self.assertEqual([item.message_type for item in thread], ["user", "assistant"])
        self.assertEqual([item.content for item in thread], ["hello", "hi"])
        self.assertEqual(thread[0].conversation_id, conversation.id)

    def test_conversation_tip_is_derived_from_latest_message(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=11,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hi",
            status="complete",
        )

        tip = self.storage.conversations.get_conversation_tip_message(conversation.id)
        self.assertIsNotNone(tip)
        assert tip is not None
        self.assertEqual(tip.id, assistant_message_id)

    def test_conversation_streaming_state_is_derived_from_latest_assistant(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=11,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="",
            status="streaming",
        )

        self.assertTrue(self.storage.conversations.is_conversation_streaming(conversation.id))
        self.storage.conversations.update_message(assistant_message_id, content="done", status="complete")
        self.assertFalse(self.storage.conversations.is_conversation_streaming(conversation.id))

    def test_direct_telegram_lookup_resolves_seed_and_user_messages(self) -> None:
        conversation = self.storage.conversations.create_conversation(
            chat_id=1,
            user_id=10,
            model_alias="o",
            system_prompt_override="be concise",
        )
        seed_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=111,
            message_type="seed",
            parent_message_id=None,
            provider=None,
            model_id=None,
            model_alias="o",
            content="be concise",
            status="complete",
        )
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=112,
            message_type="user",
            parent_message_id=seed_message_id,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )

        resolved_seed = self.storage.conversations.get_message_by_telegram(chat_id=1, telegram_message_id=111)
        self.assertIsNotNone(resolved_seed)
        assert resolved_seed is not None
        self.assertEqual(resolved_seed.id, seed_message_id)

        resolved_user = self.storage.conversations.get_message_by_telegram(chat_id=1, telegram_message_id=112)
        self.assertIsNotNone(resolved_user)
        assert resolved_user is not None
        self.assertEqual(resolved_user.id, user_message_id)

    def test_assistant_telegram_link_resolution(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )
        self.storage.conversations.link_telegram_message(chat_id=1, telegram_message_id=999, logical_message_id=assistant_message_id, part_index=0)
        resolved = self.storage.conversations.get_message_by_telegram(chat_id=1, telegram_message_id=999)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.id, assistant_message_id)
        self.assertEqual(resolved.conversation_id, conversation.id)

    def test_media_group_item_link_resolution(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=112,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="album",
            status="complete",
        )
        self.storage.conversations.link_telegram_message(chat_id=1, telegram_message_id=111, logical_message_id=user_message_id, part_index=0)

        resolved = self.storage.conversations.get_message_by_telegram(chat_id=1, telegram_message_id=111)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.id, user_message_id)
        self.assertEqual(resolved.message_type, "user")

    def test_pending_queue_drains_in_order(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        self.storage.conversations.enqueue_pending_message(conversation_id=conversation.id, telegram_message_id=101, content="first")
        self.storage.conversations.enqueue_pending_message(conversation_id=conversation.id, telegram_message_id=102, content="second")

        drained = self.storage.conversations.drain_pending_messages(conversation_id=conversation.id)
        self.assertEqual([item.content for item in drained], ["first", "second"])
        self.assertEqual(self.storage.conversations.drain_pending_messages(conversation_id=conversation.id), [])

    def test_message_image_payload_round_trips(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        image = self.storage.blobs.store_image(mime_type="image/jpeg", data=b"image-bytes")
        message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=11,
            message_type="user",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="look",
            images=(image,),
            status="complete",
        )
        message = self.storage.conversations.get_message(message_id)
        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.content, "look")
        self.assertEqual(len(message.images), 1)
        self.assertEqual(self.storage.blobs.load_image_bytes(message.images[0]), b"image-bytes")

    def test_pending_queue_preserves_images(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        image = self.storage.blobs.store_image(mime_type="image/png", data=b"png-bytes")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=101,
            content="first",
            images=(image,),
        )
        drained = self.storage.conversations.drain_pending_messages(conversation_id=conversation.id)
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].content, "first")
        self.assertEqual(len(drained[0].images), 1)
        self.assertEqual(self.storage.blobs.load_image_bytes(drained[0].images[0]), b"png-bytes")

    def test_pending_queue_preserves_interleaved_parts(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        image = self.storage.blobs.store_image(mime_type="image/png", data=b"png-bytes")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=101,
            content="summary",
            images=(image,),
            parts=(
                ContentPart(kind="text", text="1.\nfirst"),
                ContentPart(kind="image", image=image),
            ),
        )
        drained = self.storage.conversations.drain_pending_messages(conversation_id=conversation.id)
        self.assertEqual(len(drained), 1)
        self.assertEqual([part.kind for part in drained[0].parts], ["text", "image"])
        self.assertEqual(drained[0].parts[0].text, "1.\nfirst")
        self.assertEqual(
            self.storage.blobs.load_image_bytes(drained[0].parts[1].image) if drained[0].parts[1].image is not None else None,
            b"png-bytes",
        )

    def test_pending_queue_preserves_source_telegram_message_ids(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=102,
            source_telegram_message_ids=(100, 101, 102),
            content="album",
        )

        drained = self.storage.conversations.drain_pending_messages(conversation_id=conversation.id)
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].telegram_message_id, 102)
        self.assertEqual(drained[0].source_telegram_message_ids, (100, 101, 102))

    def test_find_pending_message_by_source_telegram_id(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=102,
            source_telegram_message_ids=(100, 101, 102),
            content="album",
        )

        resolved = self.storage.conversations.find_pending_message_by_telegram(chat_id=1, telegram_message_id=100)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.telegram_message_id, 102)
        self.assertEqual(resolved.source_telegram_message_ids, (100, 101, 102))

    def test_allowlist_toggle_and_lookup(self) -> None:
        self.assertFalse(self.storage.settings.is_reply_allowed(chat_id=1, user_id=10))

        self.assertTrue(self.storage.settings.toggle_allowlist_entry(kind="chat", target_id=1))
        self.assertTrue(self.storage.settings.is_reply_allowed(chat_id=1, user_id=999))
        self.assertFalse(self.storage.settings.is_reply_allowed(chat_id=2, user_id=10))

        self.assertTrue(self.storage.settings.toggle_allowlist_entry(kind="user", target_id=10))
        self.assertTrue(self.storage.settings.is_reply_allowed(chat_id=2, user_id=10))

        entries = self.storage.settings.list_allowlist_entries()
        self.assertEqual([(entry.kind, entry.target_id) for entry in entries], [("chat", 1), ("user", 10)])

        self.assertFalse(self.storage.settings.toggle_allowlist_entry(kind="chat", target_id=1))
        self.assertFalse(self.storage.settings.is_reply_allowed(chat_id=1, user_id=999))

    def test_list_linked_telegram_message_ids_returns_part_order(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=1, user_id=10, model_alias="o")
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=1,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=None,
            provider="openai",
            model_id="gpt",
            model_alias="o",
            content="hello",
            status="complete",
        )
        self.storage.conversations.link_telegram_message(chat_id=1, telegram_message_id=1002, logical_message_id=assistant_message_id, part_index=1)
        self.storage.conversations.link_telegram_message(chat_id=1, telegram_message_id=1001, logical_message_id=assistant_message_id, part_index=0)

        self.assertEqual(
            self.storage.conversations.list_linked_telegram_message_ids(logical_message_id=assistant_message_id),
            [1001, 1002],
        )

    def test_inbox_update_round_trips_images_and_parts(self) -> None:
        incoming = IncomingMessage(
            update_id=1,
            chat_id=100,
            message_id=10,
            user_id=200,
            chat_type="group",
            text="look",
            from_bot=False,
            mentions_bot=False,
            source_message_ids=(10,),
            images=(ImageRef.telegram(file_id="photo-1", mime_type="image/jpeg", file_size=20),),
            parts=(
                ContentPart(kind="text", text="look"),
                ContentPart(kind="image", image=ImageRef.telegram(file_id="photo-1", mime_type="image/jpeg", file_size=20)),
            ),
            media_group_id="group-1",
        )

        self.storage.inbox.enqueue_messages(messages=[incoming])

        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.message.text, "look")
        self.assertEqual(stored.message.media_group_id, "group-1")
        self.assertEqual(stored.message.images[0].file_id, "photo-1")
        self.assertEqual([part.kind for part in stored.message.parts], ["text", "image"])

    def test_inbox_claim_skips_unready_album_for_ready_plain_message(self) -> None:
        album_item = IncomingMessage(
            update_id=1,
            chat_id=100,
            message_id=10,
            user_id=200,
            chat_type="group",
            text="album",
            from_bot=False,
            mentions_bot=False,
            source_message_ids=(10,),
            images=(ImageRef.telegram(file_id="photo-1", mime_type="image/jpeg", file_size=20),),
            parts=(
                ContentPart(kind="text", text="album"),
                ContentPart(kind="image", image=ImageRef.telegram(file_id="photo-1", mime_type="image/jpeg", file_size=20)),
            ),
            media_group_id="group-1",
        )
        plain_item = IncomingMessage(
            update_id=2,
            chat_id=100,
            message_id=11,
            user_id=200,
            chat_type="group",
            text="plain",
            from_bot=False,
            mentions_bot=False,
            source_message_ids=(11,),
        )
        self.storage.inbox.enqueue_messages(messages=[album_item, plain_item])

        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=60.0)
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual([item.update_id for item in claim], [2])

    def test_inbox_claim_honors_media_group_boundary_grace(self) -> None:
        self.storage.inbox.enqueue_messages(
            messages=[
                self._make_inbox_message(
                    update_id=1,
                    message_id=10,
                    media_group_id="group-1",
                )
            ]
        )

        claim = self.storage.inbox.claim_next_ready(
            media_group_delay_seconds=0.0,
            media_group_boundary_grace_seconds=60.0,
        )

        self.assertIsNone(claim)

    def test_inbox_claim_skips_media_group_with_claimed_sibling(self) -> None:
        self.storage.inbox.enqueue_messages(
            messages=[
                self._make_inbox_message(update_id=1, message_id=10, media_group_id="group-1"),
            ]
        )
        first_claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(first_claim)

        self.storage.inbox.enqueue_messages(
            messages=[
                self._make_inbox_message(update_id=2, message_id=11, media_group_id="group-1"),
            ]
        )

        self.assertIsNone(self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0))
        sibling_claim = self.storage.inbox.claim_queued_media_group_siblings(media_group_key="100:200:group-1")
        self.assertEqual([entry.update_id for entry in sibling_claim], [2])

    def test_empty_existing_database_is_initialized(self) -> None:
        self.storage.close()
        self.db_path.unlink()
        self.db_path.touch()

        self.storage = Storage(
            self.db_path,
            default_model_alias="o",
            default_reply_mode="auto",
            default_skip_prefix="//",
        )

        settings = self.storage.settings.get_chat_settings(123)
        self.assertEqual(settings.default_model_alias, "o")
        version = int(self.storage._conn.execute("PRAGMA user_version").fetchone()[0])
        self.assertEqual(version, 1)

    def test_incompatible_schema_version_raises_and_preserves_database(self) -> None:
        self.storage.close()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE stale_table (value INTEGER)")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()
        conn.close()

        with self.assertRaisesRegex(RuntimeError, "Incompatible SQLite schema version 4"):
            Storage(
                self.db_path,
                default_model_alias="o",
                default_reply_mode="auto",
                default_skip_prefix="//",
            )

        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'stale_table'"
            ).fetchone()
            self.assertIsNotNone(row)
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(version, 4)
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
