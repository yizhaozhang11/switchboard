from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    class AsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def aclose(self) -> None:
            return None

    httpx_stub.HTTPError = HTTPError
    httpx_stub.AsyncClient = AsyncClient
    sys.modules["httpx"] = httpx_stub

from app.router import Router
from app.richtext import RichText
from app.storage import Storage
from app.telegram_app import InboxClaim, TelegramApp


def make_update(
    *,
    update_id: int = 1,
    chat_id: int = 100,
    message_id: int = 10,
    user_id: int = 200,
    text: str | None = "hello",
    caption: str | None = None,
    photo_file_id: str | None = None,
    image_document_file_id: str | None = None,
    image_document_mime_type: str = "image/png",
    video_file_id: str | None = None,
    media_group_id: str | None = None,
    reply_to_user_id: int | None = None,
    reply_to_bot: bool = False,
) -> dict:
    message: dict[str, object] = {
        "message_id": message_id,
        "chat": {"id": chat_id, "type": "group"},
        "from": {"id": user_id, "is_bot": False},
    }
    if text is not None:
        message["text"] = text
    if caption is not None:
        message["caption"] = caption
    if photo_file_id is not None:
        message["photo"] = [
            {"file_id": photo_file_id + "-small", "file_size": 10},
            {"file_id": photo_file_id, "file_size": 20},
        ]
    if image_document_file_id is not None:
        message["document"] = {
            "file_id": image_document_file_id,
            "file_size": 25,
            "mime_type": image_document_mime_type,
        }
    if video_file_id is not None:
        message["video"] = {
            "file_id": video_file_id,
            "file_size": 30,
            "mime_type": "video/mp4",
        }
    if media_group_id is not None:
        message["media_group_id"] = media_group_id
    if reply_to_user_id is not None or reply_to_bot:
        reply_from = {
            "id": 999 if reply_to_bot else reply_to_user_id,
            "is_bot": reply_to_bot,
        }
        message["reply_to_message"] = {
            "message_id": 9,
            "text": "earlier",
            "from": reply_from,
        }
    return {"update_id": update_id, "message": message}


class FakeTelegramAPI:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, object]] = []
        self.edits: list[dict[str, object]] = []
        self.file_bytes: dict[str, bytes] = {}
        self.send_failures_remaining = 0
        self.before_send = None

    async def send_message(
        self,
        chat_id: int,
        text: str | RichText,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        if self.send_failures_remaining > 0:
            self.send_failures_remaining -= 1
            raise RuntimeError("send failed")
        if self.before_send is not None:
            self.before_send(chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id)
        rendered_text, entities = RichText.coerce(text).to_telegram()
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "text": rendered_text,
                "entities": entities,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        return 1000 + len(self.sent_messages)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str | RichText) -> None:
        rendered_text, entities = RichText.coerce(text).to_telegram()
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": rendered_text, "entities": entities})

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        return None

    async def download_file_bytes(self, file_id: str) -> bytes:
        return self.file_bytes[file_id]


class FakePollingTelegramAPI(FakeTelegramAPI):
    def __init__(self) -> None:
        super().__init__()
        self.get_updates_calls: list[int | None] = []
        self.get_updates_handler = None

    async def get_me(self) -> dict:
        return {"id": 999, "username": "TestBot"}

    async def get_updates(self, *, offset: int | None, timeout: int) -> list[dict]:
        self.get_updates_calls.append(offset)
        assert self.get_updates_handler is not None
        return await self.get_updates_handler(offset=offset, timeout=timeout)


class FakeService:
    def __init__(self) -> None:
        self.allowed = False
        self.manage_allowed = True
        self.owners_configured = True
        self.allowlist_manage_allowed = True
        self.generate_calls: list[dict[str, object]] = []
        self.toggle_chat_calls: list[int] = []
        self.toggle_user_calls: list[int] = []
        self.whitelist_calls = 0
        self.recovered_assistant_ids: list[int] = []
        self.recovery_calls: list[dict[str, object]] = []
        self.recovery_result = True

    def help_text(self, *, settings) -> str:
        return f"Help for {settings.default_model_alias}"

    def command_help_text(self, *, topic, settings):
        normalized = topic.strip().lstrip("/").lower()
        if normalized not in {"new", "c", "s"}:
            return None
        return f"Explain {normalized} for {settings.default_model_alias}"

    def is_reply_allowed(self, *, chat_id: int, user_id: int) -> bool:
        return self.allowed

    async def generate_reply(self, **kwargs) -> None:
        self.generate_calls.append(kwargs)

    async def recover_interrupted_assistant_turn(self, *, api, assistant_message_id: int, **kwargs) -> bool:
        _ = api
        self.recovered_assistant_ids.append(assistant_message_id)
        self.recovery_calls.append({"assistant_message_id": assistant_message_id, **kwargs})
        return self.recovery_result

    def can_manage_chat(self, user_id: int) -> bool:
        return self.manage_allowed

    def has_configured_owners(self) -> bool:
        return self.owners_configured

    def can_manage_allowlist(self, user_id: int) -> bool:
        return self.allowlist_manage_allowed

    def set_default_model(self, *, chat_id: int, alias: str, commit: bool = True) -> str:
        _ = (chat_id, commit)
        if not alias:
            raise ValueError("Unknown model alias: ")
        return f"Default model set to {alias}"

    def set_reply_mode(self, *, chat_id: int, reply_mode: str, commit: bool = True) -> str:
        _ = (chat_id, commit)
        if reply_mode not in {"auto", "mention", "off"}:
            raise ValueError("Reply mode must be one of auto, mention, off")
        return f"Reply mode set to {reply_mode}"

    def toggle_chat_allowlist(self, *, chat_id: int, commit: bool = True) -> str:
        _ = commit
        self.toggle_chat_calls.append(chat_id)
        return f"Chat {chat_id} toggled"

    def toggle_user_allowlist(self, *, user_id: int, commit: bool = True) -> str:
        _ = commit
        self.toggle_user_calls.append(user_id)
        return f"User {user_id} toggled"

    def whitelist_text(self) -> str:
        self.whitelist_calls += 1
        return "Whitelisted chats:\n- 100"

    def ping_text(self, message, settings) -> str:
        return f"chat_id={message.chat_id}\nuser_id={message.user_id}\nreply_mode={settings.reply_mode}\ndefault_model={settings.default_model_alias}"

    def list_models_text(self, settings) -> str:
        return f"Default model: {settings.default_model_alias}"


class DummyConfig:
    poll_timeout_seconds = 30


class TelegramAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "bot.sqlite3"
        self.storage = Storage(
            self.db_path,
            default_model_alias="o",
            default_reply_mode="auto",
            default_skip_prefix="//",
        )
        self.service = FakeService()
        self.api = FakeTelegramAPI()
        self.app = TelegramApp(
            config=DummyConfig(),
            storage=self.storage,
            service=self.service,
            api=self.api,
        )
        self.app.bot_id = 999
        self.app.bot_username = "TestBot"
        self.app.router = Router(bot_username="TestBot")
        self.app.scheduler_poll_seconds = 0.01

    def tearDown(self) -> None:
        self.storage.close()
        self.tempdir.cleanup()

    def _enqueue_updates(self, *updates: dict) -> None:
        inbox_messages = []
        for update in updates:
            incoming = self.app._parse_incoming_message(update)
            if incoming is not None:
                inbox_messages.append(incoming)
        self.storage.inbox.enqueue_messages(messages=inbox_messages)

    async def _process_next_claim(self) -> InboxClaim | None:
        entries = self.storage.inbox.claim_next_ready(
            media_group_delay_seconds=self.app.media_group_delay_seconds,
            media_group_boundary_grace_seconds=self.app._media_group_boundary_grace_seconds(),
        )
        if entries is None:
            return None
        claim = self.app._claim_from_entries(entries)
        await self.app._process_claim(claim)
        return claim

    async def test_plain_message_ignored_when_not_allowlisted(self) -> None:
        await self.app._handle_update(make_update(text="hello"))
        self.assertEqual(self.service.generate_calls, [])
        self.assertEqual(self.api.sent_messages, [])

    async def test_safe_handle_update_logs_only_sanitized_summary(self) -> None:
        update = make_update(
            update_id=77,
            chat_id=987654321,
            user_id=123456789,
            text="super sensitive text",
            caption="hidden caption",
            media_group_id="album-secret",
        )

        async def boom(_: dict) -> None:
            raise RuntimeError("boom")

        self.app._handle_update = boom  # type: ignore[method-assign]

        with self.assertLogs(level=logging.ERROR) as captured:
            await self.app._safe_handle_update(update)

        output = "\n".join(captured.output)
        self.assertIn("Failed to handle update (update_id=77 kinds=message message_features=text,caption,media_group_id)", output)
        self.assertNotIn("super sensitive text", output)
        self.assertNotIn("hidden caption", output)
        self.assertNotIn("987654321", output)
        self.assertNotIn("123456789", output)
        self.assertNotIn("album-secret", output)

    async def test_run_continues_polling_before_handler_completion(self) -> None:
        api = FakePollingTelegramAPI()
        app = TelegramApp(
            config=DummyConfig(),
            storage=self.storage,
            service=self.service,
            api=api,
        )
        app.bot_id = 999
        app.bot_username = "TestBot"
        app.router = Router(bot_username="TestBot")
        app.scheduler_poll_seconds = 0.01
        update_started = asyncio.Event()
        second_poll_started = asyncio.Event()
        release_update = asyncio.Event()

        async def get_updates_handler(*, offset: int | None, timeout: int) -> list[dict]:
            if timeout == 0:
                return []
            if len(api.get_updates_calls) == 1:
                return [make_update(update_id=1)]
            second_poll_started.set()
            await asyncio.Future()

        async def fake_handle(incoming, *, inbox_update_ids=()) -> None:
            _ = (incoming, inbox_update_ids)
            update_started.set()
            await release_update.wait()

        api.get_updates_handler = get_updates_handler
        app._handle_incoming_message = fake_handle  # type: ignore[method-assign]

        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(update_started.wait(), timeout=1.0)
        await asyncio.wait_for(second_poll_started.wait(), timeout=1.0)

        self.assertEqual(api.get_updates_calls, [None, 2])

        release_update.set()
        run_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run_task

    async def test_run_persists_updates_in_inbox_before_processing(self) -> None:
        api = FakePollingTelegramAPI()
        app = TelegramApp(
            config=DummyConfig(),
            storage=self.storage,
            service=self.service,
            api=api,
        )
        app.scheduler_poll_seconds = 0.01
        app.bot_id = 999
        app.bot_username = "TestBot"
        app.router = Router(bot_username="TestBot")
        release_update = asyncio.Event()
        started = asyncio.Event()

        async def get_updates_handler(*, offset: int | None, timeout: int) -> list[dict]:
            if timeout == 0:
                return []
            if len(api.get_updates_calls) == 1:
                return [make_update(update_id=1, text="persist me")]
            await asyncio.Future()

        async def fake_handle(incoming, *, inbox_update_ids=()) -> None:
            _ = inbox_update_ids
            started.set()
            stored = self.storage.inbox.get_update(update_id=incoming.update_id)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.state, "claimed")
            await release_update.wait()

        api.get_updates_handler = get_updates_handler
        app._handle_incoming_message = fake_handle  # type: ignore[method-assign]

        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        release_update.set()
        run_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run_task

    async def test_run_drops_bad_update_and_keeps_polling(self) -> None:
        api = FakePollingTelegramAPI()
        app = TelegramApp(
            config=DummyConfig(),
            storage=self.storage,
            service=self.service,
            api=api,
        )
        app.scheduler_poll_seconds = 0.01
        app.bot_id = 999
        app.bot_username = "TestBot"
        app.router = Router(bot_username="TestBot")
        second_poll_started = asyncio.Event()
        release_update = asyncio.Event()
        started = asyncio.Event()

        bad_update = make_update(update_id=1, text="bad")
        bad_update["message"]["chat"]["id"] = "not-an-int"

        async def get_updates_handler(*, offset: int | None, timeout: int) -> list[dict]:
            if timeout == 0:
                return []
            if len(api.get_updates_calls) == 1:
                return [bad_update, make_update(update_id=2, text="valid")]
            second_poll_started.set()
            await asyncio.Future()

        async def fake_handle(incoming, *, inbox_update_ids=()) -> None:
            _ = inbox_update_ids
            if incoming.update_id == 2:
                started.set()
                await release_update.wait()

        api.get_updates_handler = get_updates_handler
        app._handle_incoming_message = fake_handle  # type: ignore[method-assign]

        with self.assertLogs(level=logging.ERROR) as captured:
            run_task = asyncio.create_task(app.run())
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await asyncio.wait_for(second_poll_started.wait(), timeout=1.0)
            self.assertEqual(api.get_updates_calls, [None, 3])
            release_update.set()
            run_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run_task

        output = "\n".join(captured.output)
        self.assertIn("Failed to parse update (update_id=1 kinds=message message_features=text)", output)

    async def test_duplicate_updates_are_ignored_by_update_id(self) -> None:
        self._enqueue_updates(make_update(update_id=1, text="hello"))
        self._enqueue_updates(make_update(update_id=1, text="hello again"))
        updates = self.storage.inbox.list_updates()
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].message.text, "hello")

    async def test_media_group_claim_waits_but_later_plain_message_can_run(self) -> None:
        self.service.allowed = True
        self.app.media_group_delay_seconds = 0.02
        self._enqueue_updates(
            make_update(update_id=1, message_id=10, text=None, caption="album caption", photo_file_id="album-1", media_group_id="group-1"),
            make_update(update_id=2, message_id=11, text="plain follow-up"),
            make_update(update_id=3, message_id=12, text=None, photo_file_id="album-2", media_group_id="group-1"),
        )

        first_claim = await self._process_next_claim()
        assert first_claim is not None
        self.assertEqual(first_claim.update_ids, (2,))

        await asyncio.sleep(0.04)
        second_claim = await self._process_next_claim()
        assert second_claim is not None
        self.assertEqual(second_claim.update_ids, (1, 3))

        self.assertEqual(len(self.service.generate_calls), 2)
        plain_action = next(call["action"] for call in self.service.generate_calls if call["action"].content == "plain follow-up")
        self.assertEqual(plain_action.content, "plain follow-up")
        album_action = next(call["action"] for call in self.service.generate_calls if len(call["action"].images) == 2)
        self.assertEqual(album_action.content, "album caption")

    async def test_media_group_claim_absorbs_queued_sibling_before_handling(self) -> None:
        self.service.allowed = True
        self._enqueue_updates(
            make_update(update_id=1, message_id=10, text=None, caption="album caption", photo_file_id="album-1", media_group_id="group-1"),
        )
        entries = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(entries)
        assert entries is not None

        self._enqueue_updates(
            make_update(update_id=2, message_id=11, text=None, photo_file_id="album-2", media_group_id="group-1"),
        )

        await self.app._process_claim(self.app._claim_from_entries(entries))

        self.assertEqual(len(self.service.generate_calls), 1)
        action = self.service.generate_calls[0]["action"]
        self.assertEqual(action.content, "album caption")
        self.assertEqual(len(action.images), 2)
        stored_first = self.storage.inbox.get_update(update_id=1)
        stored_second = self.storage.inbox.get_update(update_id=2)
        self.assertIsNotNone(stored_first)
        self.assertIsNotNone(stored_second)
        assert stored_first is not None
        assert stored_second is not None
        self.assertEqual(stored_first.state, "completed")
        self.assertEqual(stored_second.state, "completed")

    async def test_shutdown_final_poll_enqueues_updates_without_processing_them(self) -> None:
        api = FakePollingTelegramAPI()
        app = TelegramApp(
            config=DummyConfig(),
            storage=self.storage,
            service=self.service,
            api=api,
        )
        app.scheduler_poll_seconds = 0.01
        app.bot_id = 999
        app.bot_username = "TestBot"
        app.router = Router(bot_username="TestBot")
        second_poll_started = asyncio.Event()
        final_poll_seen = asyncio.Event()

        async def get_updates_handler(*, offset: int | None, timeout: int) -> list[dict]:
            if timeout == 0:
                self.assertEqual(offset, 2)
                final_poll_seen.set()
                return [
                    make_update(
                        update_id=2,
                        message_id=11,
                        text=None,
                        caption="shutdown album",
                        photo_file_id="album-1",
                        media_group_id="group-1",
                    )
                ]
            if len(api.get_updates_calls) == 1:
                return [make_update(update_id=1, text="first")]
            second_poll_started.set()
            await asyncio.Future()

        api.get_updates_handler = get_updates_handler

        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(second_poll_started.wait(), timeout=1.0)
        run_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run_task

        await asyncio.wait_for(final_poll_seen.wait(), timeout=1.0)
        stored = self.storage.inbox.get_update(update_id=2)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state, "queued")
        self.assertEqual(self.service.generate_calls, [])

    async def test_claimed_unrealized_updates_reset_to_queue_on_recovery(self) -> None:
        self._enqueue_updates(make_update(update_id=1, text="hello"))
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "queued")

    async def test_claimed_pending_message_updates_complete_on_recovery(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=10,
            content="queued",
        )
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="queued"))
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "completed")

    async def test_claimed_reply_only_updates_complete_on_recovery(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="/help"))
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)
        self.storage.inbox.mark_reply_started(update_ids=(1,))
        self.storage.inbox.mark_reply_sent(update_ids=(1,))

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "completed")

    async def test_reply_only_response_marks_started_but_not_sent_before_send(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="/help"))

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            _ = (chat_id, text, reply_to_message_id)
            stored = self.storage.inbox.get_update(update_id=1)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertIsNotNone(stored.reply_started_at)
            self.assertIsNone(stored.reply_sent_at)

        self.api.before_send = before_send
        claim = await self._process_next_claim()

        assert claim is not None
        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state, "completed")
        self.assertIsNotNone(stored.reply_sent_at)

    async def test_claimed_reply_only_updates_with_started_but_unsent_reply_stay_claimed_on_recovery(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="/help"))
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)
        self.storage.inbox.mark_reply_started(update_ids=(1,))

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "claimed")

    async def test_reply_only_send_failure_leaves_started_claim_claimed(self) -> None:
        self.api.send_failures_remaining = 1
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="/help"))

        claim = await self._process_next_claim()

        assert claim is not None
        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state, "claimed")
        self.assertIsNotNone(stored.reply_started_at)
        self.assertIsNone(stored.reply_sent_at)

    async def test_requeued_reply_only_update_clears_stale_progress_before_recovery(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="/help"))
        first_claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(first_claim)
        self.storage.inbox.mark_reply_started(update_ids=(1,))
        self.storage.inbox.reset_updates_to_queued(update_ids=(1,))

        requeued = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(requeued)
        assert requeued is not None
        self.assertEqual(requeued.state, "queued")
        self.assertIsNone(requeued.reply_started_at)
        self.assertIsNone(requeued.reply_sent_at)

        second_claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(second_claim)

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "queued")

    async def test_orphaned_claim_requeues_without_stale_assistant_metadata(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="hello"))
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=11,
            assistant_message_id=12,
        )
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=12,
            phase="final_pending",
            final_status="complete",
            reply_text="hello",
            reasoning_blocks=("reasoning",),
            render_markdown=True,
        )
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "queued")
        self.assertIsNone(recovered.realized_user_message_id)
        self.assertIsNone(recovered.realized_assistant_message_id)
        self.assertIsNone(recovered.assistant_render_phase)
        self.assertIsNone(recovered.assistant_render_final_status)
        self.assertIsNone(recovered.assistant_render_reply_text)
        self.assertEqual(recovered.assistant_render_reasoning_blocks, ())
        self.assertIsNone(recovered.assistant_render_markdown)

    async def test_realized_assistant_claim_stays_claimed_after_handler_failure(self) -> None:
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="hello"))
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=11,
            assistant_message_id=12,
        )

        async def boom(incoming, *, inbox_update_ids=()) -> None:
            _ = (incoming, inbox_update_ids)
            raise RuntimeError("boom")

        self.app._handle_incoming_message = boom  # type: ignore[method-assign]
        claim = self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.assertIsNotNone(claim)
        assert claim is not None

        await self.app._process_claim(
            InboxClaim(
                update_ids=tuple(entry.update_id for entry in claim),
                messages=tuple(entry.message for entry in claim),
            )
        )

        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state, "claimed")

    async def test_claimed_assistant_updates_stay_claimed_when_recovery_fails(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=10,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="partial",
            status="streaming",
        )
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="hello"))
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)
        self.service.recovery_result = False

        await self.app._recover_claimed_updates()

        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "claimed")

    async def test_claimed_streaming_assistant_updates_trigger_recovery(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=10,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="partial",
            status="streaming",
        )
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="hello"))
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)

        await self.app._recover_claimed_updates()

        self.assertEqual(self.service.recovered_assistant_ids, [assistant_message_id])
        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "completed")
        self.assertEqual(self.service.recovery_calls[0]["reply_to_message_id"], 10)

    async def test_completed_pending_follow_up_streaming_assistant_triggers_recovery(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        first_user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=10,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="first",
            status="complete",
        )
        self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=1000,
            message_type="assistant",
            parent_message_id=first_user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="first reply",
            status="complete",
        )
        self._enqueue_updates(make_update(update_id=2, message_id=20, text="queued follow-up"))
        self.storage.inbox.complete_updates(update_ids=(2,))
        follow_up_user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=20,
            message_type="user",
            parent_message_id=first_user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="queued follow-up",
            status="complete",
        )
        orphan_assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=follow_up_user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="partial follow-up",
            status="streaming",
        )
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=orphan_assistant_message_id,
            phase="final_rendered",
            final_status="complete",
            reply_text="done follow-up",
            reasoning_blocks=("reasoning",),
            render_markdown=False,
        )

        await self.app._recover_claimed_updates()

        self.assertEqual(self.service.recovered_assistant_ids, [orphan_assistant_message_id])
        self.assertEqual(self.service.recovery_calls[0]["reply_to_message_id"], 20)
        self.assertEqual(self.service.recovery_calls[0]["final_render_phase"], "final_rendered")
        self.assertEqual(self.service.recovery_calls[0]["final_render_status"], "complete")
        self.assertEqual(self.service.recovery_calls[0]["final_render_reply_text"], "done follow-up")
        self.assertEqual(self.service.recovery_calls[0]["final_render_reasoning_blocks"], ("reasoning",))
        self.assertFalse(self.service.recovery_calls[0]["final_render_markdown"])
        stored_update = self.storage.inbox.get_update(update_id=2)
        self.assertIsNotNone(stored_update)
        assert stored_update is not None
        self.assertEqual(stored_update.state, "completed")

    async def test_claimed_final_assistant_updates_trigger_recovery(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=10,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="hello",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="model",
            model_alias="o",
            content="done",
            status="complete",
        )
        self._enqueue_updates(make_update(update_id=1, message_id=10, text="hello"))
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.claim_next_ready(media_group_delay_seconds=0.0)

        await self.app._recover_claimed_updates()

        self.assertEqual(self.service.recovered_assistant_ids, [assistant_message_id])
        recovered = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.state, "completed")

    async def test_command_mutation_completes_update_before_confirmation_send(self) -> None:
        self.api.send_failures_remaining = 1
        self._enqueue_updates(make_update(update_id=1, text="/togglechat"))

        claim = await self._process_next_claim()
        assert claim is not None

        self.assertEqual(self.service.toggle_chat_calls, [100])
        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.state, "completed")

    async def test_reply_mode_mention_allows_reply_to_stored_seed_anchor(self) -> None:
        self.service.allowed = True
        self.storage.settings.set_reply_mode(100, "mention")
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=9,
            message_type="seed",
            parent_message_id=None,
            provider=None,
            model_id=None,
            model_alias="o",
            content="be concise",
            status="complete",
        )

        await self.app._handle_update(make_update(text="follow-up", reply_to_user_id=321))

        self.assertEqual(len(self.service.generate_calls), 1)
        self.assertEqual(self.service.generate_calls[0]["action"].intent, "plain")

    async def test_reply_mode_mention_allows_reply_to_pending_user_anchor(self) -> None:
        self.service.allowed = True
        self.storage.settings.set_reply_mode(100, "mention")
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        self.storage.conversations.enqueue_pending_message(
            conversation_id=conversation.id,
            telegram_message_id=9,
            content="queued",
        )

        await self.app._handle_update(make_update(text="follow-up", reply_to_user_id=321))

        self.assertEqual(len(self.service.generate_calls), 1)
        self.assertEqual(self.service.generate_calls[0]["action"].intent, "plain")

    async def test_photo_with_caption_routes_images(self) -> None:
        self.service.allowed = True
        self.api.file_bytes["photo-1"] = b"jpeg-bytes"
        await self.app._handle_update(make_update(text=None, caption="look", photo_file_id="photo-1"))
        self.assertEqual(len(self.service.generate_calls), 1)
        action = self.service.generate_calls[0]["action"]
        self.assertEqual(action.content, "look")
        self.assertEqual(len(action.images), 1)
        self.assertEqual(action.images[0].file_id, "photo-1")

    async def test_unsupported_svg_image_document_is_ignored(self) -> None:
        self.service.allowed = True
        await self.app._handle_update(
            make_update(
                text=None,
                caption="describe this graphic",
                image_document_file_id="doc-svg",
                image_document_mime_type="image/svg+xml",
            )
        )
        self.assertEqual(self.service.generate_calls, [])

    async def test_new_and_system_prompt_commands_bypass_off_mode_gating(self) -> None:
        self.service.allowed = True
        self.storage.settings.set_reply_mode(100, "off")

        await self.app._handle_update(make_update(text="/new hello"))
        await self.app._handle_update(make_update(text="/s be concise", message_id=11))

        self.assertEqual(len(self.service.generate_calls), 2)
        self.assertEqual(self.service.generate_calls[0]["action"].intent, "new")
        self.assertEqual(self.service.generate_calls[1]["action"].intent, "set_system_prompt")


if __name__ == "__main__":
    unittest.main()
