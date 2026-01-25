from __future__ import annotations

import asyncio
import hashlib
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

from app.chat_service import ChatService
from app.providers.registry import ProviderRegistry
from app.richtext import RichText
from app.storage import Storage
from app.types import ChatAction, ChatRequest, ChatSettings, ContentPart, ImageRef, IncomingMessage, ModelSpec, StoredMessage, StreamEvent


def make_message(
    *,
    message_id: int,
    text: str,
    images: tuple[ImageRef, ...] = (),
    parts: tuple[ContentPart, ...] = (),
    chat_id: int = 100,
    user_id: int = 200,
    reply_to_message_id: int | None = None,
    reply_to_bot: bool = False,
    source_message_ids: tuple[int, ...] = (),
) -> IncomingMessage:
    effective_source_message_ids = source_message_ids or (message_id,)
    return IncomingMessage(
        update_id=message_id,
        chat_id=chat_id,
        message_id=message_id,
        user_id=user_id,
        chat_type="group",
        text=text,
        from_bot=False,
        mentions_bot=False,
        source_message_ids=effective_source_message_ids,
        reply_to_message_id=reply_to_message_id,
        reply_to_user_id=None,
        reply_to_bot=reply_to_bot,
        reply_to_text=None,
        images=images,
        parts=parts,
    )


def make_settings(*, default_model_alias: str = "o") -> ChatSettings:
    return ChatSettings(
        chat_id=100,
        enabled=True,
        reply_mode="auto",
        default_model_alias=default_model_alias,
        skip_prefix="//",
    )


class FakeTelegramAPI:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, int | str | list[dict[str, int | str]] | None]] = []
        self.edits: list[dict[str, int | str | list[dict[str, int | str]]]] = []
        self.deletes: list[dict[str, int]] = []
        self.file_bytes: dict[str, bytes] = {}
        self.download_counts: dict[str, int] = {}
        self.download_started: dict[str, asyncio.Event] = {}
        self.download_release: dict[str, asyncio.Event] = {}
        self._next_message_id = 1000
        self.send_failures_remaining = 0

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
        message_id = self._next_message_id
        self._next_message_id += 1
        rendered_text, entities = RichText.coerce(text).to_telegram()
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": rendered_text,
                "entities": entities,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        return message_id

    async def edit_message_text(self, chat_id: int, message_id: int, text: str | RichText) -> None:
        rendered_text, entities = RichText.coerce(text).to_telegram()
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": rendered_text, "entities": entities})

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self.deletes.append({"chat_id": chat_id, "message_id": message_id})

    async def download_file_bytes(self, file_id: str) -> bytes:
        self.download_counts[file_id] = self.download_counts.get(file_id, 0) + 1
        started = self.download_started.get(file_id)
        if started is not None:
            started.set()
        release = self.download_release.get(file_id)
        if release is not None:
            await release.wait()
        return self.file_bytes[file_id]


class InspectingTelegramAPI(FakeTelegramAPI):
    def __init__(self, *, before_send=None, before_edit=None, fail_edit=None) -> None:
        super().__init__()
        self.before_send = before_send
        self.before_edit = before_edit
        self.fail_edit = fail_edit

    async def send_message(
        self,
        chat_id: int,
        text: str | RichText,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        if self.before_send is not None:
            self.before_send(chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id)
        return await super().send_message(chat_id, text, reply_to_message_id=reply_to_message_id)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str | RichText) -> None:
        if self.before_edit is not None:
            self.before_edit(chat_id=chat_id, message_id=message_id, text=text)
        if self.fail_edit is not None:
            self.fail_edit(chat_id=chat_id, message_id=message_id, text=text)
        await super().edit_message_text(chat_id, message_id, text)


class ControlledProvider:
    def __init__(self) -> None:
        self.name = "fake"
        self.requests: list[ChatRequest] = []
        self.request_events: dict[int, list[StreamEvent]] = {}
        self.first_request_started = asyncio.Event()
        self.blocked_request_started = asyncio.Event()
        self.paused_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()
        self.release_blocked_request = asyncio.Event()
        self.release_paused_request = asyncio.Event()
        self.block_first_request = False
        self.block_request_numbers: set[int] = set()
        self.pause_after_first_event_requests: set[int] = set()
        self._models = [
            ModelSpec(provider="fake", model_id="default-model", aliases=("o",), supports_images=True, supports_tools=True),
            ModelSpec(provider="fake", model_id="alt-model", aliases=("alt",), supports_images=True),
        ]

    def get_models(self) -> list[ModelSpec]:
        return list(self._models)

    async def stream_reply(self, request: ChatRequest):
        request_number = len(self.requests) + 1
        self.requests.append(request)
        if request_number == 1 and self.block_first_request:
            self.first_request_started.set()
            await self.release_first_request.wait()
        elif request_number in self.block_request_numbers:
            self.blocked_request_started.set()
            await self.release_blocked_request.wait()
        events = self.request_events.get(
            request_number,
            [
                StreamEvent(kind="text_delta", text=f"reply {request_number}"),
                StreamEvent(kind="done", text=f"reply {request_number}"),
            ],
        )
        for index, event in enumerate(events):
            yield event
            if index == 0 and request_number in self.pause_after_first_event_requests:
                self.paused_request_started.set()
                await self.release_paused_request.wait()


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "bot.sqlite3"
        self.storage = Storage(
            self.db_path,
            default_model_alias="o",
            default_reply_mode="auto",
            default_skip_prefix="//",
        )
        self.provider = ControlledProvider()
        self.registry = ProviderRegistry([self.provider])
        self.service = ChatService(
            storage=self.storage,
            registry=self.registry,
            system_prompt="system",
            owner_user_ids=(),
            conversation_timeout_seconds=300,
            render_limit=3900,
            render_edit_interval_seconds=0.0,
            safety_identifier_salt="test-safety-salt-1234",
        )
        self.api = FakeTelegramAPI()
        self.settings = make_settings()

    def tearDown(self) -> None:
        self.storage.close()
        self.tempdir.cleanup()

    async def test_plain_recent_follow_up_reuses_same_conversation(self) -> None:
        await self._send_plain(message_id=1, text="first")
        await self._send_plain(message_id=2, text="second")

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(
            self.provider.requests[0].safety_identifier,
            hashlib.sha256(b"test-safety-salt-1234200").hexdigest(),
        )
        self.assertEqual([message.role for message in self.provider.requests[1].conversation], ["user", "assistant", "user"])
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["first", "reply 1", "second"])

    async def test_plain_message_after_timeout_starts_new_conversation(self) -> None:
        await self._send_plain(message_id=1, text="first")
        self.storage._conn.execute(
            "UPDATE messages SET created_at = ? WHERE conversation_id = ?",
            ("2000-01-01T00:00:00+00:00", 1),
        )
        self.storage._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", 1),
        )
        self.storage._conn.commit()

        await self._send_plain(message_id=2, text="second")

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["second"])

    async def test_plain_message_after_old_message_timestamps_uses_recent_conversation_activity(self) -> None:
        await self._send_plain(message_id=1, text="first")
        self.storage._conn.execute(
            "UPDATE messages SET created_at = ? WHERE conversation_id = ?",
            ("2000-01-01T00:00:00+00:00", 1),
        )
        self.storage._conn.commit()

        await self._send_plain(message_id=2, text="second")

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_reply_to_latest_assistant_continues_same_conversation_after_timeout(self) -> None:
        await self._send_plain(message_id=1, text="first")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        self.storage._conn.execute(
            "UPDATE messages SET created_at = ? WHERE conversation_id = ?",
            ("2000-01-01T00:00:00+00:00", 1),
        )
        self.storage._conn.commit()

        await self._send_plain(
            message_id=2,
            text="follow-up",
            reply_to_message_id=assistant_telegram_message_id,
            reply_to_bot=True,
        )

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["first", "reply 1", "follow-up"])

    async def test_reply_to_older_assistant_forks_new_conversation(self) -> None:
        await self._send_plain(message_id=1, text="first")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_plain(
            message_id=2,
            text="follow-up",
            reply_to_message_id=assistant_telegram_message_id,
            reply_to_bot=True,
        )
        await self._send_plain(
            message_id=3,
            text="branch",
            reply_to_message_id=assistant_telegram_message_id,
            reply_to_bot=True,
        )

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(
            [message.content for message in self.provider.requests[2].conversation],
            ["first", "reply 1", "branch"],
        )

    async def test_multiple_replies_to_same_old_assistant_create_separate_branches(self) -> None:
        await self._send_plain(message_id=1, text="first")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_plain(message_id=2, text="follow-up", reply_to_message_id=assistant_telegram_message_id, reply_to_bot=True)
        await self._send_plain(message_id=3, text="branch one", reply_to_message_id=assistant_telegram_message_id, reply_to_bot=True)
        await self._send_plain(message_id=4, text="branch two", reply_to_message_id=assistant_telegram_message_id, reply_to_bot=True)

        self.assertEqual(self._conversation_ids(), [1, 2, 3])
        self.assertEqual(
            [message.content for message in self.provider.requests[2].conversation],
            ["first", "reply 1", "branch one"],
        )
        self.assertEqual(
            [message.content for message in self.provider.requests[3].conversation],
            ["first", "reply 1", "branch two"],
        )

    async def test_plain_follow_up_after_branching_uses_most_recent_branch(self) -> None:
        await self._send_plain(message_id=1, text="first")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_plain(message_id=2, text="follow-up", reply_to_message_id=assistant_telegram_message_id, reply_to_bot=True)
        await self._send_plain(message_id=3, text="branch", reply_to_message_id=assistant_telegram_message_id, reply_to_bot=True)
        await self._send_plain(message_id=4, text="plain")

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(
            [message.content for message in self.provider.requests[3].conversation],
            ["first", "reply 1", "branch", "reply 3", "plain"],
        )

    async def test_new_reply_to_unused_seed_materializes_same_conversation(self) -> None:
        await self._send_system_prompt(message_id=1, prompt="be concise")

        self.assertEqual(len(self.provider.requests), 0)
        self.assertEqual(self._conversation_ids(), [1])

        await self._send_new(message_id=2, text="hello", reply_to_message_id=1)

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(self.provider.requests[0].system_prompt, "be concise")
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["hello"])

    async def test_new_without_seed_starts_fresh_default_conversation(self) -> None:
        await self._send_plain(message_id=1, text="old")
        await self._send_new(message_id=2, text="fresh")

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(self.provider.requests[1].system_prompt, "system")
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["fresh"])

    async def test_choose_model_reply_to_seed_preserves_seed_metadata(self) -> None:
        await self._send_system_prompt(message_id=1, prompt="be concise")
        await self._send_choose_model(message_id=2, alias="o", text="hello", reply_to_message_id=1)

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(self.provider.requests[0].model.model_id, "default-model")
        self.assertEqual(self.provider.requests[0].system_prompt, "be concise")

    async def test_choose_model_search_suffix_sets_requested_tools(self) -> None:
        await self._send_choose_model(message_id=1, alias="o-s", text="hello")

        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self.provider.requests[0].model.model_id, "default-model")
        self.assertEqual(self.provider.requests[0].requested_tools, ("search",))

    def test_command_help_text_for_new_mentions_default_model_and_seed_materialization(self) -> None:
        text = self.service.command_help_text(topic="new", settings=self.settings)
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("/new <content>", text)
        self.assertIn("chat default model alias o", text)
        self.assertIn("seed state whose visible history is empty", text)

    def test_command_help_text_for_c_mentions_attachable_non_user_state(self) -> None:
        text = self.service.command_help_text(topic="c", settings=self.settings)
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("/c <alias> <content>", text)
        self.assertIn("attachable non-user state", text)

    def test_command_help_text_for_unknown_topic_returns_none(self) -> None:
        self.assertIsNone(self.service.command_help_text(topic="mode", settings=self.settings))

    async def test_choose_model_reply_to_seed_with_new_alias_forks_branch(self) -> None:
        await self._send_system_prompt(message_id=1, prompt="be concise")
        await self._send_choose_model(message_id=2, alias="alt", text="hello", reply_to_message_id=1)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(self.provider.requests[0].model.model_id, "alt-model")
        self.assertEqual(self.provider.requests[0].system_prompt, "be concise")

    async def test_system_prompt_unattached_creates_seed_only(self) -> None:
        await self._send_system_prompt(message_id=1, prompt="be concise")

        self.assertEqual(len(self.provider.requests), 0)
        self.assertEqual(self._conversation_ids(), [1])
        conversation = self.storage.conversations.get_conversation(1)
        assert conversation is not None
        self.assertEqual(conversation.system_prompt_override, "be concise")
        tip = self._conversation_tip(conversation.id)
        assert tip is not None
        self.assertEqual(tip.message_type, "seed")
        self.assertEqual(tip.telegram_message_id, 1)

    async def test_plain_non_reply_after_seed_within_timeout_materializes_seed(self) -> None:
        await self._send_system_prompt(message_id=1, prompt="be concise")
        await self._send_plain(message_id=2, text="hello")

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(self.provider.requests[0].system_prompt, "be concise")
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["hello"])

    async def test_system_prompt_reply_to_assistant_creates_seed_branch_without_generating(self) -> None:
        await self._send_plain(message_id=1, text="hello")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])

        await self._send_system_prompt(message_id=2, prompt="be formal", reply_to_message_id=assistant_telegram_message_id)

        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._conversation_ids(), [1, 2])
        conversation = self.storage.conversations.get_conversation(2)
        assert conversation is not None
        self.assertEqual(conversation.system_prompt_override, "be formal")
        tip = self._conversation_tip(conversation.id)
        assert tip is not None
        self.assertEqual(tip.message_type, "seed")
        self.assertEqual(tip.parent_message_id, 2)

    async def test_system_prompt_reply_to_user_ended_state_creates_seeded_branch_and_generates(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=77,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="draft",
            status="complete",
        )
        await self._send_system_prompt(message_id=2, prompt="be formal", reply_to_message_id=77)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self.provider.requests[0].system_prompt, "be formal")
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["draft"])

    async def test_system_prompt_album_seed_is_attachable_from_earlier_album_item(self) -> None:
        await self._send_system_prompt(
            message_id=11,
            prompt="be concise",
            source_message_ids=(10, 11),
        )

        await self._send_plain(message_id=12, text="hello", reply_to_message_id=10)

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(self.provider.requests[0].system_prompt, "be concise")
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["hello"])

    async def test_plain_reply_to_seed_over_user_merges_underlying_user_turn(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=77,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="draft",
            status="complete",
        )
        await self._send_system_prompt(message_id=2, prompt="be formal", reply_to_message_id=77)
        await self._send_plain(message_id=3, text="follow-up", reply_to_message_id=2)

        self.assertEqual(self._conversation_ids(), [1, 2, 3])
        self.assertEqual(self.provider.requests[1].system_prompt, "be formal")
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["draft\n\nfollow-up"])

    async def test_plain_direct_reply_to_other_users_user_state_is_noop(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=999, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=77,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="draft",
            status="complete",
        )
        await self._send_plain(message_id=2, text="follow-up", reply_to_message_id=77)

        self.assertEqual(len(self.provider.requests), 0)
        self.assertEqual(len(self.api.sent_messages), 0)

    async def test_plain_direct_reply_to_same_users_user_state_merges_into_replacement_turn(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=77,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="draft",
            status="complete",
        )
        await self._send_plain(message_id=2, text="follow-up", reply_to_message_id=77)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["draft\n\nfollow-up"])

    async def test_messages_are_queued_while_selected_branch_is_streaming(self) -> None:
        self.provider.block_first_request = True
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")

        conversation = self.storage.conversations.get_conversation(1)
        assert conversation is not None
        self.assertTrue(self._conversation_is_streaming(conversation.id))
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._pending_count(1), 1)

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        conversation = self.storage.conversations.get_conversation(1)
        assert conversation is not None
        self.assertFalse(self._conversation_is_streaming(conversation.id))
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self._pending_count(1), 0)

    async def test_reply_to_still_streaming_assistant_message_stays_attached_to_same_branch(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_plain(
            message_id=2,
            text="second",
            reply_to_message_id=assistant_telegram_message_id,
            reply_to_bot=True,
        )

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._pending_count(1), 1)

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_reply_to_queued_user_message_merges_into_same_pending_turn(self) -> None:
        self.provider.block_first_request = True
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")
        await self._send_plain(message_id=3, text="clarification", reply_to_message_id=2)

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._pending_count(1), 1)

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second\n\nclarification"],
        )

    async def test_image_reply_to_queued_user_message_merges_into_same_pending_turn(self) -> None:
        self.provider.block_first_request = True
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.download_started["img-1"] = asyncio.Event()

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")
        reply_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="clarification",
                images=(ImageRef(file_id="img-1", mime_type="image/jpeg"),),
                reply_to_message_id=2,
            )
        )

        await asyncio.sleep(0)
        self.assertFalse(self.api.download_started["img-1"].is_set())

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(self.api.download_started["img-1"].wait(), timeout=1.0)
        await asyncio.wait_for(reply_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        queued_message = self.provider.requests[1].conversation[-1]
        self.assertEqual(queued_message.content, "second\n\nclarification")
        image_parts = [part for part in queued_message.parts if part.kind == "image"]
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(image_parts[0].image.data if image_parts[0].image is not None else None, b"image-one")

    async def test_plain_message_while_long_stream_exceeds_timeout_still_queues_on_active_conversation(self) -> None:
        self.provider.block_first_request = True
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        assistant_message = self.storage.conversations.get_latest_message(1)
        assert assistant_message is not None
        self.storage._conn.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", assistant_message.id),
        )
        self.storage._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", 1),
        )
        self.storage._conn.commit()

        await self._send_plain(message_id=2, text="second")

        self.assertEqual(self._conversation_ids(), [1])
        self.assertEqual(self._pending_count(1), 1)

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_recover_interrupted_assistant_turn_marks_failed_and_runs_pending_follow_up(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_message = self.storage.conversations.get_latest_message(1)
        assert assistant_message is not None
        self.assertEqual(assistant_message.message_type, "assistant")
        self.assertEqual(assistant_message.status, "streaming")

        await self._send_plain(message_id=2, text="second")
        self.assertEqual(self._pending_count(1), 1)

        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task

        await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message.id,
        )

        recovered_assistant = self.storage.conversations.get_message(assistant_message.id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "failed")
        self.assertIn("[interrupted: bot restarted before reply completed]", recovered_assistant.content)
        self.assertEqual(self._pending_count(1), 0)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertTrue(self.api.edits)
        self.assertIn("interrupted: bot restarted", str(self.api.edits[-1]["text"]))

    async def test_recover_interrupted_assistant_turn_without_existing_render_sends_notice(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=1,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
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
            model_id="default-model",
            model_alias="o",
            content="partial",
            status="streaming",
        )

        recovered = await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message_id,
            reply_to_message_id=1,
        )

        self.assertTrue(recovered)
        recovered_assistant = self.storage.conversations.get_message(assistant_message_id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "failed")
        self.assertTrue(self.api.sent_messages)
        self.assertIn("interrupted: bot restarted", str(self.api.sent_messages[-1]["text"]))

    async def test_recover_final_assistant_turn_runs_pending_follow_up(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_message = self.storage.conversations.get_latest_message(1)
        assert assistant_message is not None
        self.assertEqual(assistant_message.message_type, "assistant")
        self.assertEqual(assistant_message.status, "streaming")

        await self._send_plain(message_id=2, text="second")
        self.assertEqual(self._pending_count(1), 1)

        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task

        self.storage.conversations.update_message(
            assistant_message.id,
            content="reply 1",
            status="complete",
        )

        await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message.id,
        )

        recovered_assistant = self.storage.conversations.get_message(assistant_message.id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "complete")
        self.assertEqual(recovered_assistant.content, "reply 1")
        self.assertEqual(self._pending_count(1), 0)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_recover_final_pending_assistant_rerenders_final_reply(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=1,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="format",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="**bold**",
            status="streaming",
        )
        self.storage.conversations.link_telegram_message(
            chat_id=100,
            telegram_message_id=1000,
            logical_message_id=assistant_message_id,
            part_index=0,
        )
        self.storage.inbox.enqueue_messages(messages=[make_message(message_id=1, text="format")])
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=assistant_message_id,
            phase="final_pending",
            final_status="complete",
            reply_text="**bold**",
            reasoning_blocks=(),
            render_markdown=True,
        )

        await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message_id,
            reply_to_message_id=1,
            final_render_phase="final_pending",
            final_render_status="complete",
            final_render_reply_text="**bold**",
            final_render_reasoning_blocks=(),
            final_render_markdown=True,
        )

        recovered_assistant = self.storage.conversations.get_message(assistant_message_id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "complete")
        self.assertEqual(recovered_assistant.content, "**bold**")
        self.assertTrue(self.api.edits)
        self.assertEqual(self.api.edits[-1]["text"], "[o] bold")

    async def test_recover_final_pending_assistant_without_existing_links_sends_final_reply(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=1,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="final only",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="final only",
            status="streaming",
        )
        self.storage.inbox.enqueue_messages(messages=[make_message(message_id=1, text="final only")])
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=assistant_message_id,
            phase="final_pending",
            final_status="complete",
            reply_text="final only",
            reasoning_blocks=(),
            render_markdown=True,
        )

        recovered = await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message_id,
            reply_to_message_id=1,
            final_render_phase="final_pending",
            final_render_status="complete",
            final_render_reply_text="final only",
            final_render_reasoning_blocks=(),
            final_render_markdown=True,
        )

        self.assertTrue(recovered)
        recovered_assistant = self.storage.conversations.get_message(assistant_message_id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "complete")
        self.assertEqual(recovered_assistant.content, "final only")
        self.assertTrue(self.api.sent_messages)
        self.assertEqual(self.api.sent_messages[-1]["text"], "[o] final only")

    async def test_recover_final_pending_assistant_retries_plain_text_after_markdown_failure(self) -> None:
        conversation = self.storage.conversations.create_conversation(chat_id=100, user_id=200, model_alias="o")
        user_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=1,
            message_type="user",
            parent_message_id=None,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="format",
            status="complete",
        )
        assistant_message_id = self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=100,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=user_message_id,
            provider="fake",
            model_id="default-model",
            model_alias="o",
            content="**bold**",
            status="streaming",
        )
        self.storage.conversations.link_telegram_message(
            chat_id=100,
            telegram_message_id=1000,
            logical_message_id=assistant_message_id,
            part_index=0,
        )
        self.storage.inbox.enqueue_messages(messages=[make_message(message_id=1, text="format")])
        self.storage.inbox.mark_updates_realized(
            update_ids=(1,),
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=assistant_message_id,
            phase="final_pending",
            final_status="complete",
            reply_text="**bold**",
            reasoning_blocks=(),
            render_markdown=True,
        )

        failed_formatted_edit = False

        def fail_edit(*, chat_id: int, message_id: int, text: str | RichText) -> None:
            nonlocal failed_formatted_edit
            _ = (chat_id, message_id)
            rendered_text, _entities = RichText.coerce(text).to_telegram()
            if rendered_text == "[o] bold" and not failed_formatted_edit:
                failed_formatted_edit = True
                raise RuntimeError("markdown rejected")

        self.api = InspectingTelegramAPI(fail_edit=fail_edit)

        await self.service.recover_interrupted_assistant_turn(
            api=self.api,
            assistant_message_id=assistant_message_id,
            reply_to_message_id=1,
            final_render_phase="final_pending",
            final_render_status="complete",
            final_render_reply_text="**bold**",
            final_render_reasoning_blocks=(),
            final_render_markdown=True,
        )

        recovered_assistant = self.storage.conversations.get_message(assistant_message_id)
        assert recovered_assistant is not None
        self.assertEqual(recovered_assistant.status, "complete")
        self.assertTrue(failed_formatted_edit)
        self.assertTrue(self.api.edits)
        self.assertEqual(self.api.edits[-1]["text"], "[o] **bold**")

    async def test_new_reply_to_still_streaming_assistant_starts_fresh_immediately(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await asyncio.wait_for(
            self._send_new(
                message_id=2,
                text="fresh",
                reply_to_message_id=assistant_telegram_message_id,
            ),
            timeout=1.0,
        )

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["fresh"])

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

    async def test_system_prompt_reply_to_still_streaming_assistant_creates_seed_immediately(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await asyncio.wait_for(
            self._send_system_prompt(
                message_id=2,
                prompt="be formal",
                reply_to_message_id=assistant_telegram_message_id,
            ),
            timeout=1.0,
        )

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(len(self.provider.requests), 1)
        conversation = self.storage.conversations.get_conversation(2)
        assert conversation is not None
        self.assertEqual(conversation.system_prompt_override, "be formal")
        tip = self._conversation_tip(conversation.id)
        assert tip is not None
        self.assertEqual(tip.message_type, "seed")

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

    async def test_plain_reply_to_seed_over_streaming_assistant_waits_for_completion(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_system_prompt(
            message_id=2,
            prompt="be formal",
            reply_to_message_id=assistant_telegram_message_id,
        )

        branch_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="second",
                reply_to_message_id=2,
            )
        )

        await asyncio.sleep(0)
        self.assertFalse(branch_task.done())
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._conversation_ids(), [1, 2])

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(branch_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.provider.requests[1].system_prompt, "be formal")
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_deferred_seed_reply_does_not_block_later_fresh_new_request(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_system_prompt(
            message_id=2,
            prompt="be formal",
            reply_to_message_id=assistant_telegram_message_id,
        )

        branch_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="second",
                reply_to_message_id=2,
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(branch_task.done())

        fresh_task = asyncio.create_task(self._send_new(message_id=4, text="fresh"))
        await asyncio.wait_for(fresh_task, timeout=1.0)

        self.assertFalse(branch_task.done())
        self.assertEqual(self._conversation_ids(), [1, 2, 3])
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["fresh"])

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(branch_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 3)
        self.assertEqual(self.provider.requests[2].system_prompt, "be formal")
        self.assertEqual(
            [message.content for message in self.provider.requests[2].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_choose_model_reply_to_seed_over_streaming_assistant_waits_for_completion(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_system_prompt(
            message_id=2,
            prompt="be formal",
            reply_to_message_id=assistant_telegram_message_id,
        )

        branch_task = asyncio.create_task(
            self._send_choose_model(
                message_id=3,
                alias="alt",
                text="second",
                reply_to_message_id=2,
            )
        )

        await asyncio.sleep(0)
        self.assertFalse(branch_task.done())
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._conversation_ids(), [1, 2])

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(branch_task, timeout=1.0)

        self.assertEqual(self._conversation_ids(), [1, 2, 3])
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.provider.requests[1].model.model_id, "alt-model")
        self.assertEqual(self.provider.requests[1].system_prompt, "be formal")
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_choose_model_reply_to_still_streaming_assistant_waits_for_anchor_completion(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        branch_task = asyncio.create_task(
            self._send_choose_model(
                message_id=2,
                alias="alt",
                text="second",
                reply_to_message_id=assistant_telegram_message_id,
            )
        )

        await asyncio.sleep(0)
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self._conversation_ids(), [1])

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(branch_task, timeout=1.0)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(self.provider.requests[1].model.model_id, "alt-model")
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "second"],
        )

    async def test_reply_to_older_assistant_starts_new_branch_while_latest_branch_streams(self) -> None:
        await self._send_plain(message_id=1, text="first")
        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])

        self.provider.block_request_numbers = {2}
        latest_branch_task = asyncio.create_task(
            self._send_plain(
                message_id=2,
                text="follow-up",
                reply_to_message_id=assistant_telegram_message_id,
                reply_to_bot=True,
            )
        )
        await asyncio.wait_for(self.provider.blocked_request_started.wait(), timeout=1.0)

        await self._send_plain(
            message_id=3,
            text="branch",
            reply_to_message_id=assistant_telegram_message_id,
            reply_to_bot=True,
        )

        first_conversation = self.storage.conversations.get_conversation(1)
        second_conversation = self.storage.conversations.get_conversation(2)
        assert first_conversation is not None
        assert second_conversation is not None
        self.assertTrue(self._conversation_is_streaming(first_conversation.id))
        self.assertFalse(self._conversation_is_streaming(second_conversation.id))
        self.assertEqual(
            [message.content for message in self.provider.requests[2].conversation],
            ["first", "reply 1", "branch"],
        )

        self.provider.release_blocked_request.set()
        await asyncio.wait_for(latest_branch_task, timeout=1.0)

    async def test_multiple_queued_messages_merge_into_one_canonical_turn(self) -> None:
        self.provider.block_first_request = True
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")
        await self._send_plain(message_id=3, text="third")

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        expected_merged = (
            "Additional user messages sent while you were replying:\n\n"
            "1.\n"
            "second\n\n"
            "2.\n"
            "third"
        )
        self.assertEqual([message.content for message in self.provider.requests[1].conversation], ["start", "reply 1", expected_merged])
        self.assertEqual(self.api.sent_messages[-1]["reply_to_message_id"], 3)

        user_messages = self._message_rows(message_type="user", conversation_id=1)
        self.assertEqual([message.content for message in user_messages], ["start", expected_merged])

    async def test_queued_image_follow_up_joins_existing_pending_batch(self) -> None:
        self.provider.block_first_request = True
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.download_started["img-1"] = asyncio.Event()

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")
        image_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="third",
                images=(ImageRef(file_id="img-1", mime_type="image/jpeg"),),
            )
        )

        await asyncio.sleep(0)
        self.assertFalse(self.api.download_started["img-1"].is_set())

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(self.api.download_started["img-1"].wait(), timeout=1.0)
        await asyncio.wait_for(image_task, timeout=1.0)

        expected_merged = (
            "Additional user messages sent while you were replying:\n\n"
            "1.\n"
            "second\n\n"
            "2.\n"
            "third"
        )
        self.assertEqual(len(self.provider.requests), 2)
        queued_message = self.provider.requests[1].conversation[-1]
        self.assertEqual(queued_message.content, expected_merged)
        image_parts = [part for part in queued_message.parts if part.kind == "image"]
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(image_parts[0].image.data if image_parts[0].image is not None else None, b"image-one")

    async def test_queued_text_messages_survive_later_queued_image_download_failure(self) -> None:
        self.provider.block_first_request = True

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="second")
        await self._send_plain(
            message_id=3,
            text="broken image",
            images=(ImageRef(file_id="missing", mime_type="image/jpeg"),),
        )
        await self._send_plain(message_id=4, text="third")

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        expected_merged = (
            "Additional user messages sent while you were replying:\n\n"
            "1.\n"
            "second\n\n"
            "2.\n"
            "third"
        )
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["start", "reply 1", expected_merged],
        )
        error_replies = [message for message in self.api.sent_messages if message["reply_to_message_id"] == 3]
        self.assertEqual(
            [message["text"] for message in error_replies],
            ["Failed to download the image from Telegram. Please resend it and try again."],
        )
        self.assertEqual(self._pending_count(1), 0)

    async def test_message_with_text_and_multiple_images_is_built_into_request(self) -> None:
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.file_bytes["img-2"] = b"image-two"
        await self._send_plain(
            message_id=1,
            text="look",
            images=(
                ImageRef(file_id="img-1", mime_type="image/jpeg"),
                ImageRef(file_id="img-2", mime_type="image/png"),
            ),
        )

        first_message = self.provider.requests[0].conversation[0]
        self.assertEqual(first_message.content, "look")
        self.assertEqual(first_message.images, ())
        image_parts = [part for part in first_message.parts if part.kind == "image"]
        self.assertEqual(len(image_parts), 2)
        self.assertEqual(image_parts[0].image.data if image_parts[0].image is not None else None, b"image-one")
        self.assertEqual(image_parts[1].image.data if image_parts[1].image is not None else None, b"image-two")

    async def test_message_parts_preserve_caption_image_order_in_request(self) -> None:
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.file_bytes["img-2"] = b"image-two"
        incoming_parts = (
            ContentPart(kind="text", text="caption one"),
            ContentPart(kind="image", image=ImageRef(file_id="img-1", mime_type="image/jpeg")),
            ContentPart(kind="text", text="caption two"),
            ContentPart(kind="image", image=ImageRef(file_id="img-2", mime_type="image/png")),
        )

        await self._send_plain(
            message_id=1,
            text="caption one\n\ncaption two",
            images=(
                ImageRef(file_id="img-1", mime_type="image/jpeg"),
                ImageRef(file_id="img-2", mime_type="image/png"),
            ),
            parts=incoming_parts,
        )

        first_message = self.provider.requests[0].conversation[0]
        self.assertEqual([part.kind for part in first_message.parts], ["text", "image", "text", "image"])
        self.assertEqual([part.text for part in first_message.parts if part.kind == "text"], ["caption one", "caption two"])

    async def test_image_download_failure_sends_retryable_error_and_does_not_create_turn(self) -> None:
        await self._send_plain(
            message_id=1,
            text="look",
            images=(ImageRef(file_id="missing", mime_type="image/jpeg"),),
        )

        self.assertEqual(len(self.provider.requests), 0)
        self.assertEqual(self._conversation_ids(), [])
        self.assertEqual(
            self.api.sent_messages,
            [
                {
                    "chat_id": 100,
                    "message_id": 1000,
                    "text": "Failed to download the image from Telegram. Please resend it and try again.",
                    "entities": [],
                    "reply_to_message_id": 1,
                }
            ],
        )

    async def test_deferred_image_turn_reuses_downloaded_input(self) -> None:
        self.provider.pause_after_first_event_requests = {1}
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.download_started["img-1"] = asyncio.Event()

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_system_prompt(
            message_id=2,
            prompt="be formal",
            reply_to_message_id=assistant_telegram_message_id,
        )

        second_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="describe",
                images=(ImageRef(file_id="img-1", mime_type="image/jpeg"),),
                reply_to_message_id=2,
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(self.api.download_started["img-1"].is_set())
        self.assertEqual(self.api.download_counts.get("img-1", 0), 0)

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(self.api.download_started["img-1"].wait(), timeout=1.0)
        await asyncio.wait_for(second_task, timeout=1.0)

        self.assertEqual(self.api.download_counts["img-1"], 1)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "reply 1", "describe"],
        )

    async def test_later_same_user_message_cannot_overtake_blocked_image_turn(self) -> None:
        self.api.file_bytes["img-1"] = b"image-one"
        self.api.download_started["img-1"] = asyncio.Event()
        self.api.download_release["img-1"] = asyncio.Event()

        first_task = asyncio.create_task(
            self._send_plain(
                message_id=1,
                text="first image",
                images=(ImageRef(file_id="img-1", mime_type="image/jpeg"),),
            )
        )
        await asyncio.wait_for(self.api.download_started["img-1"].wait(), timeout=1.0)

        second_task = asyncio.create_task(self._send_plain(message_id=2, text="second"))
        await asyncio.sleep(0)
        self.assertEqual(len(self.provider.requests), 0)

        self.api.download_release["img-1"].set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(second_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([message.content for message in self.provider.requests[0].conversation], ["first image"])
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first image", "reply 1", "second"],
        )

    async def test_reply_to_earlier_media_group_item_resolves_logical_user_state(self) -> None:
        self.api.file_bytes["album-1"] = b"album-image-one"
        self.api.file_bytes["album-2"] = b"album-image-two"

        await self._send_plain(
            message_id=11,
            text="album",
            images=(
                ImageRef(file_id="album-1", mime_type="image/jpeg"),
                ImageRef(file_id="album-2", mime_type="image/png"),
            ),
            source_message_ids=(10, 11),
        )

        await self._send_plain(message_id=12, text="follow-up", reply_to_message_id=10)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["album\n\nfollow-up"],
        )

    async def test_queued_media_group_retains_links_for_all_source_message_ids(self) -> None:
        self.provider.block_first_request = True
        self.api.file_bytes["album-1"] = b"album-image-one"
        self.api.file_bytes["album-2"] = b"album-image-two"

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(
            message_id=11,
            text="album",
            images=(
                ImageRef(file_id="album-1", mime_type="image/jpeg"),
                ImageRef(file_id="album-2", mime_type="image/png"),
            ),
            source_message_ids=(10, 11),
        )

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        await self._send_plain(message_id=12, text="follow-up", reply_to_message_id=10)

        self.assertEqual(self._conversation_ids(), [1, 2])
        self.assertEqual(
            [message.content for message in self.provider.requests[2].conversation],
            ["start", "reply 1", "album\n\nfollow-up"],
        )

    async def test_reasoning_blocks_are_rendered_above_reply_text(self) -> None:
        self.provider.request_events[1] = [
            StreamEvent(kind="reasoning_delta", text="First step"),
            StreamEvent(kind="reasoning_delimiter"),
            StreamEvent(kind="reasoning_delta", text="Second step"),
            StreamEvent(kind="text_delta", text="Final answer"),
            StreamEvent(kind="done", text="Final answer"),
        ]

        await self._send_plain(message_id=1, text="think")

        final_render = self.api.edits[-1] if self.api.edits else self.api.sent_messages[-1]
        self.assertEqual(final_render["text"], "[o] First step\nSecond step\nFinal answer")
        self.assertEqual(
            final_render["entities"],
            [
                {"type": "expandable_blockquote", "offset": 4, "length": 10},
                {"type": "expandable_blockquote", "offset": 15, "length": 11},
            ],
        )

    async def test_streaming_markdown_body_is_plain_until_final_render(self) -> None:
        self.provider.request_events[1] = [
            StreamEvent(kind="text_delta", text="**bold"),
            StreamEvent(kind="text_delta", text="**"),
            StreamEvent(kind="done", text="**bold**"),
        ]

        await self._send_plain(message_id=1, text="format")

        first_render = self.api.sent_messages[0]
        self.assertEqual(first_render["text"], "[o] **bold")
        self.assertEqual(first_render["entities"], [])

        final_render = self.api.edits[-1] if self.api.edits else self.api.sent_messages[-1]
        self.assertEqual(final_render["text"], "[o] bold")
        self.assertEqual(final_render["entities"], [{"type": "bold", "offset": 4, "length": 4}])

    async def test_streaming_state_is_persisted_before_telegram_send_and_edit(self) -> None:
        send_checks = 0
        edit_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal send_checks
            _ = (chat_id, text, reply_to_message_id)
            assistant_message = self.storage.conversations.get_latest_message(1)
            assert assistant_message is not None
            self.assertEqual(assistant_message.message_type, "assistant")
            self.assertEqual(assistant_message.status, "streaming")
            self.assertEqual(assistant_message.content, "one")
            self.assertEqual(
                self.storage.conversations.list_linked_telegram_message_ids(
                    logical_message_id=assistant_message.id,
                ),
                [],
            )
            send_checks += 1

        def before_edit(*, chat_id: int, message_id: int, text: str | RichText) -> None:
            nonlocal edit_checks
            _ = (chat_id, text)
            assistant_message = self.storage.conversations.get_latest_message(1)
            assert assistant_message is not None
            self.assertEqual(assistant_message.message_type, "assistant")
            self.assertEqual(assistant_message.status, "streaming")
            self.assertEqual(assistant_message.content, "onetwo")
            self.assertEqual(
                self.storage.conversations.list_linked_telegram_message_ids(
                    logical_message_id=assistant_message.id,
                ),
                [message_id],
            )
            edit_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send, before_edit=before_edit)
        self.provider.request_events[1] = [
            StreamEvent(kind="text_delta", text="one"),
            StreamEvent(kind="text_delta", text="two"),
            StreamEvent(kind="done", text="onetwo"),
        ]

        await self._send_plain(message_id=1, text="persist")

        self.assertEqual(send_checks, 1)
        self.assertEqual(edit_checks, 1)

    async def test_first_final_send_marks_final_pending_before_send(self) -> None:
        send_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal send_checks
            _ = (chat_id, text, reply_to_message_id)
            stored = self.storage.inbox.get_update(update_id=1)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertIsNotNone(stored.realized_assistant_message_id)
            self.assertEqual(stored.assistant_render_phase, "final_pending")
            self.assertEqual(stored.assistant_render_final_status, "complete")
            self.assertEqual(stored.assistant_render_reply_text, "final only")
            self.assertTrue(stored.assistant_render_markdown)
            self.assertEqual(
                self.storage.conversations.list_linked_telegram_message_ids(
                    logical_message_id=stored.realized_assistant_message_id,
                ),
                [],
            )
            send_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send)
        self.provider.request_events[1] = [
            StreamEvent(kind="done", text="final only"),
        ]
        incoming_message = make_message(message_id=1, text="final only")
        self.storage.inbox.enqueue_messages(messages=[incoming_message])

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming_message,
            settings=self.settings,
            action=ChatAction(content="final only", intent="plain"),
            inbox_update_ids=(1,),
        )

        self.assertEqual(send_checks, 1)

    async def test_pending_turn_final_send_marks_assistant_render_state_before_send(self) -> None:
        self.provider.block_first_request = True
        self.provider.request_events[2] = [
            StreamEvent(kind="done", text="pending final"),
        ]
        pending_send_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal pending_send_checks
            _ = (chat_id, reply_to_message_id)
            rendered_text, _entities = RichText.coerce(text).to_telegram()
            if rendered_text != "[o] pending final":
                return
            assistant_message = self.storage.conversations.get_latest_message(1)
            assert assistant_message is not None
            self.assertEqual(assistant_message.message_type, "assistant")
            state = self.storage.inbox.get_assistant_render_state(
                assistant_message_id=assistant_message.id,
            )
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.phase, "final_pending")
            self.assertEqual(state.final_status, "complete")
            self.assertEqual(state.reply_text, "pending final")
            self.assertTrue(state.render_markdown)
            pending_send_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send)
        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="queued follow-up")

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        self.assertEqual(pending_send_checks, 1)
        assistant_message = self.storage.conversations.get_latest_message(1)
        assert assistant_message is not None
        self.assertIsNone(
            self.storage.inbox.get_assistant_render_state(
                assistant_message_id=assistant_message.id,
            )
        )

    async def test_pre_realization_error_reply_marks_inbox_progress(self) -> None:
        send_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal send_checks
            _ = (chat_id, text, reply_to_message_id)
            stored = self.storage.inbox.get_update(update_id=1)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertIsNotNone(stored.reply_started_at)
            self.assertIsNone(stored.reply_sent_at)
            self.assertIsNone(stored.realized_assistant_message_id)
            send_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send)
        incoming_message = make_message(message_id=1, text="/c missing hello")
        self.storage.inbox.enqueue_messages(messages=[incoming_message])

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming_message,
            settings=make_settings(default_model_alias="missing"),
            action=ChatAction(content="hello", intent="choose_model", model_alias="missing"),
            inbox_update_ids=(1,),
        )

        self.assertEqual(send_checks, 1)
        stored = self.storage.inbox.get_update(update_id=1)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertIsNotNone(stored.reply_started_at)
        self.assertIsNotNone(stored.reply_sent_at)
        self.assertIsNone(stored.realized_assistant_message_id)

    async def test_first_final_send_persists_each_sent_chunk_link(self) -> None:
        self.service.render_limit = 16
        send_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal send_checks
            _ = (chat_id, text, reply_to_message_id)
            stored = self.storage.inbox.get_update(update_id=1)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertIsNotNone(stored.realized_assistant_message_id)
            self.assertEqual(stored.assistant_render_phase, "final_pending")
            self.assertEqual(stored.assistant_render_reply_text, "alpha beta gamma delta epsilon")
            self.assertEqual(
                self.storage.conversations.list_linked_telegram_message_ids(
                    logical_message_id=stored.realized_assistant_message_id,
                ),
                list(range(1000, 1000 + send_checks)),
            )
            send_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send)
        self.provider.request_events[1] = [
            StreamEvent(kind="done", text="alpha beta gamma delta epsilon"),
        ]
        incoming_message = make_message(message_id=1, text="long final")
        self.storage.inbox.enqueue_messages(messages=[incoming_message])

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming_message,
            settings=self.settings,
            action=ChatAction(content="long final", intent="plain"),
            inbox_update_ids=(1,),
        )

        self.assertEqual(send_checks, 3)
        stored = self.storage.inbox.get_update(update_id=1)
        assert stored is not None
        assert stored.realized_assistant_message_id is not None
        self.assertEqual(
            self.storage.conversations.list_linked_telegram_message_ids(
                logical_message_id=stored.realized_assistant_message_id,
            ),
            [1000, 1001, 1002],
        )

    async def test_final_render_with_existing_streamed_reply_persists_new_chunk_links(self) -> None:
        self.service.render_limit = 16
        self.service.render_edit_interval_seconds = 60.0
        final_send_checks = 0

        def before_send(*, chat_id: int, text: str | RichText, reply_to_message_id: int | None) -> None:
            nonlocal final_send_checks
            _ = (chat_id, text, reply_to_message_id)
            stored = self.storage.inbox.get_update(update_id=1)
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertIsNotNone(stored.realized_assistant_message_id)
            if stored.assistant_render_phase != "final_pending":
                return
            self.assertEqual(
                self.storage.conversations.list_linked_telegram_message_ids(
                    logical_message_id=stored.realized_assistant_message_id,
                ),
                [1000] + list(range(1001, 1001 + final_send_checks)),
            )
            final_send_checks += 1

        self.api = InspectingTelegramAPI(before_send=before_send)
        self.provider.request_events[1] = [
            StreamEvent(kind="text_delta", text="alpha"),
            StreamEvent(kind="text_delta", text=" beta gamma delta epsilon"),
            StreamEvent(kind="done", text="alpha beta gamma delta epsilon"),
        ]
        incoming_message = make_message(message_id=1, text="long final")
        self.storage.inbox.enqueue_messages(messages=[incoming_message])

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming_message,
            settings=self.settings,
            action=ChatAction(content="long final", intent="plain"),
            inbox_update_ids=(1,),
        )

        self.assertEqual(final_send_checks, 2)
        stored = self.storage.inbox.get_update(update_id=1)
        assert stored is not None
        assert stored.realized_assistant_message_id is not None
        self.assertEqual(
            self.storage.conversations.list_linked_telegram_message_ids(
                logical_message_id=stored.realized_assistant_message_id,
            ),
            [1000, 1001, 1002],
        )

    async def test_final_render_stays_streaming_before_final_edit(self) -> None:
        final_edit_checks = 0

        def before_edit(*, chat_id: int, message_id: int, text: str | RichText) -> None:
            nonlocal final_edit_checks
            _ = (chat_id, message_id)
            rendered_text, _entities = RichText.coerce(text).to_telegram()
            if rendered_text != "[o] bold":
                return
            assistant_message = self.storage.conversations.get_latest_message(1)
            assert assistant_message is not None
            self.assertEqual(assistant_message.message_type, "assistant")
            self.assertEqual(assistant_message.status, "streaming")
            self.assertEqual(assistant_message.content, "**bold**")
            final_edit_checks += 1

        self.api = InspectingTelegramAPI(before_edit=before_edit)
        self.provider.request_events[1] = [
            StreamEvent(kind="text_delta", text="**bold"),
            StreamEvent(kind="text_delta", text="**"),
            StreamEvent(kind="done", text="**bold**"),
        ]

        await self._send_plain(message_id=1, text="format")

        self.assertEqual(final_edit_checks, 1)

    async def test_render_failure_releases_conversation_and_processes_pending_messages(self) -> None:
        self.provider.block_first_request = True
        self.api.send_failures_remaining = 1

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="after failure")

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        conversation = self.storage.conversations.get_conversation(1)
        assert conversation is not None
        self.assertFalse(self._conversation_is_streaming(conversation.id))
        self.assertEqual(self._pending_count(1), 0)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.api.sent_messages[-1]["reply_to_message_id"], 2)

        assistant_messages = self._message_rows(message_type="assistant", conversation_id=1)
        self.assertEqual([message.status for message in assistant_messages], ["failed", "complete"])
        self.assertEqual(assistant_messages[0].content, "reply 1")
        self.assertEqual(assistant_messages[1].content, "reply 2")

    async def test_final_render_failure_releases_conversation_and_processes_pending_messages(self) -> None:
        self.provider.block_first_request = True
        self.provider.request_events[1] = [
            StreamEvent(kind="done", text="final only"),
        ]
        self.api.send_failures_remaining = 4

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="start"))
        await asyncio.wait_for(self.provider.first_request_started.wait(), timeout=1.0)

        await self._send_plain(message_id=2, text="after failure")

        self.provider.release_first_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)

        conversation = self.storage.conversations.get_conversation(1)
        assert conversation is not None
        self.assertFalse(self._conversation_is_streaming(conversation.id))
        self.assertEqual(self._pending_count(1), 0)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.api.sent_messages[-1]["reply_to_message_id"], 2)

        assistant_messages = self._message_rows(message_type="assistant", conversation_id=1)
        self.assertEqual([message.status for message in assistant_messages], ["failed", "complete"])
        self.assertEqual(assistant_messages[0].content, "final only")
        self.assertEqual(assistant_messages[1].content, "reply 2")

    async def test_final_render_failure_releases_deferred_reply_waiting_on_completion(self) -> None:
        def fail_edit(*, chat_id: int, message_id: int, text: str | RichText) -> None:
            _ = (chat_id, message_id, text)
            raise RuntimeError("edit failed")

        self.api = InspectingTelegramAPI(fail_edit=fail_edit)
        self.service.render_edit_interval_seconds = 60.0
        self.provider.pause_after_first_event_requests = {1}
        self.provider.request_events[1] = [
            StreamEvent(kind="text_delta", text="partial"),
            StreamEvent(kind="text_delta", text=" done"),
            StreamEvent(kind="done", text="partial done"),
        ]

        first_task = asyncio.create_task(self._send_plain(message_id=1, text="first"))
        await asyncio.wait_for(self.provider.paused_request_started.wait(), timeout=1.0)

        assistant_telegram_message_id = int(self.api.sent_messages[-1]["message_id"])
        await self._send_system_prompt(
            message_id=2,
            prompt="be formal",
            reply_to_message_id=assistant_telegram_message_id,
        )

        branch_task = asyncio.create_task(
            self._send_plain(
                message_id=3,
                text="second",
                reply_to_message_id=2,
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(branch_task.done())

        self.provider.release_paused_request.set()
        await asyncio.wait_for(first_task, timeout=1.0)
        await asyncio.wait_for(branch_task, timeout=1.0)

        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.provider.requests[1].system_prompt, "be formal")
        self.assertEqual(
            [message.content for message in self.provider.requests[1].conversation],
            ["first", "partial done", "second"],
        )

        first_conversation = self.storage.conversations.get_conversation(1)
        assert first_conversation is not None
        self.assertFalse(self._conversation_is_streaming(first_conversation.id))
        assistant_messages = self._message_rows(message_type="assistant", conversation_id=1)
        self.assertEqual([message.status for message in assistant_messages], ["failed"])

    async def _send_plain(
        self,
        *,
        message_id: int,
        text: str,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] = (),
        reply_to_message_id: int | None = None,
        reply_to_bot: bool = False,
        user_id: int = 200,
        source_message_ids: tuple[int, ...] = (),
    ) -> None:
        await self._send_action(
            incoming_message=make_message(
                message_id=message_id,
                text=text,
                images=images,
                parts=parts,
                reply_to_message_id=reply_to_message_id,
                reply_to_bot=reply_to_bot,
                user_id=user_id,
                source_message_ids=source_message_ids,
            ),
            action=ChatAction(content=text, intent="plain", images=images, parts=parts),
        )

    async def _send_new(
        self,
        *,
        message_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> None:
        await self._send_action(
            incoming_message=make_message(
                message_id=message_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            ),
            action=ChatAction(content=text, intent="new"),
        )

    async def _send_choose_model(
        self,
        *,
        message_id: int,
        alias: str,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> None:
        await self._send_action(
            incoming_message=make_message(
                message_id=message_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            ),
            action=ChatAction(content=text, intent="choose_model", model_alias=alias),
        )

    async def _send_system_prompt(
        self,
        *,
        message_id: int,
        prompt: str,
        reply_to_message_id: int | None = None,
        source_message_ids: tuple[int, ...] = (),
    ) -> None:
        await self._send_action(
            incoming_message=make_message(
                message_id=message_id,
                text=f"/s {prompt}",
                reply_to_message_id=reply_to_message_id,
                source_message_ids=source_message_ids,
            ),
            action=ChatAction(content="", intent="set_system_prompt", system_prompt=prompt),
        )

    async def _send_action(self, *, incoming_message: IncomingMessage, action: ChatAction) -> None:
        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming_message,
            settings=self.settings,
            action=action,
        )

    def _conversation_ids(self) -> list[int]:
        rows = self.storage._conn.execute("SELECT id FROM conversations ORDER BY id ASC").fetchall()
        return [int(row["id"]) for row in rows]

    def _pending_count(self, conversation_id: int) -> int:
        row = self.storage._conn.execute(
            "SELECT COUNT(*) AS pending_count FROM pending_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        assert row is not None
        return int(row["pending_count"])

    def _conversation_tip(self, conversation_id: int) -> StoredMessage | None:
        return self.storage.conversations.get_conversation_tip_message(conversation_id)

    def _conversation_is_streaming(self, conversation_id: int) -> bool:
        return self.storage.conversations.is_conversation_streaming(conversation_id)

    def _message_rows(self, *, message_type: str, conversation_id: int) -> list[StoredMessage]:
        rows = self.storage._conn.execute(
            "SELECT id FROM messages WHERE message_type = ? AND conversation_id = ? ORDER BY id ASC",
            (message_type, conversation_id),
        ).fetchall()
        messages: list[StoredMessage] = []
        for row in rows:
            message = self.storage.conversations.get_message(int(row["id"]))
            assert message is not None
            messages.append(message)
        return messages


if __name__ == "__main__":
    unittest.main()
