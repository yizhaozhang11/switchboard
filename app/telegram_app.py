from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from app.chat_service import ChatService
from app.config import Config
from app.router import Router
from app.storage import InboxUpdate, Storage
from app.telegram_api import TelegramBotAPI
from app.types import (
    ChatAction,
    CommandAction,
    IgnoreAction,
    ImageRef,
    IncomingMessage,
    StoredMessage,
    build_content_parts,
)


@dataclass(frozen=True)
class InboxClaim:
    update_ids: tuple[int, ...]
    messages: tuple[IncomingMessage, ...]
    media_group_key: str | None = None


SUPPORTED_IMAGE_DOCUMENT_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class TelegramApp:
    media_group_delay_seconds = 0.3
    scheduler_poll_seconds = 0.05
    media_group_boundary_grace_seconds: float | None = None
    shutdown_poll_timeout_seconds = 1.0

    def __init__(
        self,
        *,
        config: Config,
        storage: Storage,
        service: ChatService,
        api: TelegramBotAPI,
    ) -> None:
        self.config = config
        self.storage = storage
        self.service = service
        self.api = api
        self.bot_id: int | None = None
        self.bot_username: str | None = None
        self.router = Router(bot_username=None)
        self._tasks: set[asyncio.Task] = set()
        self._scheduler_task: asyncio.Task | None = None
        self._stopping = False

    async def run(self) -> None:
        me = await self.api.get_me()
        self.bot_id = int(me["id"])
        self.bot_username = str(me.get("username") or "")
        self.router = Router(bot_username=self.bot_username)
        logging.info("Bot connected as @%s (%s)", self.bot_username, self.bot_id)

        await self._recover_claimed_updates()

        offset: int | None = None
        self._stopping = False
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        try:
            while True:
                updates = await self.api.get_updates(offset=offset, timeout=self.config.poll_timeout_seconds)
                inbox_messages: list[IncomingMessage] = []
                for update in updates:
                    update_id, incoming = self._parse_update_for_inbox(update)
                    if incoming is not None:
                        inbox_messages.append(incoming)
                    if update_id is not None:
                        offset = update_id + 1
                self.storage.inbox.enqueue_messages(messages=inbox_messages)
        finally:
            self._stopping = True
            if self._scheduler_task is not None:
                self._scheduler_task.cancel()
                await asyncio.gather(self._scheduler_task, return_exceptions=True)
                self._scheduler_task = None
            await self._cancel_tracked_tasks()
            await self._enqueue_shutdown_updates(offset=offset)

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        self._tasks.add(task)
        task.add_done_callback(self._discard_tracked_task)
        return task

    def _discard_tracked_task(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)

    async def _cancel_tracked_tasks(self) -> None:
        if not self._tasks:
            return
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run_scheduler(self) -> None:
        try:
            while not self._stopping:
                claim_entries = self.storage.inbox.claim_next_ready(
                    media_group_delay_seconds=self.media_group_delay_seconds,
                    media_group_boundary_grace_seconds=self._media_group_boundary_grace_seconds(),
                )
                if claim_entries is not None:
                    claim = self._claim_from_entries(claim_entries)
                    self._track_task(asyncio.create_task(self._process_claim(claim)))
                    await asyncio.sleep(0)
                    continue
                await asyncio.sleep(self.scheduler_poll_seconds)
        except asyncio.CancelledError:
            raise

    async def _process_claim(self, claim: InboxClaim) -> None:
        claim = self._claim_with_media_group_siblings(claim)
        incoming = self._message_from_claim(claim)
        try:
            await self._handle_incoming_message(incoming, inbox_update_ids=claim.update_ids)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception(
                "Failed to handle claimed inbox update(s) (%s)",
                ",".join(str(update_id) for update_id in claim.update_ids),
            )
            current_entries = [
                entry
                for update_id in claim.update_ids
                if (entry := self.storage.inbox.get_update(update_id=update_id)) is not None
            ]
            if any(entry.realized_assistant_message_id is not None and entry.state != "completed" for entry in current_entries):
                return

            complete_update_ids: list[int] = []
            reset_update_ids: list[int] = []
            for entry in current_entries:
                if entry.state == "completed":
                    continue
                if entry.reply_sent_at is not None:
                    complete_update_ids.append(entry.update_id)
                    continue
                if entry.reply_started_at is not None:
                    continue
                if self.storage.conversations.get_message_by_telegram(
                    chat_id=entry.chat_id,
                    telegram_message_id=entry.message_id,
                ) is not None:
                    complete_update_ids.append(entry.update_id)
                    continue
                if self.storage.conversations.find_pending_message_by_telegram(
                    chat_id=entry.chat_id,
                    telegram_message_id=entry.message_id,
                ) is not None:
                    complete_update_ids.append(entry.update_id)
                    continue
                reset_update_ids.append(entry.update_id)

            self.storage.inbox.complete_updates(update_ids=tuple(sorted(set(complete_update_ids))))
            self.storage.inbox.reset_updates_to_queued(update_ids=tuple(sorted(set(reset_update_ids))))
        else:
            self.storage.inbox.complete_updates(update_ids=claim.update_ids)

    def _claim_with_media_group_siblings(self, claim: InboxClaim) -> InboxClaim:
        if claim.media_group_key is None:
            return claim
        sibling_entries = self.storage.inbox.claim_queued_media_group_siblings(
            media_group_key=claim.media_group_key,
        )
        if not sibling_entries:
            return claim
        messages_by_update_id = {update_id: message for update_id, message in zip(claim.update_ids, claim.messages)}
        for entry in sibling_entries:
            messages_by_update_id[entry.update_id] = entry.message
        ordered_update_ids = tuple(sorted(messages_by_update_id))
        return InboxClaim(
            update_ids=ordered_update_ids,
            messages=tuple(messages_by_update_id[update_id] for update_id in ordered_update_ids),
            media_group_key=claim.media_group_key,
        )

    async def _recover_claimed_updates(self) -> None:
        claimed_updates = self.storage.inbox.list_updates(state="claimed")

        reset_update_ids: list[int] = []
        complete_update_ids: list[int] = []
        assistant_updates: dict[int, list[InboxUpdate]] = {}

        for entry in claimed_updates:
            if entry.realized_assistant_message_id is not None:
                assistant_updates.setdefault(entry.realized_assistant_message_id, []).append(entry)
                continue
            if entry.reply_sent_at is not None:
                complete_update_ids.append(entry.update_id)
                continue
            if entry.reply_started_at is not None:
                continue
            if self.storage.conversations.get_message_by_telegram(
                chat_id=entry.chat_id,
                telegram_message_id=entry.message_id,
            ) is not None:
                complete_update_ids.append(entry.update_id)
                continue
            if self.storage.conversations.find_pending_message_by_telegram(
                chat_id=entry.chat_id,
                telegram_message_id=entry.message_id,
            ) is not None:
                complete_update_ids.append(entry.update_id)
                continue
            reset_update_ids.append(entry.update_id)

        for assistant_message_id, entries in assistant_updates.items():
            assistant_message = self.storage.conversations.get_message(assistant_message_id)
            if assistant_message is None:
                reset_update_ids.extend(entry.update_id for entry in entries)
                continue
            render_entry = entries[0]
            render_state = self.storage.inbox.get_assistant_render_state(
                assistant_message_id=assistant_message_id,
            )
            recovered = await self.service.recover_interrupted_assistant_turn(
                api=self.api,
                assistant_message_id=assistant_message_id,
                reply_to_message_id=render_entry.message_id,
                final_render_phase=(
                    render_state.phase if render_state is not None else render_entry.assistant_render_phase
                ),
                final_render_status=(
                    render_state.final_status
                    if render_state is not None
                    else render_entry.assistant_render_final_status
                ),
                final_render_reply_text=(
                    render_state.reply_text if render_state is not None else render_entry.assistant_render_reply_text
                ),
                final_render_reasoning_blocks=(
                    render_state.reasoning_blocks
                    if render_state is not None
                    else render_entry.assistant_render_reasoning_blocks
                ),
                final_render_markdown=(
                    render_state.render_markdown if render_state is not None else render_entry.assistant_render_markdown
                ),
            )
            if recovered:
                complete_update_ids.extend(entry.update_id for entry in entries)

        self.storage.inbox.reset_updates_to_queued(update_ids=tuple(sorted(set(reset_update_ids))))
        self.storage.inbox.complete_updates(update_ids=tuple(sorted(set(complete_update_ids))))
        await self._recover_orphaned_streaming_assistant_turns(
            skip_assistant_message_ids=set(assistant_updates),
        )

    async def _recover_orphaned_streaming_assistant_turns(self, *, skip_assistant_message_ids: set[int]) -> None:
        for assistant_message in self.storage.conversations.list_streaming_assistant_messages():
            if assistant_message.id in skip_assistant_message_ids:
                continue
            render_state = self.storage.inbox.get_assistant_render_state(
                assistant_message_id=assistant_message.id,
            )
            await self.service.recover_interrupted_assistant_turn(
                api=self.api,
                assistant_message_id=assistant_message.id,
                reply_to_message_id=self._assistant_recovery_reply_to_message_id(assistant_message),
                final_render_phase=render_state.phase if render_state is not None else None,
                final_render_status=render_state.final_status if render_state is not None else None,
                final_render_reply_text=render_state.reply_text if render_state is not None else None,
                final_render_reasoning_blocks=render_state.reasoning_blocks if render_state is not None else (),
                final_render_markdown=render_state.render_markdown if render_state is not None else None,
            )

    def _assistant_recovery_reply_to_message_id(self, assistant_message: StoredMessage) -> int | None:
        if assistant_message.parent_message_id is None:
            return None
        parent_message = self.storage.conversations.get_message(assistant_message.parent_message_id)
        if parent_message is None:
            return None
        if parent_message.telegram_message_id is not None:
            return parent_message.telegram_message_id
        linked_ids = self.storage.conversations.list_linked_telegram_message_ids(
            logical_message_id=parent_message.id,
        )
        return linked_ids[0] if linked_ids else None

    def _message_from_claim(self, claim: InboxClaim) -> IncomingMessage:
        if len(claim.messages) == 1:
            return claim.messages[0]
        return self._merge_media_group_messages(list(claim.messages))

    def _claim_from_entries(self, entries: list[InboxUpdate]) -> InboxClaim:
        media_group_keys = {entry.media_group_key for entry in entries if entry.media_group_key is not None}
        media_group_key = media_group_keys.pop() if len(media_group_keys) == 1 else None
        return InboxClaim(
            update_ids=tuple(entry.update_id for entry in entries),
            messages=tuple(entry.message for entry in entries),
            media_group_key=media_group_key,
        )

    def _media_group_boundary_grace_seconds(self) -> float:
        if self.media_group_boundary_grace_seconds is not None:
            return self.media_group_boundary_grace_seconds
        return self.scheduler_poll_seconds

    def _parse_update_for_inbox(self, update: dict) -> tuple[int | None, IncomingMessage | None]:
        update_id = self._extract_update_id(update)
        try:
            return update_id, self._parse_incoming_message(update)
        except Exception:
            logging.exception("Failed to parse update (%s)", self._summarize_update(update))
            return update_id, None

    @staticmethod
    def _extract_update_id(update: dict) -> int | None:
        try:
            return int(update["update_id"])
        except (KeyError, TypeError, ValueError):
            return None

    async def _enqueue_shutdown_updates(self, *, offset: int | None) -> None:
        if offset is None:
            return
        try:
            updates = await asyncio.wait_for(
                self.api.get_updates(offset=offset, timeout=0),
                timeout=self.shutdown_poll_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logging.exception("Timed out polling final Telegram updates during shutdown")
            return
        except Exception:
            logging.exception("Failed to poll final Telegram updates during shutdown")
            return

        inbox_messages: list[IncomingMessage] = []
        for update in updates:
            _update_id, incoming = self._parse_update_for_inbox(update)
            if incoming is not None:
                inbox_messages.append(incoming)
        self.storage.inbox.enqueue_messages(messages=inbox_messages)

    async def _safe_handle_update(self, update: dict) -> None:
        try:
            await self._handle_update(update)
        except Exception:
            logging.exception("Failed to handle update (%s)", self._summarize_update(update))

    @staticmethod
    def _summarize_update(update: dict) -> str:
        update_id = update.get("update_id")
        top_level_kinds = ",".join(sorted(key for key in update.keys() if key != "update_id")) or "none"
        message = update.get("message")
        if not isinstance(message, dict):
            return f"update_id={update_id!r} kinds={top_level_kinds}"

        message_features: list[str] = []
        for key in ("text", "caption", "photo", "document", "video", "media_group_id", "reply_to_message"):
            if key in message:
                message_features.append(key)
        features = ",".join(message_features) or "none"
        return f"update_id={update_id!r} kinds={top_level_kinds} message_features={features}"

    async def _handle_update(self, update: dict) -> None:
        incoming = self._parse_incoming_message(update)
        if incoming is None:
            return
        await self._handle_incoming_message(incoming)

    async def _handle_incoming_message(
        self,
        incoming: IncomingMessage,
        *,
        inbox_update_ids: tuple[int, ...] = (),
    ) -> None:
        settings = self.storage.settings.get_chat_settings(incoming.chat_id)
        if settings.skip_prefix and incoming.text.startswith(settings.skip_prefix):
            return

        action = self.router.route(incoming)

        if isinstance(action, IgnoreAction):
            return

        if isinstance(action, CommandAction):
            await self._handle_command(
                incoming,
                settings,
                action,
                inbox_update_ids=inbox_update_ids,
            )
            return

        assert isinstance(action, ChatAction)
        if not settings.enabled:
            return
        if action.intent == "plain":
            if settings.reply_mode == "off":
                return
            if settings.reply_mode == "mention" and not self._is_mention_mode_plain_reply_allowed(incoming):
                return
        if not self.service.is_reply_allowed(chat_id=incoming.chat_id, user_id=incoming.user_id):
            return

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming,
            settings=settings,
            action=action,
            inbox_update_ids=inbox_update_ids,
        )

    def _is_mention_mode_plain_reply_allowed(self, incoming: IncomingMessage) -> bool:
        if incoming.mentions_bot or incoming.reply_to_bot:
            return True
        if incoming.reply_to_message_id is None:
            return False
        message = self.storage.conversations.get_message_by_telegram(
            chat_id=incoming.chat_id,
            telegram_message_id=incoming.reply_to_message_id,
        )
        if message is not None:
            conversation = self.storage.conversations.get_conversation(message.conversation_id)
            return conversation is not None and conversation.user_id == incoming.user_id
        pending_message = self.storage.conversations.find_pending_message_by_telegram(
            chat_id=incoming.chat_id,
            telegram_message_id=incoming.reply_to_message_id,
        )
        if pending_message is None:
            return False
        conversation = self.storage.conversations.get_conversation(pending_message.conversation_id)
        return conversation is not None and conversation.user_id == incoming.user_id

    async def _send_reply_only_response(
        self,
        *,
        message: IncomingMessage,
        text: str,
        inbox_update_ids: tuple[int, ...] = (),
    ) -> None:
        if inbox_update_ids:
            self.storage.inbox.mark_reply_started(update_ids=inbox_update_ids)
        await self.api.send_message(
            message.chat_id,
            text,
            reply_to_message_id=message.message_id,
        )
        if inbox_update_ids:
            self.storage.inbox.mark_reply_sent(update_ids=inbox_update_ids)

    async def _handle_command(
        self,
        message: IncomingMessage,
        settings,
        action: CommandAction,
        *,
        inbox_update_ids: tuple[int, ...] = (),
    ) -> None:
        if action.name == "usage_error":
            await self._send_reply_only_response(
                message=message,
                text=action.content or "Invalid command",
                inbox_update_ids=inbox_update_ids,
            )
            return

        if action.name == "ping":
            await self._send_reply_only_response(
                message=message,
                text=self.service.ping_text(message, settings),
                inbox_update_ids=inbox_update_ids,
            )
            return

        if action.name == "help":
            if action.argument:
                help_text = self.service.command_help_text(topic=action.argument, settings=settings)
                if help_text is None:
                    await self._send_reply_only_response(
                        message=message,
                        text=f"Unknown help topic: {action.argument}\nTry /help, /help new, /help c, or /help s.",
                        inbox_update_ids=inbox_update_ids,
                    )
                    return
                await self._send_reply_only_response(
                    message=message,
                    text=help_text,
                    inbox_update_ids=inbox_update_ids,
                )
                return
            await self._send_reply_only_response(
                message=message,
                text=self.service.help_text(settings=settings),
                inbox_update_ids=inbox_update_ids,
            )
            return

        if action.name == "models":
            await self._send_reply_only_response(
                message=message,
                text=self.service.list_models_text(settings),
                inbox_update_ids=inbox_update_ids,
            )
            return

        if action.name in {"togglechat", "toggleuser", "whitelist"}:
            if not self.service.has_configured_owners():
                await self._send_reply_only_response(
                    message=message,
                    text="Configure BOT_OWNER_USER_IDS before using allowlist commands.",
                    inbox_update_ids=inbox_update_ids,
                )
                return
            if not self.service.can_manage_allowlist(message.user_id):
                await self._send_reply_only_response(
                    message=message,
                    text="Only configured owners can change the allowlist.",
                    inbox_update_ids=inbox_update_ids,
                )
                return

        if action.name in {"model", "mode"} and not self.service.can_manage_chat(message.user_id):
            await self._send_reply_only_response(
                message=message,
                text="Only configured owners can change chat settings.",
                inbox_update_ids=inbox_update_ids,
            )
            return

        if action.name == "model":
            completed_in_transaction = False
            try:
                with self.storage.transaction():
                    reply_text = self.service.set_default_model(
                        chat_id=message.chat_id,
                        alias=action.argument or "",
                        commit=False,
                    )
                    if inbox_update_ids:
                        self.storage.inbox.complete_updates(update_ids=inbox_update_ids, commit=False)
                    completed_in_transaction = True
            except ValueError as exc:
                reply_text = str(exc)
            if completed_in_transaction:
                await self.api.send_message(message.chat_id, reply_text, reply_to_message_id=message.message_id)
            else:
                await self._send_reply_only_response(
                    message=message,
                    text=reply_text,
                    inbox_update_ids=inbox_update_ids,
                )
            return

        if action.name == "mode":
            completed_in_transaction = False
            try:
                with self.storage.transaction():
                    reply_text = self.service.set_reply_mode(
                        chat_id=message.chat_id,
                        reply_mode=(action.argument or ""),
                        commit=False,
                    )
                    if inbox_update_ids:
                        self.storage.inbox.complete_updates(update_ids=inbox_update_ids, commit=False)
                    completed_in_transaction = True
            except ValueError as exc:
                reply_text = str(exc)
            if completed_in_transaction:
                await self.api.send_message(message.chat_id, reply_text, reply_to_message_id=message.message_id)
            else:
                await self._send_reply_only_response(
                    message=message,
                    text=reply_text,
                    inbox_update_ids=inbox_update_ids,
                )
            return

        if action.name == "togglechat":
            completed_in_transaction = False
            try:
                target_chat_id = int(action.argument) if action.argument else message.chat_id
                with self.storage.transaction():
                    reply_text = self.service.toggle_chat_allowlist(chat_id=target_chat_id, commit=False)
                    if inbox_update_ids:
                        self.storage.inbox.complete_updates(update_ids=inbox_update_ids, commit=False)
                    completed_in_transaction = True
            except ValueError:
                reply_text = "Usage: /togglechat [chat_id]"
            if completed_in_transaction:
                await self.api.send_message(message.chat_id, reply_text, reply_to_message_id=message.message_id)
            else:
                await self._send_reply_only_response(
                    message=message,
                    text=reply_text,
                    inbox_update_ids=inbox_update_ids,
                )
            return

        if action.name == "toggleuser":
            completed_in_transaction = False
            try:
                if action.argument:
                    target_user_id = int(action.argument)
                elif message.reply_to_user_id is not None:
                    target_user_id = message.reply_to_user_id
                else:
                    target_user_id = message.user_id
                with self.storage.transaction():
                    reply_text = self.service.toggle_user_allowlist(user_id=target_user_id, commit=False)
                    if inbox_update_ids:
                        self.storage.inbox.complete_updates(update_ids=inbox_update_ids, commit=False)
                    completed_in_transaction = True
            except ValueError:
                reply_text = "Usage: /toggleuser [user_id]"
            if completed_in_transaction:
                await self.api.send_message(message.chat_id, reply_text, reply_to_message_id=message.message_id)
            else:
                await self._send_reply_only_response(
                    message=message,
                    text=reply_text,
                    inbox_update_ids=inbox_update_ids,
                )
            return

        if action.name == "whitelist":
            await self._send_reply_only_response(
                message=message,
                text=self.service.whitelist_text(),
                inbox_update_ids=inbox_update_ids,
            )
            return

        await self._send_reply_only_response(
            message=message,
            text=f"Unknown command handler: {action.name}",
            inbox_update_ids=inbox_update_ids,
        )

    def _parse_incoming_message(self, update: dict) -> IncomingMessage | None:
        message = update.get("message")
        if not isinstance(message, dict):
            return None

        text = message.get("text")
        caption = message.get("caption")
        from_user = message.get("from")
        chat = message.get("chat")
        if not isinstance(from_user, dict) or not isinstance(chat, dict):
            return None

        from_bot = bool(from_user.get("is_bot"))
        if from_bot:
            return None

        images = self._extract_images(message)
        if not isinstance(text, str) and isinstance(caption, str) and not images and self._has_unsupported_captioned_media(message):
            return None

        message_text = text if isinstance(text, str) else caption if isinstance(caption, str) else ""
        message_parts = build_content_parts(message_text, images)
        if not message_text and not images:
            return None

        reply_to = message.get("reply_to_message") or {}
        reply_to_from = reply_to.get("from") or {}
        reply_to_text = reply_to.get("text")
        if not isinstance(reply_to_text, str):
            reply_to_text = reply_to.get("caption")

        bot_username = self.bot_username or ""
        mention_pattern = rf"(?<!\w)@{re.escape(bot_username)}(?!\w)" if bot_username else r"$^"
        mentions_bot = bool(re.search(mention_pattern, message_text, flags=re.IGNORECASE))

        return IncomingMessage(
            update_id=int(update["update_id"]),
            chat_id=int(chat["id"]),
            message_id=int(message["message_id"]),
            user_id=int(from_user["id"]),
            chat_type=str(chat.get("type") or ""),
            text=message_text,
            from_bot=from_bot,
            mentions_bot=mentions_bot,
            source_message_ids=(int(message["message_id"]),),
            reply_to_message_id=int(reply_to["message_id"]) if "message_id" in reply_to else None,
            reply_to_user_id=int(reply_to_from["id"]) if "id" in reply_to_from else None,
            reply_to_bot=bool(reply_to_from.get("is_bot")) and int(reply_to_from.get("id", 0)) == self.bot_id,
            reply_to_text=reply_to_text if isinstance(reply_to_text, str) else None,
            images=images,
            parts=message_parts,
            media_group_id=str(message["media_group_id"]) if "media_group_id" in message else None,
        )

    def _extract_images(self, message: dict) -> tuple[ImageRef, ...]:
        images: list[ImageRef] = []

        photo_sizes = message.get("photo")
        if isinstance(photo_sizes, list) and photo_sizes:
            largest_photo = photo_sizes[-1]
            if isinstance(largest_photo, dict):
                file_id = largest_photo.get("file_id")
                if isinstance(file_id, str) and file_id:
                    images.append(
                        ImageRef(
                            file_id=file_id,
                            mime_type="image/jpeg",
                            file_size=largest_photo.get("file_size") if isinstance(largest_photo.get("file_size"), int) else None,
                        )
                    )

        document = message.get("document")
        if isinstance(document, dict):
            mime_type = document.get("mime_type")
            file_id = document.get("file_id")
            normalized_mime_type = self._normalize_image_document_mime_type(mime_type)
            if isinstance(file_id, str) and normalized_mime_type is not None:
                images.append(
                    ImageRef(
                        file_id=file_id,
                        mime_type=normalized_mime_type,
                        file_size=document.get("file_size") if isinstance(document.get("file_size"), int) else None,
                    )
                )

        return tuple(images)

    def _normalize_image_document_mime_type(self, mime_type: object) -> str | None:
        if not isinstance(mime_type, str):
            return None
        normalized = mime_type.casefold()
        if normalized == "image/jpg":
            normalized = "image/jpeg"
        if normalized not in SUPPORTED_IMAGE_DOCUMENT_MIME_TYPES:
            return None
        return normalized

    def _has_unsupported_captioned_media(self, message: dict) -> bool:
        if isinstance(message.get("document"), dict):
            return True
        for field_name in ("video", "animation", "audio", "voice", "video_note", "sticker"):
            if isinstance(message.get(field_name), dict):
                return True
        return False

    def _merge_media_group_messages(self, messages: list[IncomingMessage]) -> IncomingMessage:
        ordered = sorted(messages, key=lambda message: message.message_id)
        first = ordered[0]
        merged_texts = [message.text for message in ordered if message.text]
        merged_images = tuple(image for message in ordered for image in message.images)
        merged_parts = tuple(
            part
            for message in ordered
            for part in (message.parts or build_content_parts(message.text, message.images))
        )
        return IncomingMessage(
            update_id=first.update_id,
            chat_id=first.chat_id,
            message_id=ordered[-1].message_id,
            user_id=first.user_id,
            chat_type=first.chat_type,
            text="\n\n".join(merged_texts),
            from_bot=first.from_bot,
            mentions_bot=any(message.mentions_bot for message in ordered),
            source_message_ids=tuple(message.message_id for message in ordered),
            reply_to_message_id=first.reply_to_message_id,
            reply_to_user_id=first.reply_to_user_id,
            reply_to_bot=first.reply_to_bot,
            reply_to_text=first.reply_to_text,
            images=merged_images,
            parts=merged_parts,
            media_group_id=first.media_group_id,
        )
