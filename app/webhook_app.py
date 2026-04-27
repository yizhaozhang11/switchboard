from __future__ import annotations

import logging
import re

from app.chat_service import ChatService
from app.d1_storage import D1Storage
from app.router import Router
from app.telegram_api import TelegramBotAPI
from app.types import (
    ChatAction,
    CommandAction,
    IgnoreAction,
    ImageRef,
    IncomingMessage,
    build_content_parts,
)

SUPPORTED_IMAGE_DOCUMENT_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class WebhookHandler:
    """Handles a single Telegram update via webhook, routing to ChatService."""

    def __init__(
        self,
        *,
        service: ChatService,
        storage: D1Storage,
        token: str,
    ) -> None:
        self.service = service
        self.storage = storage
        self.api = TelegramBotAPI(token)
        self.bot_id: int | None = None
        self.bot_username: str | None = None
        self.router: Router | None = None

    async def _ensure_bot_info(self) -> None:
        if self.bot_id is not None:
            return
        me = await self.api.get_me()
        self.bot_id = int(me["id"])
        self.bot_username = str(me.get("username") or "")
        self.router = Router(bot_username=self.bot_username)
        logging.info("Bot connected as @%s (%s)", self.bot_username, self.bot_id)

    async def handle_update(self, update: dict) -> None:
        """Process a single Telegram update delivered via webhook."""
        await self._ensure_bot_info()
        try:
            incoming = self._parse_incoming_message(update)
        except Exception:
            logging.exception("Failed to parse update (%s)", self._summarize_update(update))
            return

        if incoming is None:
            return

        try:
            await self._handle_incoming_message(incoming)
        except Exception:
            logging.exception(
                "Failed to handle incoming message (chat=%s, msg=%s)",
                incoming.chat_id,
                incoming.message_id,
            )

    # ── Message parsing (copied from telegram_app.py) ──────────────────────

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

    @staticmethod
    def _normalize_image_document_mime_type(mime_type: object) -> str | None:
        if not isinstance(mime_type, str):
            return None
        normalized = mime_type.casefold()
        if normalized == "image/jpg":
            normalized = "image/jpeg"
        if normalized not in SUPPORTED_IMAGE_DOCUMENT_MIME_TYPES:
            return None
        return normalized

    @staticmethod
    def _has_unsupported_captioned_media(message: dict) -> bool:
        if isinstance(message.get("document"), dict):
            return True
        for field_name in ("video", "animation", "audio", "voice", "video_note", "sticker"):
            if isinstance(message.get(field_name), dict):
                return True
        return False

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

    # ── Incoming message handling (simplified — no inbox, no transactions) ──

    async def _handle_incoming_message(self, incoming: IncomingMessage) -> None:
        settings = await self.storage.settings.get_chat_settings(incoming.chat_id)
        if settings.skip_prefix and incoming.text.startswith(settings.skip_prefix):
            return

        action = self.router.route(incoming)  # type: ignore[union-attr]

        if isinstance(action, IgnoreAction):
            return

        if isinstance(action, CommandAction):
            await self._handle_command(incoming, settings, action)
            return

        assert isinstance(action, ChatAction)
        if not settings.enabled:
            return
        if action.intent == "plain":
            if settings.reply_mode == "off":
                return
            if settings.reply_mode == "mention" and not await self._is_mention_mode_plain_reply_allowed(incoming):
                return
        if not await self.service.is_reply_allowed(chat_id=incoming.chat_id, user_id=incoming.user_id):
            return

        await self.service.generate_reply(
            api=self.api,
            incoming_message=incoming,
            settings=settings,
            action=action,
        )

    async def _is_mention_mode_plain_reply_allowed(self, incoming: IncomingMessage) -> bool:
        if incoming.mentions_bot or incoming.reply_to_bot:
            return True
        if incoming.reply_to_message_id is None:
            return False
        message = await self.storage.conversations.get_message_by_telegram(
            chat_id=incoming.chat_id,
            telegram_message_id=incoming.reply_to_message_id,
        )
        if message is not None:
            conversation = await self.storage.conversations.get_conversation(message.conversation_id)
            return conversation is not None and conversation.user_id == incoming.user_id
        pending_message = await self.storage.conversations.find_pending_message_by_telegram(
            chat_id=incoming.chat_id,
            telegram_message_id=incoming.reply_to_message_id,
        )
        if pending_message is None:
            return False
        conversation = await self.storage.conversations.get_conversation(pending_message.conversation_id)
        return conversation is not None and conversation.user_id == incoming.user_id

    # ── Command handling (simplified — no inbox_update_ids, no transactions) ──

    async def _send_reply_only_response(self, *, message: IncomingMessage, text: str) -> None:
        await self.api.send_message(
            message.chat_id,
            text,
            reply_to_message_id=message.message_id,
        )

    async def _handle_command(
        self,
        message: IncomingMessage,
        settings,
        action: CommandAction,
    ) -> None:
        if action.name == "usage_error":
            await self._send_reply_only_response(
                message=message,
                text=action.content or "Invalid command",
            )
            return

        if action.name == "ping":
            await self._send_reply_only_response(
                message=message,
                text=self.service.ping_text(message, settings),
            )
            return

        if action.name == "help":
            if action.argument:
                help_text = self.service.command_help_text(topic=action.argument, settings=settings)
                if help_text is None:
                    await self._send_reply_only_response(
                        message=message,
                        text=f"Unknown help topic: {action.argument}\nTry /help, /help new, /help c, or /help s.",
                    )
                    return
                await self._send_reply_only_response(
                    message=message,
                    text=help_text,
                )
                return
            await self._send_reply_only_response(
                message=message,
                text=self.service.help_text(settings=settings),
            )
            return

        if action.name == "models":
            await self._send_reply_only_response(
                message=message,
                text=self.service.list_models_text(settings),
            )
            return

        if action.name in {"togglechat", "toggleuser", "whitelist"}:
            if not self.service.has_configured_owners():
                await self._send_reply_only_response(
                    message=message,
                    text="Configure BOT_OWNER_USER_IDS before using allowlist commands.",
                )
                return
            if not self.service.can_manage_allowlist(message.user_id):
                await self._send_reply_only_response(
                    message=message,
                    text="Only configured owners can change the allowlist.",
                )
                return

        if action.name in {"model", "mode"} and not self.service.can_manage_chat(message.user_id):
            await self._send_reply_only_response(
                message=message,
                text="Only configured owners can change chat settings.",
            )
            return

        if action.name == "model":
            try:
                reply_text = await self.service.set_default_model(
                    chat_id=message.chat_id,
                    alias=action.argument or "",
                )
            except ValueError as exc:
                reply_text = str(exc)
            await self._send_reply_only_response(message=message, text=reply_text)
            return

        if action.name == "mode":
            try:
                reply_text = await self.service.set_reply_mode(
                    chat_id=message.chat_id,
                    reply_mode=(action.argument or ""),
                )
            except ValueError as exc:
                reply_text = str(exc)
            await self._send_reply_only_response(message=message, text=reply_text)
            return

        if action.name == "togglechat":
            try:
                target_chat_id = int(action.argument) if action.argument else message.chat_id
                reply_text = await self.service.toggle_chat_allowlist(chat_id=target_chat_id)
            except ValueError:
                reply_text = "Usage: /togglechat [chat_id]"
            await self._send_reply_only_response(message=message, text=reply_text)
            return

        if action.name == "toggleuser":
            try:
                if action.argument:
                    target_user_id = int(action.argument)
                elif message.reply_to_user_id is not None:
                    target_user_id = message.reply_to_user_id
                else:
                    target_user_id = message.user_id
                reply_text = await self.service.toggle_user_allowlist(user_id=target_user_id)
            except ValueError:
                reply_text = "Usage: /toggleuser [user_id]"
            await self._send_reply_only_response(message=message, text=reply_text)
            return

        if action.name == "whitelist":
            await self._send_reply_only_response(
                message=message,
                text=await self.service.whitelist_text(),
            )
            return

        await self._send_reply_only_response(
            message=message,
            text=f"Unknown command handler: {action.name}",
        )
