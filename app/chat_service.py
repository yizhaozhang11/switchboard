from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, replace

from app.config import VALID_REPLY_MODES
from app.conversation_engine import ConversationEngine, DeferredAction, StoredUserInput, TurnPlan
from app.d1_storage import D1Storage, utcnow
from app.providers.registry import ProviderRegistry, supported_tool_aliases_for_provider
from app.render import ReplySession, render_reply_text
from app.telegram_api import TelegramBotAPI
from app.types import (
    ChatAction,
    ChatRequest,
    ChatSettings,
    ContentPart,
    ConversationMessage,
    ImageRef,
    IncomingMessage,
    PendingMessage,
    build_content_parts,
)


@dataclass(frozen=True)
class PreparedTurn:
    chat_id: int
    user_id: int
    conversation_id: int
    assistant_message_id: int
    user_message_id: int | None
    request: ChatRequest
    reply_to_message_id: int
    model_alias: str


@dataclass(frozen=True)
class PendingTurnCandidate:
    conversation_id: int
    pending_messages: tuple[PendingMessage, ...]


class ChatService:
    def __init__(
        self,
        *,
        storage: D1Storage,
        registry: ProviderRegistry,
        system_prompt: str,
        owner_user_ids: tuple[int, ...],
        conversation_timeout_seconds: int,
        render_limit: int,
        render_edit_interval_seconds: float,
        safety_identifier_salt: str | None = None,
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.system_prompt = system_prompt
        self.owner_user_ids = set(owner_user_ids)
        self.render_limit = render_limit
        self.render_edit_interval_seconds = render_edit_interval_seconds
        self.safety_identifier_salt = safety_identifier_salt
        self.engine = ConversationEngine(
            storage=storage,
            conversation_timeout_seconds=conversation_timeout_seconds,
        )

    # ── Command handlers (synchronous, no storage I/O) ──

    def list_models_text(self, settings: ChatSettings) -> str:
        lines = [f"Default model: {settings.default_model_alias}", "", "Available models:"]
        for resolved in self.registry.list_models():
            aliases = ", ".join(resolved.model.aliases)
            line = f"- {aliases} -> {resolved.model.provider}:{resolved.model.model_id}"
            if resolved.model.supports_tools:
                tool_aliases = supported_tool_aliases_for_provider(resolved.model.provider)
                if tool_aliases:
                    options = ", ".join(f"-{alias}" for alias in tool_aliases)
                    line += f" (options: {options})"
            lines.append(line)
        return "\n".join(lines)

    def help_text(self, *, settings: ChatSettings) -> str:
        return "\n".join(
            [
                "Commands:",
                "- /help - show this help",
                "- /help <new|c|s> - explain how that command uses conversation state",
                "- /ping - show chat and user IDs, reply mode, and default model",
                "- /models - list available model aliases",
                "- /c <alias> <content> - send one message with a specific model",
                "- /new <content> - start a fresh conversation",
                "- /s <prompt> - set a system prompt override for a conversation",
                "",
                "Chat settings:",
                "- /model <alias> - set the default model for this chat",
                "- /mode auto|mention|off - set how plain messages are handled",
                "",
                "Owner commands:",
                "- /togglechat [chat_id] - add or remove a chat from the whitelist",
                "- /toggleuser [user_id] - add or remove a user from the whitelist",
                "- /whitelist - show the current whitelist",
                "",
                f"Current defaults: model={settings.default_model_alias}, reply_mode={settings.reply_mode}",
            ]
        )

    def command_help_text(self, *, topic: str, settings: ChatSettings) -> str | None:
        normalized_topic = topic.strip().casefold().lstrip("/")
        if normalized_topic == "new":
            return "\n".join(
                [
                    "/new <content>",
                    f"- Starts a fresh conversation with the chat default model alias {settings.default_model_alias}.",
                    "- If you reply to a seed state whose visible history is empty, it materializes that seeded branch instead of starting from scratch.",
                    "- It does not continue newer sibling branches from elsewhere in the conversation tree.",
                ]
            )
        if normalized_topic == "c":
            return "\n".join(
                [
                    "/c <alias> <content>",
                    "- Sends one message with the model alias you choose.",
                    "- If you reply to an attachable non-user state, it continues or forks from that exact point with the requested model.",
                    "- Otherwise it starts a fresh conversation with that alias.",
                ]
            )
        if normalized_topic == "s":
            return "\n".join(
                [
                    "/s <prompt>",
                    "- Creates a seed state with a system-prompt override.",
                    "- Without a reply target, it stores that seed and waits for a later user message.",
                    "- If you reply to an attachable state, it branches from that exact point with the new system prompt.",
                    "- If the replied-to branch currently ends on a user turn, the bot immediately continues from that seeded branch.",
                ]
            )
        return None

    def ping_text(self, message: IncomingMessage, settings: ChatSettings) -> str:
        return (
            f"chat_id={message.chat_id}\n"
            f"user_id={message.user_id}\n"
            f"reply_mode={settings.reply_mode}\n"
            f"default_model={settings.default_model_alias}"
        )

    def can_manage_chat(self, user_id: int) -> bool:
        return not self.owner_user_ids or user_id in self.owner_user_ids

    def has_configured_owners(self) -> bool:
        return bool(self.owner_user_ids)

    def can_manage_allowlist(self, user_id: int) -> bool:
        return user_id in self.owner_user_ids

    async def set_default_model(self, *, chat_id: int, alias: str) -> str:
        selection = self.registry.resolve_selection(alias)
        if selection is None:
            raise ValueError(f"Unknown model alias: {alias}")
        await self.storage.settings.set_default_model_alias(chat_id, alias)
        return f"Default model set to {alias} -> {selection.model.provider}:{selection.model.model_id}"

    async def set_reply_mode(self, *, chat_id: int, reply_mode: str) -> str:
        if reply_mode not in VALID_REPLY_MODES:
            raise ValueError("Reply mode must be one of auto, mention, off")
        await self.storage.settings.set_reply_mode(chat_id, reply_mode)
        return f"Reply mode set to {reply_mode}"

    async def is_reply_allowed(self, *, chat_id: int, user_id: int) -> bool:
        return await self.storage.settings.is_reply_allowed(chat_id=chat_id, user_id=user_id)

    async def toggle_chat_allowlist(self, *, chat_id: int) -> str:
        is_allowed = await self.storage.settings.toggle_allowlist_entry(kind="chat", target_id=chat_id)
        state = "added to" if is_allowed else "removed from"
        return f"Chat {chat_id} {state} the whitelist."

    async def toggle_user_allowlist(self, *, user_id: int) -> str:
        is_allowed = await self.storage.settings.toggle_allowlist_entry(kind="user", target_id=user_id)
        state = "added to" if is_allowed else "removed from"
        return f"User {user_id} {state} the whitelist."

    async def whitelist_text(self) -> str:
        entries = await self.storage.settings.list_allowlist_entries()
        chat_ids = [str(entry.target_id) for entry in entries if entry.kind == "chat"]
        user_ids = [str(entry.target_id) for entry in entries if entry.kind == "user"]
        return "\n".join(
            [
                "Whitelisted chats:",
                *self._format_allowlist_lines(chat_ids),
                "",
                "Whitelisted users:",
                *self._format_allowlist_lines(user_ids),
            ]
        )

    def _format_allowlist_lines(self, entries: list[str]) -> list[str]:
        if not entries:
            return ["- (none)"]
        return [f"- {entry}" for entry in entries]

    # ── Core reply generation (simplified for webhook context) ──

    async def generate_reply(
        self,
        *,
        api: TelegramBotAPI,
        incoming_message: IncomingMessage,
        settings: ChatSettings,
        action: ChatAction,
    ) -> None:
        raw_user_input = (
            None
            if action.intent == "set_system_prompt"
            else self._build_user_input(
                content=action.content,
                images=action.images,
                parts=action.parts,
            )
        )

        result = await self.engine.begin_action(
            incoming_message=incoming_message,
            default_model_alias=settings.default_model_alias,
            action=action,
            user_input=raw_user_input,
        )

        if result is None:
            return

        if isinstance(result, DeferredAction):
            # In webhook context, deferring means we just queue
            # The next webhook call will pick up the pending messages
            return

        assert isinstance(result, TurnPlan)
        prepared_turn = await self._realize_turn(api=api, plan=result)
        if prepared_turn is None:
            return
        await self._run_conversation_loop(api=api, prepared_turn=prepared_turn)

    async def _realize_turn(
        self,
        *,
        api: TelegramBotAPI,
        plan: TurnPlan,
    ) -> PreparedTurn | None:
        conversation = plan.conversation
        selection = self.registry.resolve_selection(conversation.model_alias)
        if selection is None:
            await api.send_message(
                conversation.chat_id,
                f"Unknown model alias: {conversation.model_alias}",
                reply_to_message_id=plan.reply_to_message_id,
            )
            return None

        assistant_parent_message_id = plan.assistant_parent_message_id
        if plan.user_input is not None:
            if plan.user_input.images and not selection.model.supports_images:
                await api.send_message(
                    conversation.chat_id,
                    f"Model alias {conversation.model_alias} does not accept image input.",
                    reply_to_message_id=plan.reply_to_message_id,
                )
                return None

            # Download images from Telegram → loaded refs (no filesystem in Workers)
            loaded_parts = await self._ensure_loaded_images(api=api, parts=plan.user_input.parts)
            plan = TurnPlan(
                conversation=plan.conversation,
                reply_to_message_id=plan.reply_to_message_id,
                source_telegram_message_ids=plan.source_telegram_message_ids,
                user_input=StoredUserInput(
                    content=plan.user_input.content,
                    images=tuple(p.image for p in loaded_parts if p.kind == "image" and p.image is not None),
                    parts=loaded_parts,
                ),
                user_parent_message_id=plan.user_parent_message_id,
                assistant_parent_message_id=plan.assistant_parent_message_id,
                user_message_created_at=plan.user_message_created_at,
            )

        user_message_id: int | None = None

        # D1 auto-commits each call — no transaction wrapper needed
        if plan.user_input is not None:
            timestamp = plan.user_message_created_at or utcnow()
            user_message_id = await self.storage.conversations.create_message(
                conversation_id=conversation.id,
                chat_id=conversation.chat_id,
                telegram_message_id=plan.reply_to_message_id,
                message_type="user",
                parent_message_id=plan.user_parent_message_id,
                provider=selection.model.provider,
                model_id=selection.model.model_id,
                model_alias=conversation.model_alias,
                content=plan.user_input.content,
                images=plan.user_input.images,
                parts=plan.user_input.parts,
                status="complete",
                created_at=timestamp,
            )
            await self._link_logical_message_telegram_ids(
                chat_id=conversation.chat_id,
                logical_message_id=user_message_id,
                primary_telegram_message_id=plan.reply_to_message_id,
                telegram_message_ids=plan.source_telegram_message_ids,
            )
            assistant_parent_message_id = user_message_id

        if assistant_parent_message_id is None:
            raise RuntimeError("Turn plan does not have an assistant parent")

        conversation_messages = await self._build_provider_conversation(assistant_parent_message_id)
        request = ChatRequest(
            model=selection.model,
            conversation=conversation_messages,
            system_prompt=conversation.system_prompt_override or self.system_prompt,
            safety_identifier=self._build_safety_identifier(conversation.user_id),
            requested_tools=selection.requested_tools,
        )
        assistant_message_id = await self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=conversation.chat_id,
            telegram_message_id=None,
            message_type="assistant",
            parent_message_id=assistant_parent_message_id,
            provider=selection.model.provider,
            model_id=selection.model.model_id,
            model_alias=conversation.model_alias,
            content="",
            status="streaming",
        )

        return PreparedTurn(
            chat_id=conversation.chat_id,
            user_id=conversation.user_id,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message_id,
            user_message_id=user_message_id,
            request=request,
            reply_to_message_id=plan.reply_to_message_id,
            model_alias=conversation.model_alias,
        )

    async def _link_logical_message_telegram_ids(
        self,
        *,
        chat_id: int,
        logical_message_id: int,
        primary_telegram_message_id: int | None,
        telegram_message_ids: tuple[int, ...],
    ) -> None:
        seen_ids: set[int] = set()
        if primary_telegram_message_id is not None:
            seen_ids.add(primary_telegram_message_id)

        for telegram_message_id in telegram_message_ids:
            if telegram_message_id <= 0 or telegram_message_id in seen_ids:
                continue
            await self.storage.conversations.link_telegram_message(
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                logical_message_id=logical_message_id,
                part_index=0,
            )
            seen_ids.add(telegram_message_id)

    async def _ensure_loaded_images(
        self,
        *,
        api: TelegramBotAPI,
        parts: tuple[ContentPart, ...],
    ) -> tuple[ContentPart, ...]:
        """Download telegram images and store in D1 BlobStore (content-addressed)."""
        loaded_parts: list[ContentPart] = []
        for part in parts:
            if part.kind != "image" or part.image is None:
                loaded_parts.append(part)
                continue
            image = part.image
            if image.data is not None:
                loaded_parts.append(part)
                continue
            if image.file_id is None:
                loaded_parts.append(part)
                continue
            try:
                data = await api.download_file_bytes(image.file_id)
                stored = await self.storage.blobs.store_image(
                    mime_type=image.mime_type,
                    data=data,
                )
                loaded_parts.append(ContentPart(kind="image", image=stored))
            except Exception:
                logging.exception("Failed to download/store image file_id=%s", image.file_id)
                loaded_parts.append(part)
        return tuple(loaded_parts)

    async def _build_provider_conversation(self, message_id: int) -> list[ConversationMessage]:
        conversation_messages: list[ConversationMessage] = []
        thread = await self.storage.conversations.build_thread(message_id)
        for item in thread:
            if item.message_type == "seed":
                continue
            # Load stored images from D1 BlobStore → memory for provider
            loaded_parts: list[ContentPart] = []
            for part in item.parts:
                if part.kind == "image" and part.image is not None and part.image.sha256 is not None:
                    data = await self.storage.blobs.load_image_bytes(part.image)
                    loaded = ImageRef.loaded(mime_type=part.image.mime_type, data=data)
                    loaded_parts.append(ContentPart(kind="image", image=loaded))
                else:
                    loaded_parts.append(part)
            conversation_messages.append(
                ConversationMessage(
                    role=item.message_type,
                    content=item.content,
                    parts=tuple(loaded_parts),
                )
            )
        return conversation_messages

    def _build_safety_identifier(self, user_id: int) -> str | None:
        if self.safety_identifier_salt is None:
            return None
        return hashlib.sha256(f"{self.safety_identifier_salt}{user_id}".encode("utf-8")).hexdigest()

    # ── Conversation loop (streaming) ──

    async def _run_conversation_loop(self, *, api: TelegramBotAPI, prepared_turn: PreparedTurn) -> None:
        current_turn: PreparedTurn | None = prepared_turn
        while current_turn is not None:
            full_text = ""
            reasoning_blocks: list[str] = []
            final_status = "complete"
            linked_message_ids: set[int] = set()
            session = ReplySession(
                api,
                chat_id=current_turn.chat_id,
                reply_to_message_id=current_turn.reply_to_message_id,
                prefix=f"[{current_turn.model_alias}] ",
                limit=self.render_limit,
                edit_interval_seconds=self.render_edit_interval_seconds,
            )

            resolved = self.registry.resolve(current_turn.model_alias)

            async def update_render(*, force: bool, final: bool, status: str) -> None:
                persisted_status = "streaming" if final else status
                await self._persist_assistant_state(
                    prepared_turn=current_turn,
                    session=session,
                    full_text=full_text,
                    linked_message_ids=linked_message_ids,
                    status=persisted_status,
                )

                async def apply_render(*, render_markdown: bool | None) -> None:
                    used_markdown = final if render_markdown is None else render_markdown
                    if final:
                        await self._persist_assistant_render_state(
                            prepared_turn=current_turn,
                            phase="final_pending",
                            final_status=status,
                            reply_text=full_text,
                            reasoning_blocks=tuple(reasoning_blocks),
                            render_markdown=used_markdown,
                        )

                    async def on_sent_message_id(index: int, _message_id: int) -> None:
                        if not final:
                            return
                        await self._sync_streaming_telegram_links(
                            chat_id=current_turn.chat_id,
                            logical_message_id=current_turn.assistant_message_id,
                            telegram_message_ids=session.message_ids,
                            linked_message_ids=linked_message_ids,
                        )
                        await self._persist_assistant_render_state(
                            prepared_turn=current_turn,
                            phase="final_pending",
                            final_status=status,
                            reply_text=full_text,
                            reasoning_blocks=tuple(reasoning_blocks),
                            render_markdown=used_markdown,
                        )

                    await session.update(
                        render_reply_text(
                            full_text,
                            reasoning_blocks,
                            final=final,
                            render_markdown=render_markdown,
                        ),
                        force=force,
                        on_sent_message_id=on_sent_message_id if final else None,
                    )
                    await self._persist_assistant_state(
                        prepared_turn=current_turn,
                        session=session,
                        full_text=full_text,
                        linked_message_ids=linked_message_ids,
                        status=persisted_status,
                    )
                    if final:
                        await self._persist_assistant_render_state(
                            prepared_turn=current_turn,
                            phase="final_rendered",
                            final_status=status,
                            reply_text=full_text,
                            reasoning_blocks=tuple(reasoning_blocks),
                            render_markdown=used_markdown,
                        )

                try:
                    await apply_render(render_markdown=None)
                except Exception:
                    if not final:
                        raise
                    logging.exception(
                        "Formatted final render failed for conversation %s assistant message %s; retrying plain text",
                        current_turn.conversation_id,
                        current_turn.assistant_message_id,
                    )
                    await apply_render(render_markdown=False)

            try:
                if resolved is None:
                    full_text = f"[error] Unknown model alias: {current_turn.model_alias}"
                    final_status = "failed"
                else:
                    async for event in resolved.provider.stream_reply(current_turn.request):
                        if event.kind == "text_delta":
                            full_text += event.text
                            await update_render(force=False, final=False, status="streaming")
                        elif event.kind == "reasoning_delta":
                            if not reasoning_blocks:
                                reasoning_blocks.append("")
                            reasoning_blocks[-1] += event.text
                            await update_render(force=False, final=False, status="streaming")
                        elif event.kind == "reasoning_delimiter":
                            if reasoning_blocks and reasoning_blocks[-1].strip():
                                reasoning_blocks.append("")
                        elif event.kind == "error":
                            if full_text:
                                full_text += "\n\n"
                            full_text += f"[error] {event.text}"
                            final_status = "failed"
                            break
                        elif event.kind == "done" and not full_text and event.text:
                            full_text = event.text

                if not full_text.strip():
                    full_text = "(empty response)"

                await update_render(force=True, final=True, status=final_status)
            except Exception as exc:
                logging.exception(
                    "Reply generation failed for conversation %s assistant message %s",
                    current_turn.conversation_id,
                    current_turn.assistant_message_id,
                )
                final_status = "failed"
                if not full_text.strip():
                    full_text = f"[error] {type(exc).__name__}: {exc}"
                try:
                    await update_render(force=True, final=True, status=final_status)
                except Exception:
                    logging.exception(
                        "Failed to render final error reply for conversation %s assistant message %s",
                        current_turn.conversation_id,
                        current_turn.assistant_message_id,
                    )

            # After turn completes, check for pending messages
            next_turn_candidate = await self._finalize_turn(
                api=api,
                prepared_turn=current_turn,
                session=session,
                full_text=full_text,
                final_status=final_status,
            )
            if next_turn_candidate is None:
                current_turn = None
                continue
            current_turn = await self._prepare_turn_from_candidate(api=api, candidate=next_turn_candidate)

    async def _persist_assistant_state(
        self,
        *,
        prepared_turn: PreparedTurn,
        session: ReplySession,
        full_text: str,
        linked_message_ids: set[int],
        status: str,
    ) -> None:
        stored_text = full_text if full_text.strip() else ""
        await self.storage.conversations.update_message(
            prepared_turn.assistant_message_id,
            content=stored_text,
            status=status,
        )
        await self._sync_streaming_telegram_links(
            chat_id=prepared_turn.chat_id,
            logical_message_id=prepared_turn.assistant_message_id,
            telegram_message_ids=session.message_ids,
            linked_message_ids=linked_message_ids,
        )

    async def _persist_assistant_render_state(
        self,
        *,
        prepared_turn: PreparedTurn,
        phase: str | None,
        final_status: str | None,
        reply_text: str | None,
        reasoning_blocks: tuple[str, ...],
        render_markdown: bool | None,
    ) -> None:
        stored_reply_text = None
        if reply_text is not None:
            stored_reply_text = reply_text if reply_text.strip() else ""
        await self.storage.inbox.set_assistant_render_state(
            assistant_message_id=prepared_turn.assistant_message_id,
            phase=phase,
            final_status=final_status,
            reply_text=stored_reply_text,
            reasoning_blocks=reasoning_blocks,
            render_markdown=render_markdown,
        )

    async def _sync_streaming_telegram_links(
        self,
        *,
        chat_id: int,
        logical_message_id: int,
        telegram_message_ids: list[int],
        linked_message_ids: set[int],
    ) -> None:
        for index, telegram_message_id in enumerate(telegram_message_ids):
            if telegram_message_id in linked_message_ids:
                continue
            await self.storage.conversations.link_telegram_message(
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                logical_message_id=logical_message_id,
                part_index=index,
            )
            linked_message_ids.add(telegram_message_id)

    async def _finalize_turn(
        self,
        *,
        api: TelegramBotAPI,
        prepared_turn: PreparedTurn,
        session: ReplySession,
        full_text: str,
        final_status: str,
    ) -> PendingTurnCandidate | None:
        stored_text = full_text if full_text.strip() else "(empty response)"
        await self.storage.conversations.update_message(
            prepared_turn.assistant_message_id,
            content=stored_text,
            status=final_status,
        )
        await self.storage.inbox.set_assistant_render_state(
            assistant_message_id=prepared_turn.assistant_message_id,
            phase=None,
            final_status=None,
            reply_text=None,
            reasoning_blocks=(),
            render_markdown=None,
        )
        for index, telegram_message_id in enumerate(session.message_ids):
            await self.storage.conversations.link_telegram_message(
                chat_id=prepared_turn.chat_id,
                telegram_message_id=telegram_message_id,
                logical_message_id=prepared_turn.assistant_message_id,
                part_index=index,
            )

        # Check for pending messages
        pending_messages = await self.storage.conversations.list_pending_messages(
            conversation_id=prepared_turn.conversation_id,
        )
        if not pending_messages:
            return None
        return PendingTurnCandidate(
            conversation_id=prepared_turn.conversation_id,
            pending_messages=tuple(pending_messages),
        )

    async def _prepare_turn_from_candidate(
        self,
        *,
        api: TelegramBotAPI,
        candidate: PendingTurnCandidate,
    ) -> PreparedTurn | None:
        conversation = await self.storage.conversations.get_conversation(candidate.conversation_id)
        if conversation is None:
            return None

        pending_messages_list = list(candidate.pending_messages)
        if not pending_messages_list:
            return None

        await self.storage.conversations.delete_pending_messages(
            pending_message_ids=tuple(pm.id for pm in pending_messages_list),
        )

        turn_plan = await self.engine.build_pending_turn(
            conversation_id=candidate.conversation_id,
            pending_messages=pending_messages_list,
        )
        if turn_plan is None:
            return None
        return await self._realize_turn(api=api, plan=turn_plan)

    def _build_user_input(
        self,
        *,
        content: str,
        images: tuple[ImageRef, ...],
        parts: tuple[ContentPart, ...],
    ) -> StoredUserInput:
        effective_parts = parts or build_content_parts(content, images)
        normalized_parts: list[ContentPart] = []
        normalized_images: list[ImageRef] = []
        for part in effective_parts:
            if part.kind == "text":
                normalized_parts.append(ContentPart(kind="text", text=part.text))
                continue
            if part.image is None:
                continue
            normalized_parts.append(ContentPart(kind="image", image=part.image))
            normalized_images.append(part.image)
        normalized_image_tuple = tuple(normalized_images)
        return StoredUserInput(
            content=content,
            images=normalized_image_tuple,
            parts=tuple(normalized_parts) if normalized_parts else build_content_parts(content, normalized_image_tuple),
        )
