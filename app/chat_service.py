from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, replace

from app.config import VALID_REPLY_MODES
from app.conversation_engine import ConversationEngine, DeferredAction, StoredUserInput, TurnPlan
from app.providers.registry import ProviderRegistry, supported_tool_aliases_for_provider
from app.render import ReplySession, render_reply_text
from app.storage import Storage, utcnow
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
        storage: Storage,
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
        self._turn_claim_locks: defaultdict[tuple[int, int], asyncio.Lock] = defaultdict(asyncio.Lock)
        self._conversation_locks: defaultdict[tuple[int, int], asyncio.Lock] = defaultdict(asyncio.Lock)
        self._assistant_completion_events: dict[int, asyncio.Event] = {}

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

    def set_default_model(self, *, chat_id: int, alias: str, commit: bool = True) -> str:
        selection = self.registry.resolve_selection(alias)
        if selection is None:
            raise ValueError(f"Unknown model alias: {alias}")
        self.storage.settings.set_default_model_alias(chat_id, alias, commit=commit)
        return f"Default model set to {alias} -> {selection.model.provider}:{selection.model.model_id}"

    def set_reply_mode(self, *, chat_id: int, reply_mode: str, commit: bool = True) -> str:
        if reply_mode not in VALID_REPLY_MODES:
            raise ValueError("Reply mode must be one of auto, mention, off")
        self.storage.settings.set_reply_mode(chat_id, reply_mode, commit=commit)
        return f"Reply mode set to {reply_mode}"

    def is_reply_allowed(self, *, chat_id: int, user_id: int) -> bool:
        return self.storage.settings.is_reply_allowed(chat_id=chat_id, user_id=user_id)

    def toggle_chat_allowlist(self, *, chat_id: int, commit: bool = True) -> str:
        is_allowed = self.storage.settings.toggle_allowlist_entry(kind="chat", target_id=chat_id, commit=commit)
        state = "added to" if is_allowed else "removed from"
        return f"Chat {chat_id} {state} the whitelist."

    def toggle_user_allowlist(self, *, user_id: int, commit: bool = True) -> str:
        is_allowed = self.storage.settings.toggle_allowlist_entry(kind="user", target_id=user_id, commit=commit)
        state = "added to" if is_allowed else "removed from"
        return f"User {user_id} {state} the whitelist."

    def whitelist_text(self) -> str:
        entries = self.storage.settings.list_allowlist_entries()
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

    async def generate_reply(
        self,
        *,
        api: TelegramBotAPI,
        incoming_message: IncomingMessage,
        settings: ChatSettings,
        action: ChatAction,
        inbox_update_ids: tuple[int, ...] = (),
    ) -> None:
        prepared_turn: PreparedTurn | None = None
        raw_user_input = (
            None
            if action.intent == "set_system_prompt"
            else self._build_user_input(
                content=action.content,
                images=action.images,
                parts=action.parts,
            )
        )
        lock_key = (incoming_message.chat_id, incoming_message.user_id)
        while True:
            deferred_action: DeferredAction | None = None
            turn_plan: TurnPlan | None = None
            async with self._turn_claim_locks[lock_key]:
                async with self._conversation_locks[lock_key]:
                    result = self.engine.begin_action(
                        incoming_message=incoming_message,
                        default_model_alias=settings.default_model_alias,
                        action=action,
                        user_input=raw_user_input,
                    )
                    if isinstance(result, DeferredAction):
                        deferred_action = result
                    elif isinstance(result, TurnPlan):
                        turn_plan = result
                    else:
                        prepared_turn = None
                        break
                if turn_plan is not None:
                    stored_turn_plan = await self._try_store_turn_plan_images(api=api, plan=turn_plan)
                    if stored_turn_plan is None:
                        return
                    async with self._conversation_locks[lock_key]:
                        prepared_turn = await self._realize_turn_locked(
                            api=api,
                            plan=stored_turn_plan,
                            inbox_update_ids=inbox_update_ids,
                        )
                    break
            if deferred_action is not None:
                await self._wait_for_assistant_completion(deferred_action.assistant_message_id)

        if prepared_turn is None:
            return
        await self._run_conversation_loop(api=api, prepared_turn=prepared_turn)

    async def _wait_for_assistant_completion(self, assistant_message_id: int) -> None:
        while True:
            message = self.storage.conversations.get_message(assistant_message_id)
            if message is None or message.status != "streaming":
                return

            completion_event = self._assistant_completion_events.get(assistant_message_id)
            if completion_event is None:
                await asyncio.sleep(0.05)
                continue
            await completion_event.wait()

    async def _realize_turn_locked(
        self,
        *,
        api: TelegramBotAPI,
        plan: TurnPlan,
        inbox_update_ids: tuple[int, ...] = (),
    ) -> PreparedTurn | None:
        conversation = plan.conversation
        selection = self.registry.resolve_selection(conversation.model_alias)
        if selection is None:
            await self._send_pre_realization_reply(
                api=api,
                chat_id=conversation.chat_id,
                text=f"Unknown model alias: {conversation.model_alias}",
                reply_to_message_id=plan.reply_to_message_id,
                inbox_update_ids=inbox_update_ids,
            )
            return None

        assistant_parent_message_id = plan.assistant_parent_message_id
        if plan.user_input is not None:
            if plan.user_input.images and not selection.model.supports_images:
                await self._send_pre_realization_reply(
                    api=api,
                    chat_id=conversation.chat_id,
                    text=f"Model alias {conversation.model_alias} does not accept image input.",
                    reply_to_message_id=plan.reply_to_message_id,
                    inbox_update_ids=inbox_update_ids,
                )
                return None

        user_message_id: int | None = None

        with self.storage.transaction():
            if plan.user_input is not None:
                timestamp = plan.user_message_created_at or utcnow()
                user_message_id = self.storage.conversations.create_message(
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
                    commit=False,
                )
                self._link_logical_message_telegram_ids(
                    chat_id=conversation.chat_id,
                    logical_message_id=user_message_id,
                    primary_telegram_message_id=plan.reply_to_message_id,
                    telegram_message_ids=plan.source_telegram_message_ids,
                    commit=False,
                )
                assistant_parent_message_id = user_message_id

            if assistant_parent_message_id is None:
                raise RuntimeError("Turn plan does not have an assistant parent")

            conversation_messages = self._build_provider_conversation(assistant_parent_message_id)
            request = ChatRequest(
                model=selection.model,
                conversation=conversation_messages,
                system_prompt=conversation.system_prompt_override or self.system_prompt,
                safety_identifier=self._build_safety_identifier(conversation.user_id),
                requested_tools=selection.requested_tools,
            )
            assistant_message_id = self.storage.conversations.create_message(
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
                commit=False,
            )
            if inbox_update_ids:
                self.storage.inbox.mark_updates_realized(
                    update_ids=inbox_update_ids,
                    user_message_id=user_message_id,
                    assistant_message_id=assistant_message_id,
                    commit=False,
                )

        self._assistant_completion_events[assistant_message_id] = asyncio.Event()
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

    async def _send_pre_realization_reply(
        self,
        *,
        api: TelegramBotAPI,
        chat_id: int,
        text: str,
        reply_to_message_id: int,
        inbox_update_ids: tuple[int, ...],
    ) -> None:
        if inbox_update_ids:
            self.storage.inbox.mark_reply_started(update_ids=inbox_update_ids)
        await api.send_message(
            chat_id,
            text,
            reply_to_message_id=reply_to_message_id,
        )
        if inbox_update_ids:
            self.storage.inbox.mark_reply_sent(update_ids=inbox_update_ids)

    def _link_logical_message_telegram_ids(
        self,
        *,
        chat_id: int,
        logical_message_id: int,
        primary_telegram_message_id: int | None,
        telegram_message_ids: tuple[int, ...],
        commit: bool = True,
    ) -> None:
        seen_ids: set[int] = set()
        if primary_telegram_message_id is not None:
            seen_ids.add(primary_telegram_message_id)

        for telegram_message_id in telegram_message_ids:
            if telegram_message_id <= 0 or telegram_message_id in seen_ids:
                continue
            self.storage.conversations.link_telegram_message(
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                logical_message_id=logical_message_id,
                part_index=0,
                commit=commit,
            )
            seen_ids.add(telegram_message_id)

    def _build_provider_conversation(self, message_id: int) -> list[ConversationMessage]:
        conversation_messages: list[ConversationMessage] = []
        for item in self.storage.conversations.build_thread(message_id):
            if item.message_type == "seed":
                continue
            conversation_messages.append(
                ConversationMessage(
                    role=item.message_type,
                    content=item.content,
                    parts=tuple(self._loaded_part_from_message_part(part) for part in item.parts),
                )
            )
        return conversation_messages

    def _loaded_part_from_message_part(self, part: ContentPart) -> ContentPart:
        if part.kind == "text":
            return ContentPart(kind="text", text=part.text)
        assert part.image is not None
        image = part.image
        if image.data is not None:
            loaded_image = image
        else:
            loaded_image = ImageRef.loaded(
                mime_type=image.mime_type,
                data=self.storage.blobs.load_image_bytes(image),
            )
        return ContentPart(kind="image", image=loaded_image)

    def _build_safety_identifier(self, user_id: int) -> str | None:
        if self.safety_identifier_salt is None:
            return None
        return hashlib.sha256(f"{self.safety_identifier_salt}{user_id}".encode("utf-8")).hexdigest()

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
                self._persist_assistant_state(
                    prepared_turn=current_turn,
                    session=session,
                    full_text=full_text,
                    linked_message_ids=linked_message_ids,
                    status=persisted_status,
                )

                async def apply_render(*, render_markdown: bool | None) -> None:
                    used_markdown = final if render_markdown is None else render_markdown
                    if final:
                        self._persist_assistant_render_state(
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
                        self._sync_streaming_telegram_links(
                            chat_id=current_turn.chat_id,
                            logical_message_id=current_turn.assistant_message_id,
                            telegram_message_ids=session.message_ids,
                            linked_message_ids=linked_message_ids,
                        )
                        self._persist_assistant_render_state(
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
                    self._persist_assistant_state(
                        prepared_turn=current_turn,
                        session=session,
                        full_text=full_text,
                        linked_message_ids=linked_message_ids,
                        status=persisted_status,
                    )
                    if final:
                        self._persist_assistant_render_state(
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

            next_turn_candidate: PendingTurnCandidate | None = None
            async with self._conversation_locks[(current_turn.chat_id, current_turn.user_id)]:
                next_turn_candidate = await self._finalize_turn_locked(
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

    def _persist_assistant_state(
        self,
        *,
        prepared_turn: PreparedTurn,
        session: ReplySession,
        full_text: str,
        linked_message_ids: set[int],
        status: str,
    ) -> None:
        stored_text = full_text if full_text.strip() else ""
        self.storage.conversations.update_message(
            prepared_turn.assistant_message_id,
            content=stored_text,
            status=status,
        )
        self._sync_streaming_telegram_links(
            chat_id=prepared_turn.chat_id,
            logical_message_id=prepared_turn.assistant_message_id,
            telegram_message_ids=session.message_ids,
            linked_message_ids=linked_message_ids,
        )

    def _persist_assistant_render_state(
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
        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=prepared_turn.assistant_message_id,
            phase=phase,
            final_status=final_status,
            reply_text=stored_reply_text,
            reasoning_blocks=reasoning_blocks,
            render_markdown=render_markdown,
        )

    def _sync_streaming_telegram_links(
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
            self.storage.conversations.link_telegram_message(
                chat_id=chat_id,
                telegram_message_id=telegram_message_id,
                logical_message_id=logical_message_id,
                part_index=index,
            )
            linked_message_ids.add(telegram_message_id)

    async def _finalize_turn_locked(
        self,
        *,
        api: TelegramBotAPI,
        prepared_turn: PreparedTurn,
        session: ReplySession,
        full_text: str,
        final_status: str,
    ) -> PendingTurnCandidate | None:
        stored_text = full_text if full_text.strip() else "(empty response)"
        try:
            self.storage.conversations.update_message(
                prepared_turn.assistant_message_id,
                content=stored_text,
                status=final_status,
            )
            self._persist_assistant_render_state(
                prepared_turn=prepared_turn,
                phase=None,
                final_status=None,
                reply_text=None,
                reasoning_blocks=(),
                render_markdown=None,
            )
            for index, telegram_message_id in enumerate(session.message_ids):
                self.storage.conversations.link_telegram_message(
                    chat_id=prepared_turn.chat_id,
                    telegram_message_id=telegram_message_id,
                    logical_message_id=prepared_turn.assistant_message_id,
                    part_index=index,
                )
        finally:
            completion_event = self._assistant_completion_events.pop(prepared_turn.assistant_message_id, None)
            if completion_event is not None:
                completion_event.set()

        return await self._prepare_next_pending_turn_locked(
            api=api,
            conversation_id=prepared_turn.conversation_id,
        )

    async def _prepare_next_pending_turn_locked(
        self,
        *,
        api: TelegramBotAPI,
        conversation_id: int,
    ) -> PendingTurnCandidate | None:
        _ = api
        pending_messages = self.storage.conversations.list_pending_messages(conversation_id=conversation_id)
        if not pending_messages:
            return None
        return PendingTurnCandidate(
            conversation_id=conversation_id,
            pending_messages=tuple(pending_messages),
        )

    async def recover_interrupted_assistant_turn(
        self,
        *,
        api: TelegramBotAPI,
        assistant_message_id: int,
        reply_to_message_id: int | None = None,
        final_render_phase: str | None = None,
        final_render_status: str | None = None,
        final_render_reply_text: str | None = None,
        final_render_reasoning_blocks: tuple[str, ...] = (),
        final_render_markdown: bool | None = None,
    ) -> bool:
        message = self.storage.conversations.get_message(assistant_message_id)
        if message is None or message.message_type != "assistant":
            return False

        conversation = self.storage.conversations.get_conversation(message.conversation_id)
        if conversation is None:
            return False

        needs_interruption_recovery = message.status == "streaming"

        if needs_interruption_recovery and final_render_phase == "final_rendered" and final_render_status is not None:
            stored_text = final_render_reply_text if final_render_reply_text is not None else message.content
            self.storage.conversations.update_message(
                assistant_message_id,
                content=stored_text,
                status=final_render_status,
            )
            needs_interruption_recovery = False
        elif needs_interruption_recovery and final_render_phase == "final_pending" and final_render_status is not None:
            stored_text = final_render_reply_text if final_render_reply_text is not None else message.content
            telegram_message_ids = self.storage.conversations.list_linked_telegram_message_ids(
                logical_message_id=assistant_message_id,
            )
            session = ReplySession(
                api,
                chat_id=conversation.chat_id,
                reply_to_message_id=reply_to_message_id or (telegram_message_ids[0] if telegram_message_ids else message.id),
                prefix=f"[{message.model_alias or conversation.model_alias}] ",
                limit=self.render_limit,
                edit_interval_seconds=self.render_edit_interval_seconds,
            )
            session.message_ids = list(telegram_message_ids)

            async def apply_final_recovery_render(*, render_markdown: bool | None) -> None:
                await session.update(
                    render_reply_text(
                        stored_text,
                        list(final_render_reasoning_blocks),
                        final=True,
                        render_markdown=render_markdown,
                    ),
                    force=True,
                )

            try:
                await apply_final_recovery_render(render_markdown=final_render_markdown)
            except Exception:
                logging.exception(
                    "Failed to recover formatted final assistant render for conversation %s assistant message %s",
                    conversation.id,
                    assistant_message_id,
                )
                if final_render_markdown is False:
                    return False
                else:
                    try:
                        await apply_final_recovery_render(render_markdown=False)
                    except Exception:
                        logging.exception(
                            "Failed to recover plain-text final assistant render for conversation %s assistant message %s",
                            conversation.id,
                            assistant_message_id,
                        )
                        return False
                    else:
                        linked_message_ids = set(telegram_message_ids)
                        self._sync_streaming_telegram_links(
                            chat_id=conversation.chat_id,
                            logical_message_id=assistant_message_id,
                            telegram_message_ids=session.message_ids,
                            linked_message_ids=linked_message_ids,
                        )
                        self.storage.conversations.update_message(
                            assistant_message_id,
                            content=stored_text,
                            status=final_render_status,
                        )
                        needs_interruption_recovery = False
            else:
                linked_message_ids = set(telegram_message_ids)
                self._sync_streaming_telegram_links(
                    chat_id=conversation.chat_id,
                    logical_message_id=assistant_message_id,
                    telegram_message_ids=session.message_ids,
                    linked_message_ids=linked_message_ids,
                )
                self.storage.conversations.update_message(
                    assistant_message_id,
                    content=stored_text,
                    status=final_render_status,
                )
                needs_interruption_recovery = False

        if needs_interruption_recovery:
            interruption_note = "[interrupted: bot restarted before reply completed]"
            if message.content.strip():
                if interruption_note in message.content:
                    stored_text = message.content
                else:
                    stored_text = f"{message.content}\n\n{interruption_note}"
            else:
                stored_text = interruption_note
            telegram_message_ids = self.storage.conversations.list_linked_telegram_message_ids(
                logical_message_id=assistant_message_id,
            )
            reply_target = telegram_message_ids[0] if telegram_message_ids else reply_to_message_id
            if reply_target is None:
                return False
            session = ReplySession(
                api,
                chat_id=conversation.chat_id,
                reply_to_message_id=reply_target,
                prefix=f"[{message.model_alias or conversation.model_alias}] ",
                limit=self.render_limit,
                edit_interval_seconds=self.render_edit_interval_seconds,
            )
            session.message_ids = list(telegram_message_ids)
            try:
                await session.update(
                    render_reply_text(stored_text, [], final=True),
                    force=True,
                )
            except Exception:
                logging.exception(
                    "Failed to update interrupted assistant render for conversation %s assistant message %s",
                    conversation.id,
                    assistant_message_id,
                )
                return False
            linked_message_ids = set(telegram_message_ids)
            self._sync_streaming_telegram_links(
                chat_id=conversation.chat_id,
                logical_message_id=assistant_message_id,
                telegram_message_ids=session.message_ids,
                linked_message_ids=linked_message_ids,
            )
            self.storage.conversations.update_message(
                assistant_message_id,
                content=stored_text,
                status="failed",
            )
        else:
            stored_text = message.content

        self.storage.inbox.set_assistant_render_state(
            assistant_message_id=assistant_message_id,
            phase=None,
            final_status=None,
            reply_text=None,
            reasoning_blocks=(),
            render_markdown=None,
        )
        completion_event = self._assistant_completion_events.pop(assistant_message_id, None)
        if completion_event is not None:
            completion_event.set()

        next_turn_candidate: PendingTurnCandidate | None = None
        lock_key = (conversation.chat_id, conversation.user_id)
        async with self._conversation_locks[lock_key]:
            next_turn_candidate = await self._prepare_next_pending_turn_locked(
                api=api,
                conversation_id=conversation.id,
            )
        if next_turn_candidate is None:
            return True
        prepared_turn = await self._prepare_turn_from_candidate(api=api, candidate=next_turn_candidate)
        if prepared_turn is None:
            return True
        await self._run_conversation_loop(api=api, prepared_turn=prepared_turn)
        return True

    async def _prepare_turn_from_candidate(
        self,
        *,
        api: TelegramBotAPI,
        candidate: PendingTurnCandidate,
    ) -> PreparedTurn | None:
        conversation = self.storage.conversations.get_conversation(candidate.conversation_id)
        if conversation is None:
            return None
        lock_key = (conversation.chat_id, conversation.user_id)
        async with self._turn_claim_locks[lock_key]:
            successful_pending_messages: list[PendingMessage] = []
            processed_pending_ids: list[int] = []
            for pending_message in candidate.pending_messages:
                try:
                    stored_user_input = await self._store_user_input_images(
                        api=api,
                        user_input=self.engine._stored_user_input_from_pending_message(pending_message),
                    )
                except Exception as exc:
                    logging.error(
                        "Failed to ingest image input for chat %s user %s message %s",
                        conversation.chat_id,
                        conversation.user_id,
                        pending_message.telegram_message_id,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    await self._send_image_download_error(
                        api=api,
                        chat_id=conversation.chat_id,
                        reply_to_message_id=pending_message.telegram_message_id,
                    )
                    processed_pending_ids.append(pending_message.id)
                    continue

                successful_pending_messages.append(
                    replace(
                        pending_message,
                        images=stored_user_input.images,
                        parts=stored_user_input.parts,
                    )
                )
                processed_pending_ids.append(pending_message.id)

            if not processed_pending_ids:
                return None

            async with self._conversation_locks[lock_key]:
                self.storage.conversations.delete_pending_messages(
                    pending_message_ids=tuple(processed_pending_ids),
                )
                if not successful_pending_messages:
                    return None
                turn_plan = self.engine.build_pending_turn(
                    conversation_id=candidate.conversation_id,
                    pending_messages=successful_pending_messages,
                )
                if turn_plan is None:
                    return None
                return await self._realize_turn_locked(api=api, plan=turn_plan)

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

    async def _store_turn_plan_images(
        self,
        *,
        api: TelegramBotAPI,
        plan: TurnPlan,
    ) -> TurnPlan:
        if plan.user_input is None:
            return plan
        stored_user_input = await self._store_user_input_images(api=api, user_input=plan.user_input)
        return replace(plan, user_input=stored_user_input)

    async def _try_store_turn_plan_images(
        self,
        *,
        api: TelegramBotAPI,
        plan: TurnPlan,
    ) -> TurnPlan | None:
        try:
            return await self._store_turn_plan_images(api=api, plan=plan)
        except Exception as exc:
            logging.error(
                "Failed to ingest image input for chat %s user %s message %s",
                plan.conversation.chat_id,
                plan.conversation.user_id,
                plan.reply_to_message_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self.storage.conversations.delete_conversation_if_empty(plan.conversation.id)
            await self._send_image_download_error(
                api=api,
                chat_id=plan.conversation.chat_id,
                reply_to_message_id=plan.reply_to_message_id,
            )
            return None

    async def _store_user_input_images(
        self,
        *,
        api: TelegramBotAPI,
        user_input: StoredUserInput,
    ) -> StoredUserInput:
        effective_parts = user_input.parts or build_content_parts(user_input.content, user_input.images)
        if not user_input.images:
            return user_input

        stored_parts: list[ContentPart] = []
        stored_images: list[ImageRef] = []
        for part in effective_parts:
            if part.kind == "text":
                stored_parts.append(ContentPart(kind="text", text=part.text))
                continue
            if part.image is None:
                continue
            image = part.image
            if image.kind == "stored":
                stored_image = image
            elif image.kind == "telegram":
                assert image.file_id is not None
                image_bytes = await api.download_file_bytes(image.file_id)
                stored_image = self.storage.blobs.store_image(mime_type=image.mime_type, data=image_bytes)
            elif image.kind == "loaded":
                assert image.data is not None
                stored_image = self.storage.blobs.store_image(mime_type=image.mime_type, data=image.data)
            else:
                raise ValueError(f"Unsupported image ref kind: {image.kind}")
            stored_images.append(stored_image)
            stored_parts.append(ContentPart(kind="image", image=stored_image))

        stored_image_tuple = tuple(stored_images)
        return StoredUserInput(
            content=user_input.content,
            images=stored_image_tuple,
            parts=tuple(stored_parts) if stored_parts else build_content_parts(user_input.content, stored_image_tuple),
        )

    async def _send_image_download_error(
        self,
        *,
        api: TelegramBotAPI,
        chat_id: int,
        reply_to_message_id: int,
    ) -> None:
        try:
            await api.send_message(
                chat_id,
                "Failed to download the image from Telegram. Please resend it and try again.",
                reply_to_message_id=reply_to_message_id,
            )
        except Exception:
            logging.exception(
                "Failed to send image download error reply for chat %s message %s",
                chat_id,
                reply_to_message_id,
            )
