from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from app.d1_storage import D1Storage
from app.types import ChatAction, ContentPart, ImageRef, IncomingMessage, PendingMessage, StoredConversation, StoredMessage, build_content_parts


@dataclass(frozen=True)
class StoredUserInput:
    content: str
    images: tuple[ImageRef, ...]
    parts: tuple[ContentPart, ...]


@dataclass(frozen=True)
class DeferredAction:
    assistant_message_id: int


@dataclass(frozen=True)
class TurnPlan:
    conversation: StoredConversation
    reply_to_message_id: int
    source_telegram_message_ids: tuple[int, ...] = ()
    user_input: StoredUserInput | None = None
    user_parent_message_id: int | None = None
    assistant_parent_message_id: int | None = None
    user_message_created_at: str | None = None


@dataclass(frozen=True)
class ResolvedState:
    conversation: StoredConversation
    anchor_message: StoredMessage
    history_is_empty: bool
    last_history_message_type: Literal["user", "assistant"] | None
    is_conversation_tip: bool


class ConversationEngine:
    def __init__(self, *, storage: D1Storage, conversation_timeout_seconds: int) -> None:
        self.storage = storage
        self.conversation_timeout_seconds = conversation_timeout_seconds

    async def begin_action(
        self,
        *,
        incoming_message: IncomingMessage,
        default_model_alias: str,
        action: ChatAction,
        user_input: StoredUserInput | None,
    ) -> TurnPlan | DeferredAction | None:
        resolved_state, deferred_action = await self._resolve_action_state(
            incoming_message=incoming_message,
            action=action,
        )
        if deferred_action is not None:
            return deferred_action

        if action.intent == "set_system_prompt":
            return await self._handle_system_prompt(
                incoming_message=incoming_message,
                default_model_alias=default_model_alias,
                action=action,
                resolved_state=resolved_state,
            )

        assert user_input is not None
        if action.intent == "plain" and resolved_state is None:
            if await self._merge_reply_into_pending_message_if_needed(
                incoming_message=incoming_message,
                user_input=user_input,
            ):
                return None

        if action.intent == "plain":
            return await self._handle_plain(
                incoming_message=incoming_message,
                default_model_alias=default_model_alias,
                user_input=user_input,
                resolved_state=resolved_state,
            )
        if action.intent == "new":
            return await self._handle_new(
                incoming_message=incoming_message,
                default_model_alias=default_model_alias,
                user_input=user_input,
                resolved_state=resolved_state,
            )
        if action.intent == "choose_model":
            return await self._handle_choose_model(
                incoming_message=incoming_message,
                user_input=user_input,
                resolved_state=resolved_state,
                model_alias=action.model_alias or "",
            )

        raise RuntimeError(f"Unknown chat action intent: {action.intent}")

    async def find_deferred_action(
        self,
        *,
        incoming_message: IncomingMessage,
        action: ChatAction,
    ) -> DeferredAction | None:
        _, deferred_action = await self._resolve_action_state(
            incoming_message=incoming_message,
            action=action,
        )
        return deferred_action

    async def _resolve_state(
        self,
        *,
        incoming_message: IncomingMessage,
        intent: str,
    ) -> ResolvedState | None:
        attached_message: StoredMessage | None = None
        if incoming_message.reply_to_message_id is not None:
            attached_message = await self.storage.conversations.get_message_by_telegram(
                chat_id=incoming_message.chat_id,
                telegram_message_id=incoming_message.reply_to_message_id,
            )
        elif intent == "plain":
            cutoff = (
                datetime.now(timezone.utc) - timedelta(seconds=self.conversation_timeout_seconds)
            ).replace(microsecond=0).isoformat()
            attached_message = await self.storage.conversations.find_recent_state_message(
                chat_id=incoming_message.chat_id,
                user_id=incoming_message.user_id,
                not_before=cutoff,
            )

        if attached_message is None:
            return None

        conversation = await self.storage.conversations.get_conversation(attached_message.conversation_id)
        if conversation is None:
            return None

        thread = await self.storage.conversations.build_thread(attached_message.id)
        visible_thread = [message for message in thread if message.message_type != "seed"]
        last_history_message_type = visible_thread[-1].message_type if visible_thread else None
        tip_message = await self.storage.conversations.get_conversation_tip_message(conversation.id)
        return ResolvedState(
            conversation=conversation,
            anchor_message=attached_message,
            history_is_empty=not visible_thread,
            last_history_message_type=last_history_message_type,
            is_conversation_tip=tip_message is not None and tip_message.id == attached_message.id,
        )

    async def _resolve_action_state(
        self,
        *,
        incoming_message: IncomingMessage,
        action: ChatAction,
    ) -> tuple[ResolvedState | None, DeferredAction | None]:
        resolved_state = await self._resolve_state(
            incoming_message=incoming_message,
            intent=action.intent,
        )
        deferred_action = await self._defer_streaming_assistant_branch(
            incoming_message=incoming_message,
            action=action,
            resolved_state=resolved_state,
        )
        return resolved_state, deferred_action

    async def _defer_streaming_assistant_branch(
        self,
        *,
        incoming_message: IncomingMessage,
        action: ChatAction,
        resolved_state: ResolvedState | None,
    ) -> DeferredAction | None:
        if action.intent in {"new", "set_system_prompt"}:
            return None
        if resolved_state is None:
            return None
        blocking_assistant = await self._resolve_streaming_history_assistant(resolved_state.anchor_message)
        if blocking_assistant is None:
            return None
        if (
            blocking_assistant.id == resolved_state.anchor_message.id
            and self._would_queue_on_streaming_anchor(
                incoming_user_id=incoming_message.user_id,
                action=action,
                resolved_state=resolved_state,
            )
        ):
            return None
        return DeferredAction(assistant_message_id=blocking_assistant.id)

    def _would_queue_on_streaming_anchor(
        self,
        *,
        incoming_user_id: int,
        action: ChatAction,
        resolved_state: ResolvedState,
    ) -> bool:
        if not resolved_state.is_conversation_tip:
            return False
        if resolved_state.conversation.user_id != incoming_user_id:
            return False
        if action.intent == "plain":
            return True
        if action.intent == "choose_model":
            return action.model_alias == resolved_state.conversation.model_alias
        return False

    async def _resolve_streaming_history_assistant(self, anchor_message: StoredMessage) -> StoredMessage | None:
        visible_history_message = await self._resolve_visible_history_message(anchor_message)
        if (
            visible_history_message is None
            or visible_history_message.message_type != "assistant"
            or visible_history_message.status != "streaming"
        ):
            return None
        return visible_history_message

    async def _resolve_visible_history_message(self, anchor_message: StoredMessage) -> StoredMessage | None:
        current_message = anchor_message
        while current_message.message_type == "seed":
            if current_message.parent_message_id is None:
                return None
            parent_message = await self.storage.conversations.get_message(current_message.parent_message_id)
            if parent_message is None:
                return None
            current_message = parent_message
        return current_message

    async def _handle_plain(
        self,
        *,
        incoming_message: IncomingMessage,
        default_model_alias: str,
        user_input: StoredUserInput,
        resolved_state: ResolvedState | None,
    ) -> TurnPlan | None:
        if resolved_state is None:
            conversation = await self.storage.conversations.create_conversation(
                chat_id=incoming_message.chat_id,
                user_id=incoming_message.user_id,
                model_alias=default_model_alias,
            )
            return await self._queue_or_plan_user_turn(
                conversation=conversation,
                user_input=user_input,
                reply_to_message_id=incoming_message.message_id,
                source_telegram_message_ids=incoming_message.source_message_ids,
                parent_message_id=None,
            )

        if resolved_state.last_history_message_type != "user":
            conversation = await self._continue_or_fork_from_state(
                resolved_state,
                incoming_user_id=incoming_message.user_id,
            )
            return await self._queue_or_plan_user_turn(
                conversation=conversation,
                user_input=user_input,
                reply_to_message_id=incoming_message.message_id,
                source_telegram_message_ids=incoming_message.source_message_ids,
                parent_message_id=resolved_state.anchor_message.id,
            )

        if resolved_state.conversation.user_id != incoming_message.user_id:
            return None

        visible_history_message = await self._resolve_visible_history_message(resolved_state.anchor_message)
        if visible_history_message is None or visible_history_message.message_type != "user":
            return None

        merged_input = self._combine_user_inputs(
            self._stored_user_input_from_message(visible_history_message),
            user_input,
        )
        conversation = await self._branch_from_state(
            resolved_state,
            incoming_user_id=incoming_message.user_id,
        )
        return await self._queue_or_plan_user_turn(
            conversation=conversation,
            user_input=merged_input,
            reply_to_message_id=incoming_message.message_id,
            source_telegram_message_ids=incoming_message.source_message_ids,
            parent_message_id=visible_history_message.parent_message_id,
        )

    async def _handle_new(
        self,
        *,
        incoming_message: IncomingMessage,
        default_model_alias: str,
        user_input: StoredUserInput,
        resolved_state: ResolvedState | None,
    ) -> TurnPlan | None:
        if incoming_message.reply_to_message_id is not None and resolved_state is not None and resolved_state.history_is_empty:
            conversation = await self._continue_or_fork_from_state(
                resolved_state,
                incoming_user_id=incoming_message.user_id,
            )
            return await self._queue_or_plan_user_turn(
                conversation=conversation,
                user_input=user_input,
                reply_to_message_id=incoming_message.message_id,
                source_telegram_message_ids=incoming_message.source_message_ids,
                parent_message_id=resolved_state.anchor_message.id,
            )

        conversation = await self.storage.conversations.create_conversation(
            chat_id=incoming_message.chat_id,
            user_id=incoming_message.user_id,
            model_alias=default_model_alias,
        )
        return await self._queue_or_plan_user_turn(
            conversation=conversation,
            user_input=user_input,
            reply_to_message_id=incoming_message.message_id,
            source_telegram_message_ids=incoming_message.source_message_ids,
            parent_message_id=None,
        )

    async def _handle_choose_model(
        self,
        *,
        incoming_message: IncomingMessage,
        user_input: StoredUserInput,
        resolved_state: ResolvedState | None,
        model_alias: str,
    ) -> TurnPlan | None:
        if (
            incoming_message.reply_to_message_id is not None
            and resolved_state is not None
            and resolved_state.last_history_message_type != "user"
        ):
            conversation = await self._continue_or_fork_from_state(
                resolved_state,
                incoming_user_id=incoming_message.user_id,
                model_alias=model_alias,
            )
            return await self._queue_or_plan_user_turn(
                conversation=conversation,
                user_input=user_input,
                reply_to_message_id=incoming_message.message_id,
                source_telegram_message_ids=incoming_message.source_message_ids,
                parent_message_id=resolved_state.anchor_message.id,
            )

        conversation = await self.storage.conversations.create_conversation(
            chat_id=incoming_message.chat_id,
            user_id=incoming_message.user_id,
            model_alias=model_alias,
        )
        return await self._queue_or_plan_user_turn(
            conversation=conversation,
            user_input=user_input,
            reply_to_message_id=incoming_message.message_id,
            source_telegram_message_ids=incoming_message.source_message_ids,
            parent_message_id=None,
        )

    async def _handle_system_prompt(
        self,
        *,
        incoming_message: IncomingMessage,
        default_model_alias: str,
        action: ChatAction,
        resolved_state: ResolvedState | None,
    ) -> TurnPlan | None:
        system_prompt_override = action.system_prompt or ""

        if incoming_message.reply_to_message_id is None or resolved_state is None:
            conversation = await self.storage.conversations.create_conversation(
                chat_id=incoming_message.chat_id,
                user_id=incoming_message.user_id,
                model_alias=default_model_alias,
                system_prompt_override=system_prompt_override,
            )
            await self._create_seed_message(
                conversation=conversation,
                telegram_message_id=incoming_message.message_id,
                parent_message_id=None,
                source_telegram_message_ids=incoming_message.source_message_ids,
            )
            return None

        conversation = await self._branch_from_state(
            resolved_state,
            incoming_user_id=incoming_message.user_id,
            system_prompt_override=system_prompt_override,
        )
        seed_message_id = await self._create_seed_message(
            conversation=conversation,
            telegram_message_id=incoming_message.message_id,
            parent_message_id=resolved_state.anchor_message.id,
            source_telegram_message_ids=incoming_message.source_message_ids,
        )

        if resolved_state.last_history_message_type == "user":
            return TurnPlan(
                conversation=conversation,
                reply_to_message_id=incoming_message.message_id,
                source_telegram_message_ids=incoming_message.source_message_ids,
                assistant_parent_message_id=seed_message_id,
            )
        return None

    async def _continue_or_fork_from_state(
        self,
        resolved_state: ResolvedState,
        *,
        incoming_user_id: int,
        model_alias: str | None = None,
        system_prompt_override: str | None | object = None,
    ) -> StoredConversation:
        target_model_alias = model_alias or resolved_state.conversation.model_alias
        if system_prompt_override is None:
            target_system_prompt = resolved_state.conversation.system_prompt_override
        else:
            target_system_prompt = system_prompt_override

        if (
            resolved_state.is_conversation_tip
            and resolved_state.conversation.user_id == incoming_user_id
            and resolved_state.conversation.model_alias == target_model_alias
            and resolved_state.conversation.system_prompt_override == target_system_prompt
        ):
            return resolved_state.conversation

        return await self.storage.conversations.create_conversation(
            chat_id=resolved_state.conversation.chat_id,
            user_id=incoming_user_id,
            model_alias=target_model_alias,
            system_prompt_override=target_system_prompt,
        )

    async def _branch_from_state(
        self,
        resolved_state: ResolvedState,
        *,
        incoming_user_id: int,
        model_alias: str | None = None,
        system_prompt_override: str | None | object = None,
    ) -> StoredConversation:
        target_model_alias = model_alias or resolved_state.conversation.model_alias
        if system_prompt_override is None:
            target_system_prompt = resolved_state.conversation.system_prompt_override
        else:
            target_system_prompt = system_prompt_override
        return await self.storage.conversations.create_conversation(
            chat_id=resolved_state.conversation.chat_id,
            user_id=incoming_user_id,
            model_alias=target_model_alias,
            system_prompt_override=target_system_prompt,
        )

    async def _create_seed_message(
        self,
        *,
        conversation: StoredConversation,
        telegram_message_id: int,
        parent_message_id: int | None,
        source_telegram_message_ids: tuple[int, ...] = (),
    ) -> int:
        seed_message_id = await self.storage.conversations.create_message(
            conversation_id=conversation.id,
            chat_id=conversation.chat_id,
            telegram_message_id=telegram_message_id,
            message_type="seed",
            parent_message_id=parent_message_id,
            provider=None,
            model_id=None,
            model_alias=conversation.model_alias,
            content=conversation.system_prompt_override or "",
            status="complete",
        )
        await self._link_logical_message_telegram_ids(
            chat_id=conversation.chat_id,
            logical_message_id=seed_message_id,
            primary_telegram_message_id=telegram_message_id,
            telegram_message_ids=source_telegram_message_ids,
        )
        return seed_message_id

    async def _queue_or_plan_user_turn(
        self,
        *,
        conversation: StoredConversation,
        user_input: StoredUserInput,
        reply_to_message_id: int,
        parent_message_id: int | None,
        source_telegram_message_ids: tuple[int, ...] = (),
        user_message_created_at: str | None = None,
    ) -> TurnPlan | None:
        if await self.storage.conversations.is_conversation_streaming(conversation.id):
            await self.storage.conversations.enqueue_pending_message(
                conversation_id=conversation.id,
                telegram_message_id=reply_to_message_id,
                source_telegram_message_ids=source_telegram_message_ids,
                content=user_input.content,
                images=user_input.images,
                parts=user_input.parts,
            )
            return None

        return TurnPlan(
            conversation=conversation,
            reply_to_message_id=reply_to_message_id,
            source_telegram_message_ids=source_telegram_message_ids,
            user_input=user_input,
            user_parent_message_id=parent_message_id,
            user_message_created_at=user_message_created_at,
        )

    async def build_pending_turn(
        self,
        *,
        conversation_id: int,
        pending_messages: list[PendingMessage],
    ) -> TurnPlan | None:
        if not pending_messages:
            return None

        conversation = await self.storage.conversations.get_conversation(conversation_id)
        tip_message = await self.storage.conversations.get_conversation_tip_message(conversation_id)
        if conversation is None or tip_message is None:
            return None

        merged_content = self._merge_pending_messages(pending_messages)
        reply_to_message_id = pending_messages[-1].telegram_message_id
        source_telegram_message_ids = ConversationEngine._merge_telegram_message_ids(
            *(tuple(pm.source_telegram_message_ids or (pm.telegram_message_id,)) for pm in pending_messages)
        )
        return await self._queue_or_plan_user_turn(
            conversation=conversation,
            user_input=merged_content,
            reply_to_message_id=reply_to_message_id,
            parent_message_id=tip_message.id,
            source_telegram_message_ids=source_telegram_message_ids,
            user_message_created_at=pending_messages[-1].created_at,
        )

    async def _merge_reply_into_pending_message_if_needed(
        self,
        *,
        incoming_message: IncomingMessage,
        user_input: StoredUserInput,
    ) -> bool:
        if incoming_message.reply_to_message_id is None:
            return False

        pending_message = await self.storage.conversations.find_pending_message_by_telegram(
            chat_id=incoming_message.chat_id,
            telegram_message_id=incoming_message.reply_to_message_id,
        )
        if pending_message is None:
            return False

        conversation = await self.storage.conversations.get_conversation(pending_message.conversation_id)
        if conversation is None:
            return False
        if conversation.user_id != incoming_message.user_id:
            return True

        merged_input = self._combine_user_inputs(
            self._stored_user_input_from_pending_message(pending_message),
            user_input,
        )
        merged_source_telegram_message_ids = self._merge_telegram_message_ids(
            pending_message.source_telegram_message_ids or (pending_message.telegram_message_id,),
            incoming_message.source_message_ids,
        )
        await self.storage.conversations.update_pending_message(
            pending_message.id,
            telegram_message_id=incoming_message.message_id,
            source_telegram_message_ids=merged_source_telegram_message_ids,
            content=merged_input.content,
            images=merged_input.images,
            parts=merged_input.parts,
        )
        return True

    @staticmethod
    def _stored_user_input_from_message(message: StoredMessage) -> StoredUserInput:
        return StoredUserInput(
            content=message.content,
            images=message.images,
            parts=message.parts or build_content_parts(message.content, message.images),
        )

    @staticmethod
    def _stored_user_input_from_pending_message(pending_message: PendingMessage) -> StoredUserInput:
        return StoredUserInput(
            content=pending_message.content,
            images=pending_message.images,
            parts=pending_message.parts or build_content_parts(pending_message.content, pending_message.images),
        )

    @staticmethod
    def _combine_user_inputs(left: StoredUserInput, right: StoredUserInput) -> StoredUserInput:
        merged_content = ConversationEngine._join_text_blocks(left.content, right.content)
        left_parts = list(left.parts or build_content_parts(left.content, left.images))
        right_parts = list(right.parts or build_content_parts(right.content, right.images))

        if left_parts and right_parts and left_parts[-1].kind == "text" and right_parts[0].kind == "text":
            left_tail = left_parts[-1].text
            right_head = right_parts[0].text
            left_parts[-1] = ContentPart(kind="text", text=ConversationEngine._join_text_blocks(left_tail, right_head))
            merged_parts = left_parts + right_parts[1:]
        else:
            merged_parts = left_parts + right_parts

        merged_images = left.images + right.images
        return StoredUserInput(
            content=merged_content,
            images=merged_images,
            parts=tuple(merged_parts) if merged_parts else build_content_parts(merged_content, merged_images),
        )

    @staticmethod
    def _merge_pending_messages(pending_messages: list[PendingMessage]) -> StoredUserInput:
        if len(pending_messages) == 1:
            return StoredUserInput(
                content=pending_messages[0].content,
                images=pending_messages[0].images,
                parts=pending_messages[0].parts,
            )

        sections = []
        parts: list[ContentPart] = [
            ContentPart(kind="text", text="Additional user messages sent while you were replying:")
        ]
        images: list[ImageRef] = []
        for index, pending_message in enumerate(pending_messages, start=1):
            section_text = f"{index}.\n{ConversationEngine._format_pending_message_summary(pending_message)}"
            sections.append(section_text)
            parts.extend(ConversationEngine._number_pending_message_parts(index=index, pending_message=pending_message))
            images.extend(pending_message.images)
        return StoredUserInput(
            content="Additional user messages sent while you were replying:\n\n" + "\n\n".join(sections),
            images=tuple(images),
            parts=tuple(parts),
        )

    @staticmethod
    def _number_pending_message_parts(
        *,
        index: int,
        pending_message: PendingMessage,
    ) -> tuple[ContentPart, ...]:
        message_parts = pending_message.parts or build_content_parts(
            pending_message.content,
            pending_message.images,
        )
        numbered_parts: list[ContentPart] = []
        numbered_first_text = False
        for part in message_parts:
            if part.kind == "text" and not numbered_first_text:
                numbered_parts.append(ContentPart(kind="text", text=f"{index}.\n{part.text}"))
                numbered_first_text = True
                continue
            numbered_parts.append(part)

        if not numbered_first_text:
            numbered_parts.insert(
                0,
                ContentPart(
                    kind="text",
                    text=f"{index}.\n{ConversationEngine._format_pending_message_summary(pending_message)}",
                ),
            )
        return tuple(numbered_parts)

    @staticmethod
    def _format_pending_message_summary(pending_message: PendingMessage) -> str:
        if pending_message.content:
            return pending_message.content
        image_count = len(pending_message.images)
        if image_count == 1:
            return "[1 image]"
        if image_count > 1:
            return f"[{image_count} images]"
        return "(empty message)"

    @staticmethod
    def _join_text_blocks(left: str, right: str) -> str:
        if left and right:
            return f"{left}\n\n{right}"
        return left or right

    @staticmethod
    def _merge_telegram_message_ids(*telegram_message_id_groups: tuple[int, ...]) -> tuple[int, ...]:
        merged_ids: list[int] = []
        seen_ids: set[int] = set()
        for telegram_message_id_group in telegram_message_id_groups:
            for telegram_message_id in telegram_message_id_group:
                if telegram_message_id <= 0 or telegram_message_id in seen_ids:
                    continue
                merged_ids.append(telegram_message_id)
                seen_ids.add(telegram_message_id)
        return tuple(merged_ids)

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
