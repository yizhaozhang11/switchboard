from __future__ import annotations

from collections.abc import Awaitable, Callable
import time

from app.richtext import RichText
from app.telegram_api import TelegramBotAPI


def split_text(text: str, limit: int) -> list[str]:
    if not text:
        return [""]

    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        chunk = remaining[:limit]
        split_at = chunk.rfind("\n")
        if split_at < limit // 2:
            split_at = chunk.rfind(" ")
        if split_at < limit // 2:
            split_at = limit
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts or [""]


def split_rich_text(text: RichText, limit: int) -> list[RichText]:
    if not text:
        return [RichText("")]

    parts: list[RichText] = []
    remaining = text
    while len(remaining) > limit:
        chunk = remaining.text[:limit]
        split_at = chunk.rfind("\n")
        if split_at < limit // 2:
            split_at = chunk.rfind(" ")
        if split_at < limit // 2:
            split_at = limit

        part_text = remaining.text[:split_at].rstrip()
        if part_text:
            parts.append(remaining[: len(part_text)])
        else:
            parts.append(remaining[:split_at])

        remaining_text = remaining.text[split_at:]
        trim = len(remaining_text) - len(remaining_text.lstrip())
        remaining = remaining[split_at + trim :]
    if remaining:
        parts.append(remaining)
    return parts or [RichText("")]


def render_reply_text(
    reply_text: str,
    reasoning_blocks: list[str],
    *,
    final: bool = False,
    render_markdown: bool | None = None,
) -> RichText:
    rendered = RichText()
    for block in reasoning_blocks:
        if not block:
            continue
        rendered += RichText.quote(block, collapsed=final) + "\n"
    if reply_text:
        use_markdown = final if render_markdown is None else render_markdown
        rendered += RichText.from_markdown(reply_text) if use_markdown else RichText(reply_text)
    return rendered


class ReplySession:
    def __init__(
        self,
        api: TelegramBotAPI,
        *,
        chat_id: int,
        reply_to_message_id: int,
        prefix: str = "",
        limit: int = 3900,
        edit_interval_seconds: float = 1.0,
    ) -> None:
        self.api = api
        self.chat_id = chat_id
        self.reply_to_message_id = reply_to_message_id
        self.prefix = prefix
        self.limit = limit
        self.edit_interval_seconds = edit_interval_seconds
        self.message_ids: list[int] = []
        self._last_parts: list[RichText] = []
        self._last_render_at = 0.0

    async def update(
        self,
        text: str | RichText,
        *,
        force: bool = False,
        on_sent_message_id: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> None:
        now = time.monotonic()
        if not force and self.message_ids and now - self._last_render_at < self.edit_interval_seconds:
            return
        rendered = RichText.coerce(text)
        parts = [self.prefix + part for part in split_rich_text(rendered, self.limit - len(self.prefix))]
        await self._apply(parts, on_sent_message_id=on_sent_message_id)
        self._last_render_at = now

    async def _apply(
        self,
        parts: list[RichText],
        *,
        on_sent_message_id: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> None:
        for index, part in enumerate(parts):
            if index < len(self.message_ids):
                if index < len(self._last_parts) and self._last_parts[index] == part:
                    continue
                await self.api.edit_message_text(self.chat_id, self.message_ids[index], part)
            else:
                reply_to = self.reply_to_message_id if index == 0 else self.message_ids[index - 1]
                message_id = await self.api.send_message(self.chat_id, part, reply_to_message_id=reply_to)
                self.message_ids.append(message_id)
                if on_sent_message_id is not None:
                    await on_sent_message_id(index, message_id)
            if index < len(self._last_parts):
                self._last_parts[index] = part
            else:
                self._last_parts.append(part)

        while len(self.message_ids) > len(parts):
            message_id = self.message_ids.pop()
            self._last_parts.pop()
            await self.api.delete_message(self.chat_id, message_id)
