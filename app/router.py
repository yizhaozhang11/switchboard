from __future__ import annotations

from app.types import (
    ChatAction,
    ContentPart,
    CommandAction,
    IgnoreAction,
    IncomingMessage,
    ImageRef,
    RouteAction,
    build_content_parts,
)


class Router:
    def __init__(self, bot_username: str | None) -> None:
        self.bot_username = (bot_username or "").lstrip("@").casefold()

    def route(self, message: IncomingMessage, _settings: object | None = None) -> RouteAction:
        command_action = self._parse_command(message.text, images=message.images, parts=message.parts)
        if command_action is not None:
            return command_action

        stripped = message.text.strip()
        if not stripped and not message.images:
            return IgnoreAction("empty text")

        action_parts = message.parts or build_content_parts(stripped, message.images)
        return ChatAction(content=stripped, intent="plain", images=message.images, parts=action_parts)

    def _parse_command(
        self,
        text: str,
        *,
        images: tuple[ImageRef, ...] = (),
        parts: tuple[ContentPart, ...] = (),
    ) -> RouteAction | None:
        first_line, has_newline, remainder = text.partition("\n")
        head = first_line.strip()
        if not head.startswith("/"):
            return None

        command_tokens = head.split(maxsplit=2)
        command_name = self._normalize_command_token(command_tokens[0])
        if command_name is None:
            return None

        if command_name == "ping":
            return CommandAction(name="ping")

        if command_name == "help":
            argument = command_tokens[1].strip() if len(command_tokens) >= 2 else None
            return CommandAction(name="help", argument=argument)

        if command_name == "models":
            return CommandAction(name="models")

        if command_name == "model":
            if len(command_tokens) < 2:
                return CommandAction(name="usage_error", content="Usage: /model <alias>")
            return CommandAction(name="model", argument=command_tokens[1].strip())

        if command_name == "mode":
            if len(command_tokens) < 2:
                return CommandAction(name="usage_error", content="Usage: /mode auto|mention|off")
            return CommandAction(name="mode", argument=command_tokens[1].strip())

        if command_name == "togglechat":
            argument = command_tokens[1].strip() if len(command_tokens) >= 2 else None
            return CommandAction(name="togglechat", argument=argument)

        if command_name == "toggleuser":
            argument = command_tokens[1].strip() if len(command_tokens) >= 2 else None
            return CommandAction(name="toggleuser", argument=argument)

        if command_name == "whitelist":
            return CommandAction(name="whitelist")

        if command_name == "c":
            if len(command_tokens) < 2:
                return CommandAction(name="usage_error", content="Usage: /c <alias> <content>")
            model_alias = command_tokens[1].strip()
            content = self._extract_command_content(command_tokens, has_newline, remainder)
            if not content and not images:
                return CommandAction(name="usage_error", content="Usage: /c <alias> <content>")
            return ChatAction(
                content=content,
                intent="choose_model",
                model_alias=model_alias,
                images=images,
                parts=self._rewrite_command_parts(command_name="c", content=content, parts=parts, images=images),
            )

        if command_name == "new":
            inline_parts = head.split(maxsplit=1)
            inline_content = inline_parts[1] if len(inline_parts) == 2 else ""
            content = self._combine_command_content(inline_content, has_newline, remainder)
            if not content and not images:
                return CommandAction(name="usage_error", content="Usage: /new <content>")
            return ChatAction(
                content=content,
                intent="new",
                images=images,
                parts=self._rewrite_command_parts(command_name="new", content=content, parts=parts, images=images),
            )

        if command_name == "s":
            inline_parts = head.split(maxsplit=1)
            inline_content = inline_parts[1] if len(inline_parts) == 2 else ""
            content = self._combine_command_content(inline_content, has_newline, remainder)
            if not content:
                return CommandAction(name="usage_error", content="Usage: /s <prompt>")
            return ChatAction(
                content="",
                intent="set_system_prompt",
                system_prompt=content,
                parts=self._rewrite_command_parts(command_name="s", content=content, parts=parts, images=()),
            )

        return CommandAction(name="usage_error", content=f"Unknown command: /{command_name}")

    def _extract_command_content(
        self,
        parts: list[str],
        has_newline: bool,
        remainder: str,
        *,
        content_index: int = 2,
    ) -> str:
        inline_content = parts[content_index] if len(parts) > content_index else ""
        return self._combine_command_content(inline_content, has_newline, remainder)

    def _combine_command_content(self, inline_content: str, has_newline: bool, remainder: str) -> str:
        body = remainder if has_newline else ""
        if inline_content and body:
            content = inline_content + "\n" + body
        else:
            content = inline_content or body
        return content.strip()

    def _rewrite_command_parts(
        self,
        *,
        command_name: str,
        content: str,
        parts: tuple[ContentPart, ...],
        images: tuple[ImageRef, ...],
    ) -> tuple[ContentPart, ...]:
        if not parts:
            return build_content_parts(content, images)

        rewritten_parts: list[ContentPart] = []
        rewrote_first_text = False
        for part in parts:
            if part.kind != "text" or rewrote_first_text:
                rewritten_parts.append(part)
                continue

            rewrote_first_text = True
            rewritten_text = self._strip_command_from_first_text_part(command_name=command_name, text=part.text)
            if rewritten_text:
                rewritten_parts.append(ContentPart(kind="text", text=rewritten_text))

        if not rewrote_first_text:
            return build_content_parts(content, images)
        return tuple(rewritten_parts)

    def _strip_command_from_first_text_part(self, *, command_name: str, text: str) -> str:
        first_line, has_newline, remainder = text.partition("\n")
        head = first_line.strip()
        if command_name == "c":
            return self._extract_command_content(head.split(maxsplit=2), has_newline, remainder)
        if command_name == "new":
            inline_parts = head.split(maxsplit=1)
            inline_content = inline_parts[1] if len(inline_parts) == 2 else ""
            return self._combine_command_content(inline_content, has_newline, remainder)
        if command_name == "s":
            inline_parts = head.split(maxsplit=1)
            inline_content = inline_parts[1] if len(inline_parts) == 2 else ""
            return self._combine_command_content(inline_content, has_newline, remainder)
        return text

    def _normalize_command_token(self, token: str) -> str | None:
        if not token.startswith("/"):
            return None
        command = token[1:]
        name, _, target = command.partition("@")
        if target and self.bot_username and target.casefold() != self.bot_username:
            return None
        return name.casefold()
