from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    command: str
    arguments: str
    description: str
    menu_description: str | None = None

    @property
    def usage(self) -> str:
        return f"/{self.command}{self.arguments}"

    @property
    def help_line(self) -> str:
        return f"- {self.usage} - {self.description}"

    @property
    def telegram_description(self) -> str:
        return self.menu_description or self.description


COMMAND_HELP_TOPICS = ("n", "c", "s")

CORE_COMMANDS = (
    CommandSpec("help", "", "show this help", "Show help"),
    CommandSpec("ping", "", "show chat and user IDs, reply mode, and default model", "Show chat status"),
    CommandSpec("models", " [alias]", "list model aliases or show one model's config", "List model aliases"),
    CommandSpec("r", "", "show raw text from a stored assistant or user message", "Show raw message text"),
    CommandSpec("c", " <alias> <content>", "send one message with a specific model", "Send with a specific model"),
    CommandSpec("n", " [content]", "start a fresh conversation", "Start a fresh conversation"),
    CommandSpec("s", " <prompt>", "set a system prompt override for a conversation", "Set a system prompt override"),
)

CHAT_SETTING_COMMANDS = (
    CommandSpec("model", " <alias>", "set the default model for this chat", "Set the default model"),
    CommandSpec("mode", " auto|mention|off", "set how plain messages are handled", "Set reply mode"),
    CommandSpec("timeout", " <duration>", "set the plain-message conversation timeout", "Set conversation timeout"),
)

OWNER_COMMANDS = (
    CommandSpec("togglechat", " [chat_id]", "add or remove a chat from the whitelist", "Toggle chat whitelist"),
    CommandSpec("toggleuser", " [user_id]", "add or remove a user from the whitelist", "Toggle user whitelist"),
    CommandSpec("whitelist", "", "show the current whitelist", "Show the whitelist"),
)

ALL_COMMANDS = CORE_COMMANDS + CHAT_SETTING_COMMANDS + OWNER_COMMANDS


def telegram_bot_commands() -> list[dict[str, str]]:
    return [
        {
            "command": spec.command,
            "description": spec.telegram_description,
        }
        for spec in ALL_COMMANDS
    ]
