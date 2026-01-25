from __future__ import annotations

from dataclasses import dataclass

try:
    from telegram_markdown import parse_markdown_to_entities
except ImportError:  # pragma: no cover - dependency is required in normal runtime
    parse_markdown_to_entities = None


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


@dataclass(frozen=True)
class MessageEntity:
    type: str
    offset: int
    length: int
    url: str | None = None
    language: str | None = None
    custom_emoji_id: str | None = None

    def shifted(self, delta: int) -> "MessageEntity":
        return MessageEntity(
            type=self.type,
            offset=self.offset + delta,
            length=self.length,
            url=self.url,
            language=self.language,
            custom_emoji_id=self.custom_emoji_id,
        )

    def with_range(self, *, offset: int, length: int) -> "MessageEntity":
        return MessageEntity(
            type=self.type,
            offset=offset,
            length=length,
            url=self.url,
            language=self.language,
            custom_emoji_id=self.custom_emoji_id,
        )

    def to_api_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "offset": self.offset,
            "length": self.length,
        }
        if self.url is not None:
            result["url"] = self.url
        if self.language is not None:
            result["language"] = self.language
        if self.custom_emoji_id is not None:
            result["custom_emoji_id"] = self.custom_emoji_id
        return result


class RichText:
    def __init__(self, text: str = "", entities: list[MessageEntity] | None = None) -> None:
        self.text = text
        self.entities = entities or []

    @classmethod
    def coerce(cls, value: str | "RichText") -> "RichText":
        if isinstance(value, RichText):
            return value
        return cls(value)

    @classmethod
    def from_markdown(cls, markdown: str) -> "RichText":
        if parse_markdown_to_entities is None:
            return cls(markdown)
        text, entities = parse_markdown_to_entities(markdown)
        return cls(
            text,
            [
                MessageEntity(
                    type=entity["type"],
                    offset=int(entity["offset"]),
                    length=int(entity["length"]),
                    url=entity.get("url"),
                    language=entity.get("language"),
                    custom_emoji_id=entity.get("custom_emoji_id"),
                )
                for entity in entities
            ],
        )

    @classmethod
    def quote(cls, text: str, *, collapsed: bool = False) -> "RichText":
        entity_offset, entity_length = _quote_bounds(text)
        if entity_length == 0:
            return cls(text)
        entity_type = "expandable_blockquote" if collapsed else "blockquote"
        return cls(
            text,
            [MessageEntity(type=entity_type, offset=entity_offset, length=entity_length)],
        )

    def __len__(self) -> int:
        return len(self.text)

    def __bool__(self) -> bool:
        return bool(self.text)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RichText):
            return False
        return self.text == other.text and self.entities == other.entities

    def __add__(self, other: str | "RichText") -> "RichText":
        other_text = RichText.coerce(other)
        if not self.text:
            return other_text
        if not other_text.text:
            return self
        offset = utf16_len(self.text)
        return RichText(
            self.text + other_text.text,
            self.entities + [entity.shifted(offset) for entity in other_text.entities],
        )

    def __radd__(self, other: str | "RichText") -> "RichText":
        return RichText.coerce(other) + self

    def __getitem__(self, key: slice) -> "RichText":
        if not isinstance(key, slice):
            raise TypeError("RichText only supports slicing")
        start, stop, step = key.indices(len(self.text))
        if step != 1:
            raise ValueError("RichText slicing does not support steps")
        if start >= stop:
            return RichText()

        utf16_start = utf16_len(self.text[:start])
        utf16_stop = utf16_len(self.text[:stop])
        entities: list[MessageEntity] = []
        for entity in self.entities:
            entity_start = entity.offset
            entity_stop = entity.offset + entity.length
            overlap_start = max(entity_start, utf16_start)
            overlap_stop = min(entity_stop, utf16_stop)
            if overlap_start >= overlap_stop:
                continue
            entities.append(
                entity.with_range(
                    offset=overlap_start - utf16_start,
                    length=overlap_stop - overlap_start,
                )
            )
        return RichText(self.text[start:stop], entities)

    def to_telegram(self) -> tuple[str, list[dict[str, object]]]:
        return self.text, [entity.to_api_dict() for entity in self.entities]


def _quote_bounds(text: str) -> tuple[int, int]:
    leading_chars = len(text) - len(text.lstrip())
    stripped = text.strip()
    if not stripped:
        return 0, 0
    return utf16_len(text[:leading_chars]), utf16_len(stripped)
