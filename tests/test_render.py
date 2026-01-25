from __future__ import annotations

import sys
import types
import unittest

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

from app.render import render_reply_text, split_rich_text
from app.richtext import RichText


class RenderTests(unittest.TestCase):
    def test_render_reply_text_includes_reasoning_blocks(self) -> None:
        rendered = render_reply_text(
            "Answer",
            [
                "First line\nSecond line",
                "Third line",
            ],
            final=True,
        )
        self.assertEqual(rendered.to_telegram(), (
            "First line\nSecond line\nThird line\nAnswer",
            [
                {"type": "expandable_blockquote", "offset": 0, "length": 22},
                {"type": "expandable_blockquote", "offset": 23, "length": 10},
            ],
        ))

    def test_render_reply_text_omits_empty_reasoning_blocks(self) -> None:
        rendered = render_reply_text("Answer", ["", "Used path"], final=False)
        self.assertEqual(
            rendered.to_telegram(),
            (
                "Used path\nAnswer",
                [{"type": "blockquote", "offset": 0, "length": 9}],
            ),
        )

    def test_render_reply_text_formats_markdown_body(self) -> None:
        rendered = render_reply_text("**bold** and `code`", [], final=True)
        self.assertEqual(
            rendered.to_telegram(),
            (
                "bold and code",
                [
                    {"type": "bold", "offset": 0, "length": 4},
                    {"type": "code", "offset": 9, "length": 4},
                ],
            ),
        )

    def test_render_reply_text_leaves_streaming_markdown_unparsed(self) -> None:
        rendered = render_reply_text("**bold** and `code", [], final=False)
        self.assertEqual(
            rendered.to_telegram(),
            ("**bold** and `code", []),
        )

    def test_split_rich_text_preserves_quote_entity_ranges(self) -> None:
        rich = RichText.quote("One two three four five", collapsed=False)
        parts = split_rich_text(rich, 10)
        self.assertEqual(
            [part.to_telegram() for part in parts],
            [
                ("One two", [{"type": "blockquote", "offset": 0, "length": 7}]),
                ("three", [{"type": "blockquote", "offset": 0, "length": 5}]),
                ("four five", [{"type": "blockquote", "offset": 0, "length": 9}]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
