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

import httpx

from app.telegram_api import TelegramAPIError, TelegramBotAPI


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: object | None = None,
        text: str = "",
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.content = content

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self) -> object:
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class FakeClient:
    def __init__(
        self,
        responses: list[FakeResponse | Exception],
        *,
        get_responses: list[FakeResponse | Exception] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.get_responses = list(get_responses or [])
        self.calls: list[tuple[str, dict]] = []
        self.get_calls: list[str] = []

    async def post(self, url: str, json: dict) -> FakeResponse:
        self.calls.append((url, json))
        if not self.responses:
            raise AssertionError("No fake responses remaining")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def get(self, url: str) -> FakeResponse:
        self.get_calls.append(url)
        if not self.get_responses:
            raise AssertionError("No fake GET responses remaining")
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        return None


class TelegramBotAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_request_uses_telegram_error_description_from_http_400(self) -> None:
        api = TelegramBotAPI("test-token")
        api._client = FakeClient(
            [
                FakeResponse(
                    status_code=400,
                    json_data={
                        "ok": False,
                        "error_code": 400,
                        "description": "Bad Request: message is not modified",
                    },
                )
            ]
        )

        with self.assertRaises(TelegramAPIError) as ctx:
            await api.request("editMessageText", {"chat_id": 1, "message_id": 2, "text": "same"})

        self.assertEqual(str(ctx.exception), "Bad Request: message is not modified")

    async def test_edit_message_text_ignores_message_not_modified_http_400(self) -> None:
        api = TelegramBotAPI("test-token")
        api._client = FakeClient(
            [
                FakeResponse(
                    status_code=400,
                    json_data={
                        "ok": False,
                        "error_code": 400,
                        "description": "Bad Request: message is not modified",
                    },
                )
            ]
        )

        await api.edit_message_text(100, 200, "same text")

    async def test_delete_message_swallows_telegram_api_errors(self) -> None:
        api = TelegramBotAPI("test-token")
        api._client = FakeClient(
            [
                FakeResponse(
                    status_code=400,
                    json_data={
                        "ok": False,
                        "error_code": 400,
                        "description": "Bad Request: message to delete not found",
                    },
                )
            ]
        )

        await api.delete_message(100, 200)

    async def test_request_sanitizes_http_transport_errors(self) -> None:
        api = TelegramBotAPI("test-token")
        api._client = FakeClient([httpx.HTTPError("https://api.telegram.org/bottest-token/getMe failed")])

        with self.assertRaises(TelegramAPIError) as ctx:
            await api.request("getMe")

        self.assertEqual(str(ctx.exception), "Telegram API call failed: getMe (HTTPError)")
        self.assertNotIn("test-token", str(ctx.exception))

    async def test_download_file_bytes_sanitizes_http_error_status(self) -> None:
        api = TelegramBotAPI("test-token")
        api._client = FakeClient(
            [FakeResponse(json_data={"ok": True, "result": {"file_path": "documents/file.txt"}})],
            get_responses=[FakeResponse(status_code=404, text="not found")],
        )

        with self.assertRaises(TelegramAPIError) as ctx:
            await api.download_file_bytes("file-id")

        self.assertEqual(str(ctx.exception), "Telegram file download failed (404)")
        self.assertNotIn("test-token", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
