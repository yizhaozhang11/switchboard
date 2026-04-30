from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
import re

import httpx

from app.richtext import RichText


class TelegramAPIError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TelegramBotAPI:
    def __init__(
        self,
        token: str,
        *,
        request_timeout_seconds: int = 40,
        max_rate_limit_retries: int = 1,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"
        self._client = httpx.AsyncClient(timeout=request_timeout_seconds)
        self.max_rate_limit_retries = max_rate_limit_retries
        self._sleep = sleep or asyncio.sleep

    async def __aenter__(self) -> "TelegramBotAPI":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _sanitized_transport_error(*, method: str, exc: Exception) -> TelegramAPIError:
        return TelegramAPIError(f"Telegram API call failed: {method} ({type(exc).__name__})")

    @staticmethod
    def _retry_after_from_error_data(data: dict) -> float | None:
        parameters = data.get("parameters")
        if isinstance(parameters, dict):
            retry_after = parameters.get("retry_after")
            if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
                return max(0.0, float(retry_after))

        description = data.get("description")
        if isinstance(description, str):
            match = re.search(r"\bretry after (\d+(?:\.\d+)?)\b", description, flags=re.IGNORECASE)
            if match is not None:
                return float(match.group(1))
        return None

    async def request(self, method: str, payload: dict | None = None) -> dict | list:
        attempts = 0
        while True:
            try:
                response = await self._client.post(f"{self.base_url}/{method}", json=payload or {})
            except httpx.HTTPError as exc:
                raise self._sanitized_transport_error(method=method, exc=exc) from exc
            data: object | None
            try:
                data = response.json()
            except ValueError:
                data = None

            if isinstance(data, dict) and not data.get("ok", True):
                description = str(data.get("description", f"Telegram API call failed: {method}"))
                error = TelegramAPIError(
                    description,
                    retry_after=self._retry_after_from_error_data(data),
                )
                if error.retry_after is not None and attempts < self.max_rate_limit_retries:
                    attempts += 1
                    logging.warning(
                        "Telegram API rate limited %s; retrying after %.1f seconds",
                        method,
                        error.retry_after,
                    )
                    await self._sleep(error.retry_after)
                    continue
                raise error

            if response.is_error:
                detail = ""
                if hasattr(response, "text"):
                    detail = response.text.strip()
                if detail:
                    raise TelegramAPIError(f"Telegram API call failed: {method} ({response.status_code}): {detail}")
                raise TelegramAPIError(f"Telegram API call failed: {method} ({response.status_code})")

            if not isinstance(data, dict) or "result" not in data:
                raise TelegramAPIError(f"Telegram API call failed: {method}")
            return data["result"]

    async def get_me(self) -> dict:
        return await self.request("getMe")

    async def get_updates(self, *, offset: int | None, timeout: int) -> list[dict]:
        payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        result = await self.request("getUpdates", payload)
        assert isinstance(result, list)
        return result

    async def send_message(
        self,
        chat_id: int,
        text: str | RichText,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        rendered_text, entities = RichText.coerce(text).to_telegram()
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": rendered_text,
            "disable_web_page_preview": True,
        }
        if entities:
            payload["entities"] = entities
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_message_id}
        result = await self.request("sendMessage", payload)
        return int(result["message_id"])

    async def edit_message_text(self, chat_id: int, message_id: int, text: str | RichText) -> None:
        rendered_text, entities = RichText.coerce(text).to_telegram()
        try:
            payload: dict[str, object] = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": rendered_text,
                "disable_web_page_preview": True,
            }
            if entities:
                payload["entities"] = entities
            await self.request(
                "editMessageText",
                payload,
            )
        except TelegramAPIError as exc:
            if "message is not modified" in str(exc).lower():
                return
            raise

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        try:
            await self.request("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
        except TelegramAPIError:
            return

    async def get_file(self, file_id: str) -> dict:
        result = await self.request("getFile", {"file_id": file_id})
        assert isinstance(result, dict)
        return result

    async def download_file_bytes(self, file_id: str) -> bytes:
        file_info = await self.get_file(file_id)
        file_path = file_info.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise TelegramAPIError(f"Telegram API did not return file_path for file_id={file_id}")
        try:
            response = await self._client.get(f"{self.file_base_url}/{file_path}")
        except httpx.HTTPError as exc:
            raise self._sanitized_transport_error(method="getFileContent", exc=exc) from exc
        if response.is_error:
            raise TelegramAPIError(f"Telegram file download failed ({response.status_code})")
        return response.content
