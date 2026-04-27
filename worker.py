from __future__ import annotations

from workers import WorkerEntrypoint, Response

from app.config import build_config, build_registry
from app.chat_service import ChatService
from app.webhook_app import WebhookHandler
from app.d1_storage import D1Storage


class Default(WorkerEntrypoint):
    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._handler: WebhookHandler | None = None
        self._service: ChatService | None = None
        self._storage: D1Storage | None = None

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        from workers import env

        config = build_config()
        await D1Storage.init_schema(env.DB)
        self._storage = D1Storage(
            env.DB,
            default_model_alias=config.default_model_alias,
            default_reply_mode=config.default_reply_mode,
            default_skip_prefix=config.skip_prefix,
        )
        registry = build_registry(config)
        self._service = ChatService(
            storage=self._storage,
            registry=registry,
            system_prompt=config.system_prompt,
            owner_user_ids=config.owner_user_ids,
            conversation_timeout_seconds=config.conversation_timeout_seconds,
            render_limit=config.telegram_message_limit,
            render_edit_interval_seconds=config.render_edit_interval_seconds,
            safety_identifier_salt=config.safety_identifier_salt,
        )
        self._handler = WebhookHandler(
            service=self._service,
            storage=self._storage,
            token=config.telegram_bot_token,
        )
        self._initialized = True

    async def fetch(self, request: object) -> Response:
        await self._ensure_init()
        assert self._handler is not None

        body = await request.json()  # type: ignore[union-attr]
        await self._handler.handle_update(body)

        return Response.new("ok", status=200)
