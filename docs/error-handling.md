# Error Handling And Crash Recovery

Switchboard treats SQLite as the source of truth after a restart. Telegram messages are a best-effort projection of the logical conversation state stored locally.

The recovery goal is not to perfectly resume every interrupted operation. Instead, startup recovery moves durable objects back to the nearest normal pipeline state so the bot can continue operating predictably.

## Recovery Principles

- Recovery is cleanup and projection, not workflow resumption.
- The bot does not call LLM providers during startup recovery.
- Assistant messages that were still `streaming` are marked interrupted and failed.
- If a linked Telegram bot message exists, Switchboard edits it to match the recovered DB content.
- If no linked Telegram bot message exists but the DB has non-empty assistant content, Switchboard sends a replacement assistant message and links that new Telegram message.
- If Telegram had received a bot message but Switchboard crashed before storing its Telegram message ID, that Telegram message may be orphaned. A duplicated replacement message is acceptable.
- Claimed inbox updates are normalized coarsely: updates that already created local side effects are completed, while updates without realized local messages are requeued.

## Unknown Bot Reply Targets

If a user replies to a bot-authored Telegram message that Switchboard cannot resolve to local state, the bot sends an explanatory reply instead of silently ignoring or misattaching the message.

This can happen when the process crashes after Telegram accepts a bot message but before Switchboard stores the Telegram message ID in SQLite. Since the bot cannot reliably rediscover that ID from Telegram later, the user should resend their message or reply to a different stored bot response.

## Accepted Tradeoffs

The simplified model intentionally accepts rare imperfections after a crash:

- a user message may need to be resent;
- a replacement assistant message may duplicate an orphaned Telegram message;
- a partially streamed assistant answer is finalized as interrupted rather than resumed;
- final Telegram formatting is not reconstructed from special render checkpoints.

These tradeoffs keep recovery easy to reason about and avoid a complex durable workflow engine for infrequent server failures.
