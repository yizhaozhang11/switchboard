# Switchboard

Switchboard is a simple Telegram bot for personal use and small-group chats. It keeps conversation history in SQLite, supports multiple LLM providers, and stays focused on a single-service deployment that is easy to run locally.

Requires Python 3.12 or newer.

Use [docs/overview.md](./docs/overview.md) for commands, reply behavior, conversation branching, and runtime behavior.

## Quick Start

Using conda:

```bash
cd switchboard
conda env create -f environment.yml
conda activate switchboard
cp .env.example .env
```

Edit `.env` before starting the bot. At minimum, set `TELEGRAM_BOT_TOKEN` and one provider key: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`. The copied example is a template with blank secrets, so running immediately after `cp .env.example .env` is expected to fail.

Then run:

```bash
python -m app.main
```

Using `venv`:

```bash
cd switchboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` before starting the bot. At minimum, set `TELEGRAM_BOT_TOKEN` and one provider key: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`. The copied example is a template with blank secrets, so running immediately after `cp .env.example .env` is expected to fail.

Then run:

```bash
python -m app.main
```

Environment is read from `.env` when present, then from the process environment.

## First Run

The copied `.env.example` intentionally leaves secrets blank. Switchboard does not ship fake defaults because a real bot token and at least one real provider key are required.

If you start without editing `.env`, expect startup to stop before Telegram polling:

- missing `TELEGRAM_BOT_TOKEN` prints `Startup error: TELEGRAM_BOT_TOKEN is required`
- setting the bot token but no provider key prints `Startup error: No providers are configured`

After configuration is valid and the bot starts, a fresh deployment still starts quiet. Normal chat messages do not trigger model replies until an owner allowlists a chat or user. This prevents a newly invited bot from answering every eligible message before you opt a chat in.

Bootstrap flow:

1. Set `BOT_OWNER_USER_IDS` in `.env` to your numeric Telegram user ID. Usernames are not accepted.
2. Start the bot.
3. In the Telegram chat where you want the bot active, send `/togglechat` as that owner to allowlist the current chat.
4. Use `/toggleuser [user_id]` instead if you want to allow one user across chats.
5. Use `/whitelist` to inspect the current allowlist.

Commands such as `/help` can respond before a chat is allowlisted, but allowlist management commands require a configured owner. If `BOT_OWNER_USER_IDS` is unset, `/togglechat`, `/toggleuser`, and `/whitelist` are disabled and the bot will continue ignoring normal messages in a fresh deployment.

## Configuration

Required environment:

- `TELEGRAM_BOT_TOKEN`
- at least one provider key: `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`

Optional but useful:

- `BOT_OWNER_USER_IDS`
- `BOT_DEFAULT_MODEL_ALIAS`
- `BOT_MODEL_CONFIG_PATH`
- `BOT_DEFAULT_REPLY_MODE`
- `BOT_SKIP_PREFIX`
- `BOT_CONVERSATION_TIMEOUT_SECONDS`
- `BOT_SYSTEM_PROMPT`
- `BOT_DATA_DIR`
- `SAFETY_IDENTIFIER_SALT`

Notes:

- set `BOT_OWNER_USER_IDS` before first run if you want to manage the allowlist from Telegram commands
- if `BOT_DEFAULT_MODEL_ALIAS` is unset, Switchboard picks the first alias from the first configured provider
- if you set `BOT_DEFAULT_MODEL_ALIAS`, it must match one of the configured model aliases
- `SAFETY_IDENTIFIER_SALT`, when set, must be at least 16 characters long
- the model catalog defaults to `${BOT_DATA_DIR}/models.toml`
- reasoning-capable OpenAI and Gemini entries can set `reasoning_effort` in the model catalog

## Running

Run the bot directly:

```bash
cd switchboard
python -m app.main
```

This direct-run mode is best for local development and manual operation. It does not include a process supervisor or restart policy, so transient Telegram or network failures can stop the process.

## Tests

```bash
cd switchboard
python -m unittest discover -s tests
```

## Docker

The container stores persistent state under `/data`. The provided Compose file mounts that path to `./data` on the host so conversation history and the model catalog survive restarts.
For a long-running deployment, prefer Docker Compose, which includes `restart: unless-stopped`.

```bash
cd switchboard
cp .env.example .env
```

Edit `.env` with `TELEGRAM_BOT_TOKEN`, at least one provider key, and `BOT_OWNER_USER_IDS` for first-run allowlist management.

Then start:

```bash
docker compose up --build -d
```

Useful commands:

```bash
docker compose logs -f
docker compose down
```

## License

Switchboard is distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Switchboard draws inspiration from [zzh1996/chatgpt-telegram-bot](https://github.com/zzh1996/chatgpt-telegram-bot).
