# Switchboard Overview

Switchboard is a Telegram bot for personal use and small-group chats. It keeps conversation history locally, supports multiple LLM providers, and is built to stay simple to operate without turning every interaction into a command-heavy workflow.

Use [README.md](../README.md) for setup. Use this document to understand how the bot behaves once it is running.

## What Switchboard Does

- replies in allowlisted chats or for allowlisted users
- stores history in SQLite with local attachment storage
- supports OpenAI, Gemini, and Claude providers
- handles text messages plus Telegram image inputs and albums
- keeps model selection and conversation control in Telegram commands

## How Replies Are Gated

A fresh deployment starts quiet. Normal chat messages do not trigger model replies until an owner explicitly allowlists a chat or user.

Switchboard checks normal chat input in this order:

1. recognized commands are handled first
2. non-command messages must match the chat allowlist or user allowlist
3. the chat's reply mode decides whether the bot should answer automatically

Available reply modes:

- `auto`: answer eligible messages by default
- `mention`: answer only when mentioned or when replying to the bot
- `off`: ignore normal messages, but still accept commands

For one-off opt-out, prefix a message with `//` and the bot ignores it.

## Commands

Core commands:

- `/help`
- `/help <new|c|s>`
- `/ping`
- `/models`
- `/model <alias>`
- `/c <alias> <message>`
- `/new <message>`
- `/s <prompt>`
- `/mode auto|mention|off`
- `/togglechat [chat_id]`
- `/toggleuser [user_id]`
- `/whitelist`

The basic model is:

- a plain message uses the chat's default model
- `/model` changes that default for the chat
- `/c` chooses a model for one turn or branch
- `/new` starts a fresh conversation
- `/s` creates or reframes a branch with a system-prompt override

## How Conversations Work

Switchboard tries to make branching explicit without making everyday chatting awkward.

Plain messages:

- continue the latest recent conversation for the same `(chat_id, user_id)` when it is still within the configured timeout
- otherwise start a new conversation with the chat defaults

Replies:

- replying to a stored bot state attaches directly to that point
- replying to an older assistant message forks from that exact point
- later sibling turns are not pulled into the new branch

Command behavior:

- `/new` starts fresh unless it is used on a seed created by `/s`
- `/c` switches model for the selected branch or starts a new branch with that model
- `/s` can create a reusable seed or reframe an existing branch with a different system prompt

Streaming behavior:

- if a branch is already streaming, later messages for that same branch are queued
- queued messages are merged into the next follow-up turn after the current reply completes
- different branches can still run independently

## Storage And Runtime

Switchboard runs as a single service. The default deployment shape is one bot process plus local storage.

Important runtime details:

- chat and conversation state live in SQLite
- attachment blobs are stored on disk
- in Docker, persistent data is mounted under `/data`
- the model catalog defaults to `${BOT_DATA_DIR}/models.toml`

This keeps the bot easy to back up, inspect, and run on a single machine.

## Current Limits

- no image generation
- no non-image attachment ingestion yet
- polling-based Telegram runtime

## When To Read The Code

Most operators should not need internal implementation details. If you do want the code-level layout, start with:

- [app/main.py](../app/main.py)
- [app/router.py](../app/router.py)
- [app/chat_service.py](../app/chat_service.py)
- [app/storage.py](../app/storage.py)
