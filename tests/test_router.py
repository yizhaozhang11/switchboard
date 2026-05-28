from __future__ import annotations

import unittest

from app.commands import ALL_COMMANDS
from app.router import Router
from app.types import ChatAction, ChatSettings, CommandAction, IgnoreAction, ContentPart, ImageRef, IncomingMessage


def make_message(**overrides) -> IncomingMessage:
    payload = {
        "update_id": 1,
        "chat_id": 100,
        "message_id": 10,
        "user_id": 200,
        "chat_type": "group",
        "text": "hello",
        "from_bot": False,
        "mentions_bot": False,
        "reply_to_message_id": None,
        "reply_to_user_id": None,
        "reply_to_bot": False,
        "reply_to_text": None,
    }
    payload.update(overrides)
    return IncomingMessage(**payload)


def make_settings(**overrides) -> ChatSettings:
    payload = {
        "chat_id": 100,
        "enabled": True,
        "reply_mode": "auto",
        "default_model_alias": "o",
        "skip_prefix": "//",
    }
    payload.update(overrides)
    return ChatSettings(**payload)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = Router(bot_username="MyBot")

    def test_auto_mode_routes_plain_messages(self) -> None:
        action = self.router.route(make_message(text="hello there"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "hello there")

    def test_skip_prefix_requires_outer_policy_layer(self) -> None:
        action = self.router.route(make_message(text="// do not answer"), make_settings())
        self.assertNotIsInstance(action, IgnoreAction)
        self.assertIsInstance(action, CommandAction)

    def test_reply_mode_policy_is_not_router_policy(self) -> None:
        action = self.router.route(make_message(text="hello"), make_settings(reply_mode="mention"))
        self.assertIsInstance(action, ChatAction)

    def test_mentions_still_route_as_plain_messages(self) -> None:
        action = self.router.route(
            make_message(text="@MyBot hello", mentions_bot=True),
            make_settings(reply_mode="mention"),
        )
        self.assertIsInstance(action, ChatAction)

    def test_parse_c_command_multiline(self) -> None:
        action = self.router.route(make_message(text="/c o\nhello world"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.model_alias, "o")
        self.assertEqual(action.content, "hello world")
        self.assertEqual(action.intent, "choose_model")

    def test_parse_model_command(self) -> None:
        action = self.router.route(make_message(text="/model om"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "model")
        self.assertEqual(action.argument, "om")

    def test_parse_models_command(self) -> None:
        action = self.router.route(make_message(text="/models"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "models")

    def test_parse_mode_command(self) -> None:
        action = self.router.route(make_message(text="/mode mention"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "mode")
        self.assertEqual(action.argument, "mention")

    def test_parse_timeout_command(self) -> None:
        action = self.router.route(make_message(text="/timeout 5 minutes"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "timeout")
        self.assertEqual(action.argument, "5 minutes")

    def test_timeout_command_requires_argument(self) -> None:
        action = self.router.route(make_message(text="/timeout"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "usage_error")
        self.assertEqual(action.content, "Usage: /timeout <seconds|5m|1h>")

    def test_parse_ping_command(self) -> None:
        action = self.router.route(make_message(text="/ping"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "ping")

    def test_parse_help_command(self) -> None:
        action = self.router.route(make_message(text="/help"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "help")
        self.assertIsNone(action.argument)

    def test_parse_help_command_with_topic(self) -> None:
        action = self.router.route(make_message(text="/help n"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "help")
        self.assertEqual(action.argument, "n")

    def test_registered_command_menu_matches_router_commands(self) -> None:
        registered_commands = [spec.command for spec in ALL_COMMANDS]
        self.assertIn("n", registered_commands)
        self.assertNotIn("new", registered_commands)

        examples = {
            "c": "/c o hello",
            "model": "/model o",
            "mode": "/mode auto",
            "timeout": "/timeout 5m",
            "s": "/s prompt",
        }
        for spec in ALL_COMMANDS:
            with self.subTest(command=spec.command):
                action = self.router.route(
                    make_message(text=examples.get(spec.command, f"/{spec.command}")),
                    make_settings(),
                )
                if isinstance(action, CommandAction):
                    self.assertNotEqual(action.content, f"Unknown command: /{spec.command}")
                else:
                    self.assertIsInstance(action, ChatAction)

    def test_parse_togglechat_command(self) -> None:
        action = self.router.route(make_message(text="/togglechat"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "togglechat")
        self.assertIsNone(action.argument)

    def test_parse_toggleuser_command_with_argument(self) -> None:
        action = self.router.route(make_message(text="/toggleuser 321"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "toggleuser")
        self.assertEqual(action.argument, "321")

    def test_parse_whitelist_command(self) -> None:
        action = self.router.route(make_message(text="/whitelist"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "whitelist")

    def test_parse_raw_command(self) -> None:
        action = self.router.route(make_message(text="/r"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "raw")

    def test_raw_command_rejects_arguments(self) -> None:
        action = self.router.route(make_message(text="/r extra"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "usage_error")
        self.assertEqual(action.content, "Usage: /r")

    def test_raw_command_rejects_multiline_body(self) -> None:
        action = self.router.route(make_message(text="/r\nextra"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "usage_error")
        self.assertEqual(action.content, "Usage: /r")

    def test_parse_n_command_multiline(self) -> None:
        action = self.router.route(make_message(text="/n\nfresh start"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "fresh start")
        self.assertIsNone(action.model_alias)
        self.assertEqual(action.intent, "new")

    def test_n_command_without_content_routes_empty_trigger(self) -> None:
        action = self.router.route(make_message(text="/n"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "")
        self.assertEqual(action.intent, "new")

    def test_n_command_allows_image_only_message(self) -> None:
        action = self.router.route(
            make_message(
                text="/n",
                images=(ImageRef(file_id="photo-1", mime_type="image/jpeg"),),
            ),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "")
        self.assertEqual(action.intent, "new")
        self.assertEqual(len(action.images), 1)

    def test_plain_reply_to_human_routes_as_plain_message(self) -> None:
        action = self.router.route(
            make_message(text="follow-up", reply_to_message_id=9, reply_to_bot=False),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.intent, "plain")

    def test_conversation_command_replying_to_human_still_routes(self) -> None:
        action = self.router.route(
            make_message(text="/c o fresh", reply_to_message_id=9, reply_to_bot=False),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.intent, "choose_model")

    def test_n_replying_to_human_still_routes(self) -> None:
        action = self.router.route(
            make_message(text="/n fresh", reply_to_message_id=9, reply_to_bot=False),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "fresh")
        self.assertEqual(action.intent, "new")

    def test_new_command_still_routes_as_compatibility_alias(self) -> None:
        action = self.router.route(make_message(text="/new fresh"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "fresh")
        self.assertEqual(action.intent, "new")

    def test_parse_system_prompt_command(self) -> None:
        action = self.router.route(make_message(text="/s be concise"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.intent, "set_system_prompt")
        self.assertEqual(action.system_prompt, "be concise")
        self.assertEqual(action.content, "")

    def test_parse_multiline_system_prompt_command(self) -> None:
        action = self.router.route(make_message(text="/s line one\nline two"), make_settings())
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.intent, "set_system_prompt")
        self.assertEqual(action.system_prompt, "line one\nline two")

    def test_system_prompt_command_requires_content(self) -> None:
        action = self.router.route(make_message(text="/s"), make_settings())
        self.assertIsInstance(action, CommandAction)
        assert isinstance(action, CommandAction)
        self.assertEqual(action.name, "usage_error")

    def test_image_only_message_routes(self) -> None:
        action = self.router.route(
            make_message(images=(ImageRef(file_id="photo-1", mime_type="image/jpeg"),), text=""),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "")
        self.assertEqual(len(action.images), 1)

    def test_command_preserves_images(self) -> None:
        action = self.router.route(
            make_message(
                text="/c g describe this",
                images=(ImageRef(file_id="photo-1", mime_type="image/jpeg"),),
            ),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.model_alias, "g")
        self.assertEqual(len(action.images), 1)

    def test_c_command_preserves_album_part_order(self) -> None:
        image_one = ImageRef(file_id="photo-1", mime_type="image/jpeg")
        image_two = ImageRef(file_id="photo-2", mime_type="image/jpeg")
        action = self.router.route(
            make_message(
                text="/c g caption one\n\ncaption two",
                images=(image_one, image_two),
                parts=(
                    ContentPart(kind="text", text="/c g caption one"),
                    ContentPart(kind="image", image=image_one),
                    ContentPart(kind="text", text="caption two"),
                    ContentPart(kind="image", image=image_two),
                ),
            ),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "caption one\n\ncaption two")
        self.assertEqual([part.kind for part in action.parts], ["text", "image", "text", "image"])
        self.assertEqual([part.text for part in action.parts if part.kind == "text"], ["caption one", "caption two"])
        image_parts = [part for part in action.parts if part.kind == "image"]
        self.assertEqual(image_parts[0].image, image_one)
        self.assertEqual(image_parts[1].image, image_two)

    def test_c_command_allows_image_only_message(self) -> None:
        action = self.router.route(
            make_message(
                text="/c g",
                images=(ImageRef(file_id="photo-1", mime_type="image/jpeg"),),
            ),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.model_alias, "g")
        self.assertEqual(action.content, "")
        self.assertEqual(len(action.images), 1)

    def test_n_command_preserves_album_part_order(self) -> None:
        image_one = ImageRef(file_id="photo-1", mime_type="image/jpeg")
        image_two = ImageRef(file_id="photo-2", mime_type="image/jpeg")
        action = self.router.route(
            make_message(
                text="/n caption one\n\ncaption two",
                images=(image_one, image_two),
                parts=(
                    ContentPart(kind="text", text="/n caption one"),
                    ContentPart(kind="image", image=image_one),
                    ContentPart(kind="text", text="caption two"),
                    ContentPart(kind="image", image=image_two),
                ),
            ),
            make_settings(),
        )
        self.assertIsInstance(action, ChatAction)
        assert isinstance(action, ChatAction)
        self.assertEqual(action.content, "caption one\n\ncaption two")
        self.assertEqual([part.kind for part in action.parts], ["text", "image", "text", "image"])
        self.assertEqual([part.text for part in action.parts if part.kind == "text"], ["caption one", "caption two"])
        image_parts = [part for part in action.parts if part.kind == "image"]
        self.assertEqual(image_parts[0].image, image_one)
        self.assertEqual(image_parts[1].image, image_two)

    def test_wrong_targeted_command_is_ignored_as_command(self) -> None:
        action = self.router.route(make_message(text="/models@OtherBot"), make_settings())
        self.assertIsInstance(action, ChatAction)


if __name__ == "__main__":
    unittest.main()
