import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import create_app
from markov.service import MarkovService
from markov.tokenizer import CharacterTokenizer, RegexTokenizer


class TokenizerSettingsTests(unittest.TestCase):
    def test_switching_tokenizer_rebuilds_same_corpus_and_topics(self):
        app = create_app(lambda: ["кот спит.", "пёс бежит."] * 5)
        with TestClient(app) as client:
            settings = client.get("/api/v1/model").json()["settings"]
            self.assertEqual(settings["tokenizer"], "character")
            self.assertIsInstance(app.state.service.tokenizer, CharacterTokenizer)
            client.post("/api/v1/generate", json={"topic": "кот"})
            settings.update(tokenizer="regex", min_frequency=1, max_length=1)
            response = client.post("/api/v1/settings/stream", json=settings)
            self.assertIn("event: complete", response.text)
            self.assertIsInstance(app.state.service.tokenizer, RegexTokenizer)
            self.assertIn("кот", app.state.service.tokenizer.piece_to_token)
            self.assertEqual(client.get("/api/v1/model").json()["texts_count"], 10)
            self.assertEqual(client.get("/api/v1/model").json()["settings"], settings)
            model = app.state.service.model
            with patch.object(
                model,
                "_count_transitions",
                side_effect=AssertionError("unexpected rebuild"),
            ):
                self.assertEqual(
                    client.post("/api/v1/generate", json={"topic": "кот"}).json()[
                        "text"
                    ],
                    "кот",
                )
                self.assertEqual(
                    client.post("/api/v1/settings", json=settings).status_code, 200
                )
            settings["tokenizer"] = "character"
            self.assertEqual(
                client.post("/api/v1/settings", json=settings).status_code, 200
            )
            self.assertIsInstance(app.state.service.tokenizer, CharacterTokenizer)
            self.assertIn("к", app.state.service.tokenizer.piece_to_token)
            self.assertNotIn("кот", app.state.service.tokenizer.piece_to_token)
            self.assertEqual(
                client.post("/api/v1/generate", json={"topic": "кот"}).json()["text"],
                "к",
            )
            settings["tokenizer"] = "unknown"
            current = app.state.service.model
            self.assertEqual(
                client.post("/api/v1/settings", json=settings).status_code, 422
            )
            self.assertIs(app.state.service.model, current)

    def test_startup_tokenizer_and_frequency_are_applied(self):
        for kind, tokenizer_class in (
            ("regex", RegexTokenizer),
            ("character", CharacterTokenizer),
        ):
            service = MarkovService(tokenizer_type=kind, min_frequency=2)
            service.train(["кот", "кот", "пёс"], replace=True)
            self.assertIsInstance(service.tokenizer, tokenizer_class)
            self.assertEqual(service.stats().settings.tokenizer, kind)
            self.assertNotIn(
                "пёс" if kind == "regex" else "ё", service.tokenizer.piece_to_token
            )
