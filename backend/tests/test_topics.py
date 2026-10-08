import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from markov.app import create_app
from markov.domain.model import MarkovChain
from markov.domain.tokenizers.regex import RegexTokenizer


class TopicTests(unittest.TestCase):
    def test_repeated_topic_uses_cache_and_keeps_general_transitions(self):
        model = MarkovChain(RegexTokenizer(), max_length=10, context_size=3)
        model.fit(["кот спит.", "пёс бежит."])
        original = model.transitions
        with patch.object(
            model, "_count_transitions", wraps=model._count_transitions
        ) as count:
            model.generate(topic="КОТ")
            model.generate(topic=" кот ")
            model.generate()
            self.assertEqual(count.call_count, 1)
        self.assertEqual(model.transitions, original)
        self.assertEqual(model.cached_topics, ("кот",))

    def test_corpus_changes_invalidate_topics(self):
        model = MarkovChain(RegexTokenizer(), max_length=10, context_size=2)
        model.fit(["кот спит."])
        model.generate(topic="кот")
        model.update(["кот бежит."])
        self.assertEqual(model.cached_topics, ())
        with patch.object(
            model, "_count_transitions", wraps=model._count_transitions
        ) as count:
            model.generate(topic="кот")
            self.assertEqual(count.call_args.args[0], ["кот спит.", "кот бежит."])
        model.fit(["пёс бежит."])
        with self.assertRaises(ValueError):
            model.generate(topic="кот")

    def test_ngram_change_prepares_known_topics(self):
        app = create_app(lambda: ["кот спит."] * 5 + ["пёс бежит."] * 5)
        with TestClient(app) as client:
            self.assertEqual(
                client.post("/api/v1/generate", json={"topic": "кот"}).status_code, 200
            )
            settings = client.get("/api/v1/model").json()["settings"]
            settings["n_gramm"] = 4
            self.assertEqual(
                client.post("/api/v1/settings", json=settings).status_code, 200
            )
            model = app.state.service.model
            with patch.object(
                model,
                "_count_transitions",
                side_effect=AssertionError("unexpected rebuild"),
            ):
                self.assertEqual(
                    client.post("/api/v1/generate", json={"topic": "кот"}).status_code,
                    200,
                )
                self.assertEqual(
                    client.post("/api/v1/generate", json={}).status_code, 200
                )
            response = client.post(
                "/api/v1/generate", json={"topic": "неизвестная тема"}
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("No training texts found", response.json()["detail"])

    def test_topic_cache_is_bounded(self):
        model = MarkovChain(RegexTokenizer(), max_length=2)
        model.fit(["кот пёс мышь."])
        for topic in ("кот", "пёс", "мышь"):
            model.generate(topic=topic)
        self.assertEqual(model.cached_topics, ("пёс", "мышь"))
