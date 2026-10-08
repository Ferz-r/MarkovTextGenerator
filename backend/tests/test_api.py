import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import create_app
from markov.api.schemas import TrainRequest


class APITest(unittest.TestCase):
    def test_lifecycle_generation_and_training(self):
        app = create_app(lambda: ["кот спит."] * 5)
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/v1/health").json(), {"status": "ok"})
            stats = client.get("/api/v1/model").json()
            self.assertEqual(stats["texts_count"], 5)
            response = client.post("/api/v1/generate", json={"prefix": "кот"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["text"].startswith("кот"))
            self.assertEqual(
                client.post("/api/v1/generate", json={"topic": "нет темы"}).status_code,
                422,
            )
            self.assertEqual(
                client.post("/api/v1/train", json={"texts": [" "]}).status_code, 422
            )
            updated = client.post(
                "/api/v1/train", json={"texts": ["пёс бежит."] * 5}
            ).json()
            self.assertEqual(updated["texts_count"], 10)
            replaced = client.post(
                "/api/v1/train", json={"texts": ["пёс бежит."] * 5, "replace": True}
            ).json()
            self.assertEqual(replaced["texts_count"], 5)
            self.assertEqual(
                client.post("/api/v1/generate", json={"topic": "кот"}).status_code, 422
            )
        self.assertIsNone(app.state.service)

    def test_empty_corpus_prevents_startup(self):
        with self.assertRaises(ValueError), TestClient(create_app(list)):
            pass

    def test_settings_rebuild_and_length(self):
        app = create_app(lambda: ["кот спит."] * 5)
        with TestClient(app) as client:
            settings = client.get("/api/v1/model").json()["settings"]
            original_model = app.state.service.model
            settings["max_length"] = 1
            response = client.post("/api/v1/settings", json=settings)
            self.assertEqual(response.status_code, 200)
            self.assertIs(app.state.service.model, original_model)
            self.assertEqual(
                client.post("/api/v1/generate", json={}).json()["text"], "к"
            )
            client.post("/api/v1/train", json={"texts": ["пёс бежит."] * 5})
            settings.update(n_gramm=2, min_frequency=6)
            response = client.post("/api/v1/settings", json=settings)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["settings"], settings)
            self.assertEqual(response.json()["texts_count"], 10)
            self.assertNotIn("к", app.state.service.tokenizer.piece_to_token)
            self.assertIsNot(app.state.service.model, original_model)
            settings["min_frequency"] = 1
            self.assertEqual(
                client.post("/api/v1/settings", json=settings).status_code, 200
            )
            self.assertIn("ё", app.state.service.tokenizer.piece_to_token)
            current_model = app.state.service.model
            for field, value in [
                ("n_gramm", 0),
                ("n_gramm", 51),
                ("min_frequency", 0),
                ("max_length", 10001),
                ("n_gramm", 2.5),
            ]:
                invalid = dict(settings, **{field: value})
                self.assertEqual(
                    client.post("/api/v1/settings", json=invalid).status_code, 422
                )
            self.assertIs(app.state.service.model, current_model)

    def test_streamed_training_progress(self):
        app = create_app(lambda: ["кот спит."] * 5)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/settings/stream",
                json={"n_gramm": 2, "min_frequency": 1, "max_length": 10},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                response.headers["content-type"].startswith("text/event-stream")
            )
            frames = [
                frame.splitlines() for frame in response.text.strip().split("\n\n")
            ]
            events = [(lines[0][7:], json.loads(lines[1][6:])) for lines in frames]
            progress = [payload for event, payload in events if event == "progress"]
            self.assertEqual(
                {p["stage"] for p in progress},
                {
                    "Подсчёт частот словаря",
                    "Построение словаря",
                    "Построение переходов",
                    "Индексирование: Построение переходов",
                },
            )
            self.assertTrue(any(p["percent"] == 0 for p in progress))
            self.assertEqual(progress[-1]["percent"], 100)
            self.assertEqual(events[-1][0], "complete")
            self.assertEqual(events[-1][1]["settings"]["n_gramm"], 2)
            response = client.post(
                "/api/v1/train/stream", json={"texts": ["пёс бежит."]}
            )
            self.assertIn('"texts_count": 6', response.text)
            self.assertIn("event: complete", response.text)
            with patch.object(
                app.state.service, "configure", side_effect=RuntimeError("internal")
            ):
                response = client.post(
                    "/api/v1/settings/stream",
                    json={"n_gramm": 3, "min_frequency": 1, "max_length": 10},
                )
            self.assertIn("event: error", response.text)
            self.assertNotIn("internal", response.text)
            self.assertNotIn("event: complete", response.text)
            self.assertEqual(
                client.post(
                    "/api/v1/settings/stream",
                    json={"n_gramm": 0, "min_frequency": 1, "max_length": 10},
                ).status_code,
                422,
            )

    def test_training_accepts_large_corpus(self):
        request = TrainRequest(texts=["а" * 100001])
        self.assertEqual(len(request.texts[0]), 100001)
        request = TrainRequest(texts=["текст"] * 10001)
        self.assertEqual(len(request.texts), 10001)

    def test_file_upload_training(self):
        app = create_app(lambda: ["кот спит."] * 5)
        with TestClient(app) as client:
            for encoding in ("utf-8", "cp1251", "utf-16"):
                response = client.post(
                    "/api/v1/train/file/stream",
                    files={
                        "file": (
                            "book.txt",
                            "Первая история.\r\n\r\nВторая история.".encode(encoding),
                            "text/plain",
                        )
                    },
                    data={
                        "replace": "true",
                        "texts": json.dumps(["дополнительный текст"]),
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("event: complete", response.text)
                self.assertIn('"texts_count": 3', response.text)
                self.assertIn("Чтение файла", response.text)
            response = client.post(
                "/api/v1/train/file/stream",
                files={"file": ("empty.txt", b"", "text/plain")},
            )
            self.assertIn("event: error", response.text)
            self.assertIn("Файл пустой", response.text)
            self.assertEqual(client.get("/api/v1/model").json()["texts_count"], 3)
            response = client.post(
                "/api/v1/train/file/stream",
                files={"file": ("bad.txt", b"abc\x00", "text/plain")},
            )
            self.assertIn("event: error", response.text)
            response = client.post(
                "/api/v1/train/file/stream",
                files={"file": ("bad.pdf", b"abc", "text/plain")},
            )
            self.assertEqual(response.status_code, 422)

    def test_single_newlines_are_independent_training_texts(self):
        app = create_app(lambda: ["исходный текст"])
        with TestClient(app) as client:
            settings = client.get("/api/v1/model").json()["settings"]
            settings["min_frequency"] = 1
            client.post("/api/v1/settings", json=settings)
            for uploaded in (False, True):
                if uploaded:
                    response = client.post(
                        "/api/v1/train/file/stream",
                        files={
                            "file": (
                                "words.txt",
                                "арбуз\r\nбанан\r\n\r\nвишня".encode(),
                                "text/plain",
                            )
                        },
                        data={"replace": "true"},
                    )
                    self.assertIn("event: complete", response.text)
                else:
                    response = client.post(
                        "/api/v1/train",
                        json={"texts": ["арбуз\nбанан\n\nвишня"], "replace": True},
                    )
                    self.assertEqual(response.status_code, 200)
                self.assertEqual(client.get("/api/v1/model").json()["texts_count"], 3)
                model = app.state.service.model
                tokenizer = app.state.service.tokenizer
                first = model._frequencies.get(
                    (tokenizer.bos_id,) * model._context_size
                )
                self.assertEqual(
                    {
                        tokenizer.decode([token]): count
                        for token, count in first.items()
                    },
                    {"а": 1, "б": 1, "в": 1},
                )
                for text in model._texts:
                    self.assertNotIn("\n", text)
                with patch(
                    "markov.markov_chain.random.choices",
                    side_effect=lambda tokens, weights, tokenizer=tokenizer: (
                        [tokenizer.piece_to_token["б"]]
                        if tokenizer.piece_to_token["б"] in tokens
                        else [tokens[0]]
                    ),
                ):
                    response = client.post("/api/v1/generate", json={})
                    self.assertTrue(response.json()["text"].startswith("б"))
