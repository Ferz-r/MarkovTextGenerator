import tempfile
import unittest
from collections import Counter
from pathlib import Path

from markov.application.service import MarkovService


class RetrainingTests(unittest.TestCase):
    OLD = ["кот спит.", "кот гуляет.", "One rare word."]
    NEW = ["A dog sleeps.", "A dog walks.", "One rare word.", "One rare word."]

    def assert_same_model(self, left, right):
        self.assertEqual(left.stats(), right.stats())
        self.assertEqual(left.tokenizer.piece_to_token, right.tokenizer.piece_to_token)
        self.assertEqual(left.model.transitions, right.model.transitions)
        self.assertEqual(Counter(left.model.texts), Counter(right.model.texts))

    def test_corpus_order_does_not_change_vocabulary_or_transition_weights(self):
        for tokenizer in ("character", "regex"):
            for context in (1, 3, 15):
                with self.subTest(tokenizer=tokenizer, context=context):
                    first = MarkovService(tokenizer_type=tokenizer, n_gramm=context)
                    second = MarkovService(tokenizer_type=tokenizer, n_gramm=context)
                    first.train(self.OLD + self.NEW, replace=True)
                    second.train(self.NEW + self.OLD, replace=True)
                    self.assert_same_model(first, second)

    def test_retraining_after_restore_matches_fresh_training_and_survives_restart(self):
        for tokenizer in ("character", "regex"):
            for context in (1, 3, 15):
                with self.subTest(tokenizer=tokenizer, context=context):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "model.pkl.gz"
                        initial = MarkovService(
                            tokenizer_type=tokenizer, n_gramm=context, model_path=path
                        )
                        initial.train(self.OLD, replace=True)
                        initial.generate("", "кот")
                        initial.save()
                        restored = MarkovService.load(path)
                        self.assertEqual(restored.model.cached_topics, ("кот",))
                        restored.train(self.NEW)
                        fresh = MarkovService(tokenizer_type=tokenizer, n_gramm=context)
                        fresh.train(self.OLD + self.NEW, replace=True)
                        self.assert_same_model(restored, fresh)
                        self.assertEqual(restored.model.cached_topics, ())
                        self.assert_same_model(MarkovService.load(path), fresh)
                        if tokenizer == "regex":
                            # The old snapshot excludes this word at frequency 1;
                            # new texts raise its total frequency to the threshold of 3.
                            self.assertIn("rare", restored.tokenizer.piece_to_token)
                        self.assertEqual(
                            restored.model.prepare_topic("dog").copy(),
                            fresh.model.prepare_topic("dog").copy(),
                        )

    def test_new_corpus_has_equal_starting_weight_when_text_counts_are_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            service = MarkovService(
                tokenizer_type="regex", min_frequency=1, n_gramm=3, model_path=path
            )
            service.train(["кот спит."] * 5, replace=True)
            service = MarkovService.load(path)
            service.train(["Dog sleeps."] * 5)
            bos = service.tokenizer.bos_id
            first_tokens = service.model.transitions[(bos, bos, bos)]
            self.assertEqual(
                {
                    service.tokenizer.decode([token]): count
                    for token, count in first_tokens.items()
                },
                {"кот": 5, "Dog": 5},
            )
