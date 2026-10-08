import itertools
import random
import unittest
from collections import defaultdict

from markov.domain.model import MarkovChain
from markov.domain.tokenizers.character import CharacterTokenizer
from markov.domain.tokenizers.regex import RegexTokenizer


def reference(tokenizer, texts, n):
    result = defaultdict(lambda: defaultdict(int))
    for text in texts:
        tokens = [tokenizer.bos_id] * (n - 1) + tokenizer.encode(text)
        for i in range(n, len(tokens)):
            for size in range(1, n + 1):
                result[tuple(tokens[i - size : i])][tokens[i]] += 1
    return {context: dict(counts) for context, counts in result.items()}


class TransitionTests(unittest.TestCase):
    def test_exact_counts_and_backoff_match_original(self):
        rng = random.Random(12)
        texts = ["", "а", "аб", "абба", "абба"] + [
            "".join(rng.choices("абв .", k=rng.randrange(30))) for _ in range(20)
        ]
        for tokenizer_type, n, frequency in itertools.product(
            (CharacterTokenizer, RegexTokenizer), (1, 3, 15, 50), (1, 4)
        ):
            with self.subTest(
                tokenizer=tokenizer_type.__name__, n=n, frequency=frequency
            ):
                tokenizer = tokenizer_type(
                    min_frequency=frequency, bos_id=7, eos_id=11, unk_id=4
                )
                model = MarkovChain(tokenizer, context_size=n)
                model.fit(texts)
                expected = reference(tokenizer, texts, n)
                self.assertEqual(model.transitions, expected)
                self.assertEqual(model.contexts_count, len(expected))
                for context, counts in expected.items():
                    self.assertEqual(model._frequencies.get(context), counts)
                self.assertEqual(model._frequencies.get((99999,), {}), {})

    def test_update_and_topic_use_current_vocabulary(self):
        tokenizer = CharacterTokenizer(min_frequency=3)
        model = MarkovChain(tokenizer, context_size=50)
        model.fit(["аб", "аб"])
        model.update(["аб", "вг"])
        self.assertEqual(
            model.transitions, reference(tokenizer, ["аб", "аб", "аб", "вг"], 50)
        )
        self.assertEqual(
            model.prepare_topic("аб").copy(),
            reference(tokenizer, ["аб", "аб", "аб"], 50),
        )
