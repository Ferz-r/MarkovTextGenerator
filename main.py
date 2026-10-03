from datasets import load_dataset

from markov_chain import Markovka
from tokenizer import RegexTokenizer

dataset = load_dataset("Mikimi/russian-wikipedia-top100k")
df = dataset["train"].to_pandas()
wikipedia = df.get("summary").to_list()

tokenizer = RegexTokenizer(min_frequency=5)
generator = Markovka(tokenizer, max_lengh=100)

generator.fit(texts=wikipedia)
print(f"Размер словаря {tokenizer.vocab_size}")
print(f"Словарь: {str(tokenizer.piece_to_token)[:200]}")


result = generator.generate()
print(f"Результат: {result}")
