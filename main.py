from datasets import load_dataset

# from text import texts
from tokenizer import RegexTokenizer

dataset = load_dataset("Mikimi/russian-wikipedia-top100k")
df = dataset["train"].to_pandas()
wikipedia = df.get("summary").to_list()

tokenizer = RegexTokenizer(min_frequency=1)
tokenizer.train(wikipedia)
vocab = str(tokenizer.piece_to_token)[:1000]

print(f"Размер словаря {tokenizer.vocab_size}")
print(f"Словарь: {vocab}")


tokens = tokenizer.encode("Как дела?")
text = tokenizer.decode(tokens)

print(tokens)
print(text)
