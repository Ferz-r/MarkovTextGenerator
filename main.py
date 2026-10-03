from text import texts
from tokenizer import RegexTokenizer

tokenizer = RegexTokenizer(min_frequency=1)
tokenizer.train(texts)

tokens = tokenizer.encode("Привет, мир!")
text = tokenizer.decode(tokens)

print(tokens)
print(text)
