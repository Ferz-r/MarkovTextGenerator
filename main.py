from tokenizer import BPETokenizer
from text import texts

tokenizer = BPETokenizer(vocab_size=300000)
tokenizer.train(texts)
print(tokenizer._merges)
ids = tokenizer.encode("")
print(ids)
print(tokenizer.decode(ids))
