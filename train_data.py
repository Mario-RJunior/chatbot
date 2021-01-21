import json
from nltk_utils import tokenize, stem
import string

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []


for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

punctuation = string.punctuation
all_words = [stem(w) for w in all_words if w not in punctuation]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words)
print(tags)
