import json
from nltk_utils import tokenize, stem, bag_of_word
import string
import numpy as np

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

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_word(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
