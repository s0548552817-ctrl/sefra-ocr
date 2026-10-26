# -*- coding: utf-8 -*-
import pickle

SPECIAL = {'total', 'oov_logprob', 'prefix_discount'}

with open(r"D:\פרויקט-לפיני\models\merged_words.pkl", "rb") as f:
    d = pickle.load(f)

top = sorted(
    [(k, v) for k, v in d.items() if k not in SPECIAL and isinstance(v, (int, float))],
    key=lambda x: x[1],
    reverse=True
)[:100]

for word, freq in top:
    print(f"{word}: {freq}")
