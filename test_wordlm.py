# -*- coding: utf-8 -*-
import pickle

path = r"C:\Users\User\Downloads\heb-ocr-main\heb-ocr-main\hebrew_words.pkl"

with open(path, "rb") as f:
    words = pickle.load(f)

print(f"סוג האובייקט: {type(words)}")
print(f"מספר מילים: {len(words)}")

# הצגת 5 דוגמאות לראות את הפורמט
print("\n5 דוגמאות:")
for i, (word, val) in enumerate(words.items()):
    print(f"  '{word}': {type(val).__name__} = {val}")
    if i >= 4:
        break
