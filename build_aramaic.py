# -*- coding: utf-8 -*-
# קובץ זה בונה מילון ארמי/לשון הקודש מקובץ Word

import pickle
import re
from collections import Counter
from docx import Document

# נתיב קובץ הוורד
DOCX_PATH = r"D:\ספרים למילון\pinchas.docx"

# נתיב לשמירת המילון הארמי
OUTPUT_PATH = r"D:\פרויקט-לפיני\models\aramaic_words.pkl"

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# קריאת הטקסט
print(f"קורא: {DOCX_PATH}")
text = extract_text_from_docx(DOCX_PATH)

# ספירת מילים עבריות
words = re.findall(r"[א-ת][א-ת\-'״\"]*", text)
counter = Counter(words)

print(f"נמצאו {len(words)} מילים ({len(counter)} ייחודיות)\n")

# בניית המילון
import pickle
with open(r"D:\פרויקט-לפיני\models\hebrew_words.pkl", "rb") as f:
    hebrew = pickle.load(f)

aramaic = {
    "counts":          dict(counter),
    "total":           sum(counter.values()),
    "oov_logprob":     hebrew["oov_logprob"],
    "prefix_discount": hebrew["prefix_discount"],
}

# שמירה
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(aramaic, f)

print(f"המילון נשמר: {OUTPUT_PATH}")
print(f"\n--- 50 המילים הנפוצות ביותר ---")
for word, freq in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:50]:
    print(f"  {word}: {freq}")
