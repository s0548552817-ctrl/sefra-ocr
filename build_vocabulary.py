# -*- coding: utf-8 -*-
# קובץ זה מוציא מילים מקבצי Word ומוסיף למילון הקיים

import pickle
import re
from collections import Counter
from docx import Document

WORD_LM_PATH = r"D:\פרויקט-לפיני\models\hebrew_words.pkl"

def extract_text_from_docx(docx_path):
    """מחלצת טקסט מקובץ Word"""
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def build_vocabulary(docx_files):
    """
    מקבלת רשימת נתיבי קבצי Word,
    מחשבת תדירות מילים ומוסיפה למילון הקיים.
    """
    # טעינת המילון הקיים
    with open(WORD_LM_PATH, "rb") as f:
        existing = pickle.load(f)

    print(f"מילון קיים: {len(existing)} מילים")

    # ספירת מילים מהקבצים
    counter = Counter()
    for path in docx_files:
        print(f"מעבד: {path}")
        text = extract_text_from_docx(path)
        # חילוץ מילים עבריות בלבד
        words = re.findall(r"[א-ת][א-ת\-'״\"]*", text)
        counter.update(words)
        print(f"  נמצאו {len(words)} מילים ({len(Counter(words))} ייחודיות)")

    print(f"\nסה\"כ {len(counter)} מילים ייחודיות בקבצים")

    # מיזוג עם המילון הקיים
    new_words = {}
    for word, freq in counter.items():
        # דלג על מפתחות מיוחדים של Mishkefet
        if not isinstance(word, str) or not re.match(r"[א-ת]", word):
            continue
        if word in existing:
            existing[word] += freq
        else:
            existing[word] = freq
            new_words[word] = freq

    # עדכון סך הטוקנים
    total_added = sum(counter.values())
    existing['total'] = existing.get('total', 0) + total_added

    # הדפסת המילון שהתווסף
    print(f"\nמילים חדשות שנוספו ({len(new_words)}):")
    for word, freq in sorted(new_words.items(), key=lambda x: x[1], reverse=True):
        print(f"  {word}: {freq}")

    # שמירה
    with open(WORD_LM_PATH, "wb") as f:
        pickle.dump(existing, f)

    print(f"\nנוספו {len(new_words)} מילים חדשות")
    print(f"עודכן total ל: {existing['total']}")
    print(f"סה\"כ {len(existing)} מילים במילון המעודכן")


# הרצה
build_vocabulary([
    r"D:\למילון\pinchas.docx",
])
