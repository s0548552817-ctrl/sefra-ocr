# -*- coding: utf-8 -*-
# מוודא שהתיקון ל-regex עבד ושהמילון מזהה ראשי תיבות
import sys
sys.path.insert(0, r'D:\פרויקט-לפיני\mishkefet\heb-ocr-main')
from hebocr import wordlm
from hebocr.wordlm import WordUnigramLM

print("קובץ שנטען בפועל:", wordlm.__file__)
print("תבנית WORD:", wordlm.WORD.pattern)

sample = 'ביאור הפנ"י הביא דברי מהרש"א הנ"ל והק\' דא"כ'
print("\nשורת דוגמה:", sample)
print("מילים שחולצו:", wordlm.WORD.findall(sample))

# עכשיו בדיקה אמיתית: האם המילון נותן bonus לראשי התיבות?
lm = WordUnigramLM.load(r'models/aramaic_words.pkl')
print("\n=== bonus לכל מילה (0 = לא מוכרת, גבוה = מוכרת) ===")
for w in wordlm.WORD.findall(sample):
    print(f"  {w!r:12} known={lm.known(w)}  bonus={lm.bonus(w):.2f}")
