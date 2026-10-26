# מרכז הכל ומחזיר טקסט סופי
# -*- coding: utf-8 -*-

import sys
import os
import torch
from PIL import Image

# נתיב לתיקיית heb-ocr
HEB_OCR_DIR = r"D:\פרויקט-לפיני\mishkefet\heb-ocr-main"
sys.path.insert(0, HEB_OCR_DIR)

# נתיבי קבצי המודל
MODELS_DIR = r"D:\פרויקט-לפיני\models"
CHECKPOINT = os.path.join(MODELS_DIR, "mishkefet-v1.pt")
CHAR_LM    = os.path.join(MODELS_DIR, "hebrew_char6.pkl")
WORD_LM    = os.path.join(MODELS_DIR, "aramaic_words.pkl")
# WORD_LM    = os.path.join(MODELS_DIR, "hebrew_words.pkl")
# WORD_LM    = os.path.join(MODELS_DIR, "merged_words.pkl")
USERS_DIR  = r"D:\פרויקט-לפיני\data\users"

from hebocr.lm import CharNGramLM
from hebocr.recognize import Recognizer
from hebocr.wordlm import WordUnigramLM

# =====================
# בניית מזהה (Recognizer)
# =====================
def _build_recognizer(checkpoint):
    """בונה Recognizer עם checkpoint נתון"""
    return Recognizer(
        checkpoint,
        lm=CharNGramLM.load(CHAR_LM),
        lm_weight=0.4,
        beam_width=12,
        tta=3,
        word_lm=WordUnigramLM.load(WORD_LM),
        word_weight=0.2,
    )

# def _build_recognizer(checkpoint):
#     """בונה Recognizer עם checkpoint נתון"""
#     return Recognizer(
#         checkpoint,
#         lm=CharNGramLM.load(CHAR_LM),
#         lm_weight=0.4,
#         beam_width=12,
#         tta=3,
#     )
        


# המודל הבסיסי - נטען פעם אחת
_base_model = _build_recognizer(CHECKPOINT)

# מטמון (cache) של מודלים אישיים כבר טעונים
_user_models = {}

def _get_model(user=None):
    """
    מחזיר את המודל המתאים:
    - אם למשתמש יש model.pt — טוען אותו (עם מטמון)
    - אחרת — המודל הבסיסי
    """
    if not user:
        return _base_model

    # אם כבר טעון במטמון
    if user in _user_models:
        return _user_models[user]

    # בדיקה אם קיים מודל אישי
    user_model_path = os.path.join(USERS_DIR, user, "model.pt")
    if os.path.exists(user_model_path):
        print(f"טוען מודל אישי של {user}")
        # המודל האישי נשמר כ-state_dict בלבד — טוענים לתוך הרשת של המודל הבסיסי
        recognizer = _build_recognizer(CHECKPOINT)
        state = torch.load(user_model_path, map_location="cpu")
        recognizer.model.load_state_dict(state)
        recognizer.model.eval()
        _user_models[user] = recognizer
        return recognizer

    # אין מודל אישי — בסיסי
    return _base_model


# =====================
# פונקציות ראשיות
# =====================
def decode_page(line_images, user=None):
    """מקבלת רשימת תמונות שורות ומחזירה טקסט מלא"""
    model = _get_model(user)
    pil_images = []
    for img in line_images:
        if hasattr(img, 'shape'):
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(img)
    results = model.read_tta(pil_images)
    return "\n".join(results)


def decode_line(line_image, user=None):
    """מקבלת תמונת שורה אחת ומחזירה את הטקסט שלה"""
    model = _get_model(user)
    if hasattr(line_image, 'shape'):
        pil_image = Image.fromarray(line_image)
    else:
        pil_image = line_image
    results = model.read_tta([pil_image])
    return results[0]
