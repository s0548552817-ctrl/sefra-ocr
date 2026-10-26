# -*- coding: utf-8 -*-
# אימון אישי של Mishkefet על כתב יד של משתמש ספציפי

import sys
import os
import random
import torch
import cv2
import numpy as np
from hebocr.decode import greedy_decode
from PIL import Image
from hebocr.data.transforms import preprocess, LINE_HEIGHT

# הוספת נתיב Mishkefet
sys.path.insert(0, r"D:\פרויקט-לפיני\mishkefet\heb-ocr-main")

from hebocr.recognize import Recognizer
from hebocr.lm import CharNGramLM

# =====================
# הגדרות — שני לפי הצורך
# =====================
USER         = "שבי שפירא"  # שם המשתמש שאותו מאמנים
TRAINING_DIR = fr"D:\פרויקט-לפיני\data\users\{USER}\training_data"
VAL_DIR      = fr"D:\פרויקט-לפיני\data\users\{USER}\validation_data"  # שורות אימות (loss+CER)
PINI_DIR     = r"D:\פרויקט-לפיני\data\users\פיני שפירא\validation_data"  # דף בדיקה למעקב בלבד
BASE_MODEL   = r"D:\פרויקט-לפיני\models\mishkefet-v1.pt"
CHAR_LM      = r"D:\פרויקט-לפיני\models\hebrew_char6.pkl"
# שני נתיבי מודל: הכי טוב + שני-הכי-טוב (אם קרוב) — כדי שתוכלי להשוות
BEST_MODEL   = fr"D:\פרויקט-לפיני\data\users\{USER}\model.pt"
SECOND_MODEL = fr"D:\פרויקט-לפיני\data\users\{USER}\model_second.pt"
EPOCHS       = 20   # יותר epochs כדי לפצות על LR נמוך יותר
LR           = 5e-5  # קצב למידה נמוך יותר — התכנסות יציבה (היה 1e-4 ועשה זיגזוג)
CLOSE_MARGIN = 0.10  # אם val_loss בתוך 10% מהטוב ביותר — נשמור גם כ-second

# =====================
# בחירת מכשיר
# =====================
# אוטומטי — GPU אם יש, אחרת CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"מכשיר: {device}")

# =====================
# טעינת הדאטה
# =====================
# קורא את labels.txt מתיקייה — כל שורה: "שם_קובץ|טקסט"
def load_pairs(directory):
    """טוען רשימת זוגות (נתיב_תמונה, טקסט) מתיקייה עם labels.txt."""
    labels_path = os.path.join(directory, "labels.txt")
    result = []
    if not os.path.exists(labels_path):
        return result
    with open(labels_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                filename, text = line.split('|', 1)
                img_path = os.path.join(directory, filename)
                if os.path.exists(img_path):
                    result.append((img_path, text))
    return result


pairs = load_pairs(TRAINING_DIR)          # דאטת אימון
val_pairs = load_pairs(VAL_DIR)           # דאטת אימות (loss + CER)
pini_pairs = load_pairs(PINI_DIR)         # דף בדיקה של פיני — מעקב ויזואלי בלבד

print(f"נטענו {len(pairs)} זוגות אימון, {len(val_pairs)} זוגות validation, "
      f"{len(pini_pairs)} שורות בדיקה של פיני")

# =====================
# טעינת המודל
# =====================
# טוענים את Mishkefet הבסיסי
recognizer = Recognizer(
    BASE_MODEL,
    lm=CharNGramLM.load(CHAR_LM),
    lm_weight=0.4,
    beam_width=1,  # greedy בזמן אימון — מהיר יותר
    tta=1,
)

# מוציאים את הרשת הפנימית לאימון
model = recognizer.model.to(device)
model.train()  # מצב אימון — מאפשר עדכון משקולות

# =====================
# optimizer — האחראי על עדכון המשקולות
# =====================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =====================
# פונקציית עזר — חישוב loss על זוג יחיד (בלי עדכון משקולות)
# =====================
def compute_loss(img_path, text):
    """מחשב את ה-CTC loss עבור זוג (תמונה, טקסט) יחיד."""
    image = Image.open(img_path).convert("RGB")
    arr = preprocess(image)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    width = torch.tensor([arr.shape[-1]], dtype=torch.long)

    logprobs = model(x)
    lengths = model.output_lengths(width)
    logprobs = logprobs.permute(1, 0, 2)

    charset = recognizer.charset
    targets = torch.tensor(
        [charset._to_idx[c] for c in text if c in charset._to_idx],
        dtype=torch.long
    )
    target_lengths = torch.tensor([len(targets)], dtype=torch.long)

    return torch.nn.functional.ctc_loss(
        logprobs, targets, lengths, target_lengths,
        blank=0, zero_infinity=True
    )


def decode_line(img_path):
    """מפענח שורה בודדת (greedy) ומחזיר את הטקסט."""
    image = Image.open(img_path).convert("RGB")
    arr = preprocess(image)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    width = torch.tensor([arr.shape[-1]], dtype=torch.long)
    logprobs = model(x)
    lengths = model.output_lengths(width)
    return greedy_decode(logprobs, lengths, recognizer.charset)[0]


def char_accuracy(pred, truth):
    """אחוז תווים מדויקים לפי מרחק עריכה (Levenshtein) — 100% = זהה."""
    # מטריצת מרחק עריכה סטנדרטית
    m, n = len(pred), len(truth)
    if n == 0:
        return 100.0 if m == 0 else 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if pred[i - 1] == truth[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    edits = dp[n]
    return max(0.0, (1 - edits / n) * 100)


# =====================
# מעקב המודל הכי טוב
# =====================
best_val = float("inf")   # ה-val_loss הנמוך ביותר עד כה
best_epoch = 0
best_state = None         # ה-state_dict של המודל הכי טוב (למקרה שנשמור כ-second)
second_val = float("inf") # ה-val_loss של המודל השני-הכי-טוב


# =====================
# לולאת האימון
# =====================
print(f"\nמתחיל אימון — {EPOCHS} epochs על {len(pairs)} שורות")

for epoch in range(EPOCHS):
    total_loss = 0.0

    # ערבוב השורות בכל epoch — אימון מעורב בין שני הדפים (מונע הטיה לפי סדר)
    random.shuffle(pairs)

    for img_path, text in pairs:
        # טעינת התמונה
        image = Image.open(img_path).convert("RGB")

        # עיבוד התמונה לטנסור
        arr = preprocess(image)  # numpy (1, H, W)
        x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 1, H, W)

        # רוחב התמונה — לחישוב אורך הפלט
        width = torch.tensor([arr.shape[-1]], dtype=torch.long)

        # הרצת המודל קדימה — מחזיר רק logprobs
        logprobs = model(x)                 # (batch, time, classes)
        lengths = model.output_lengths(width)  # אורך רצף הפלט

        # ctc_loss מצפה ל-(time, batch, classes) — מעבירים את הצירים
        logprobs = logprobs.permute(1, 0, 2)

        # הכנת התווית
        charset = recognizer.charset
        targets = torch.tensor(
            [charset._to_idx[c] for c in text if c in charset._to_idx],
            dtype=torch.long
        )
        target_lengths = torch.tensor([len(targets)], dtype=torch.long)

        # חישוב ה-loss
        loss = torch.nn.functional.ctc_loss(
            logprobs, targets, lengths, target_lengths,
            blank=0, zero_infinity=True
        )

        # אחורה — מחשב את הגרדיאנטים
        optimizer.zero_grad()
        loss.backward()

        # עדכון המשקולות
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(pairs)

    model.eval()
    with torch.no_grad():
        # ---- validation loss + CER על כל שורות ה-validation ----
        val_loss = float("inf")
        if val_pairs:
            val_total = 0.0
            acc_total = 0.0
            for v_img, v_text in val_pairs:
                val_total += compute_loss(v_img, v_text).item()
                pred = decode_line(v_img)
                acc_total += char_accuracy(pred, v_text)
            val_loss = val_total / len(val_pairs)
            val_cer  = acc_total / len(val_pairs)
            print(f"\n{'='*70}")
            print(f"epoch {epoch+1}/{EPOCHS} — loss={avg_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | דיוק תווים={val_cer:.1f}%")
        else:
            print(f"\nepoch {epoch+1}/{EPOCHS} — loss={avg_loss:.4f} | (אין validation)")

        # ---- הדפסת כל שורות ה-validation של שבי (אמת מול פענוח) ----
        if val_pairs:
            print(f"--- שורות validation של {USER} ---")
            for v_img, v_text in val_pairs:
                pred = decode_line(v_img)
                print(f"אמת:   {v_text}")
                print(f"פענוח: {pred}")
                print()

        # ---- הדפסת שורות הבדיקה של פיני (מעקב בלבד, בלי loss) ----
        if pini_pairs:
            print(f"--- שורות בדיקה של פיני שפירא (מעקב בלבד) ---")
            for p_img, p_text in pini_pairs:
                pred = decode_line(p_img)
                print(f"אמת:   {p_text}")
                print(f"פענוח: {pred}")
                print()
    model.train()

    # ---- שמירת המודל הכי טוב (ושני-הכי-טוב אם קרוב) ----
    if val_pairs:
        if val_loss < best_val:
            # שיא חדש. אם הטוב הקודם היה קרוב (בתוך CLOSE_MARGIN) — נשמור אותו
            # כ-second לפני שהחדש דורס אותו, כדי שתוכלי להשוות.
            if best_val != float("inf") and (best_val - val_loss) / best_val <= CLOSE_MARGIN:
                torch.save(best_state, SECOND_MODEL)
                second_val = best_val
                print(f">>> נשמר model_second.pt (val_loss={best_val:.4f}, epoch {best_epoch})")
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, BEST_MODEL)
            print(f">>> נשמר model.pt — הכי טוב עד כה (val_loss={val_loss:.4f}, epoch {best_epoch})")
        elif (val_loss - best_val) / best_val <= CLOSE_MARGIN and val_loss < second_val:
            # לא שיא, אבל קרוב מאוד לטוב ביותר — מועמד טוב ל-second
            second_val = val_loss
            torch.save(model.state_dict(), SECOND_MODEL)
            print(f">>> נשמר model_second.pt (val_loss={val_loss:.4f}, epoch {epoch+1}) — קרוב לטוב ביותר")

print(f"\n{'='*70}")
print(f"האימון הסתיים. המודל הכי טוב: epoch {best_epoch}, val_loss={best_val:.4f}")
print(f"נשמר ב: {BEST_MODEL}")
if second_val != float("inf"):
    print(f"מודל שני (להשוואה): val_loss={second_val:.4f} ב: {SECOND_MODEL}")
