# -*- coding: utf-8 -*-
# מכין דאטה לאימון מדפי האימון הסרוקים של המשתמש.
# תומך בשני פורמטים:
#   - דף ישן: סרוק לרוחב (rotate=True), כתב-יד בחצי שמאלי, 2 שורות כותרת לדלג.
#   - דף חדש: סרוק לאורך (rotate=False), כתב-יד בצד ימין (62%), בלי דילוג כותרות.
# ניתן לצבור כמה דפים לאותה תיקייה בעזרת append=True.

import sys
import os
sys.path.append("engine")
import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import space_detector

POPPLER_PATH = r"C:\poppler\poppler-23.08.0\Library\bin"


def process_training_pdf(user, pdf_path, labels_path,
                         rotate=False, x_start=0.0, x_end=0.5,
                         y_start=0.0, y_end=1.0,
                         header_lines=0, append=False,
                         clean_left_strip=0.0,
                         subdir="training_data",
                         pdf_bytes=None):
    """
    מעבד דף אימון סרוק ושומר שורות לדאטה של המשתמש.

    פרמטרים:
      user         - שם המשתמש
      pdf_path      - נתיב ל-PDF הסרוק (או השתמש ב-pdf_bytes)
      labels_path   - קובץ הטקסט המוכתב (התוויות)
      rotate        - סיבוב 90° נגד כיוון השעון (לדף סרוק לרוחב)
      x_start,x_end - טווח החיתוך האופקי באחוזים (0.0-1.0)
      y_start,y_end - טווח החיתוך האנכי באחוזים (0.0-1.0)
                      שים לב: החיתוך מתבצע *אחרי* הסיבוב.
      header_lines  - כמה שורות כותרת לדלג בעמוד הראשון
      append        - אם True: מוסיף לתיקייה הקיימת במקום לדרוס
      clean_left_strip - אחוז מרוחב החיתוך שיולבן משמאל (מסיר קו הפרדה אנכי)
      subdir        - תת-תיקייה לשמירה (training_data או validation_data)
    """
    output_dir = f"data/users/{user}/{subdir}"
    os.makedirs(output_dir, exist_ok=True)

    # אם לא מצרפים (append=False) — זו ההרצה הראשונה, מנקים את התיקייה
    # כדי שלא יישארו קבצים מהרצה קודמת שיבלבלו את המספור.
    if not append:
        for old in os.listdir(output_dir):
            if old.startswith("line_") and old.endswith(".png"):
                os.remove(os.path.join(output_dir, old))

    # טעינת הטקסט המוכתב
    with open(labels_path, encoding='utf-8') as f:
        all_labels = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[{os.path.basename(pdf_path)}] טקסט מוכתב: {len(all_labels)} שורות")

    # טעינת הדף
    if pdf_bytes:
        pil_pages = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    else:
        pil_pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    print(f"[{os.path.basename(pdf_path)}] נמצאו {len(pil_pages)} עמודים")

    all_lines   = []
    used_labels = []
    label_index = 0

    for page_num, pil_page in enumerate(pil_pages):
        # המרה ל-OpenCV
        page_cv = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)

        # סיבוב — רק אם הדף סרוק לרוחב
        if rotate:
            page_cv = cv2.rotate(page_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = page_cv.shape[:2]

        # חיתוך האזור שבו נמצא הכתב-יד (לפי אחוזים, אחרי הסיבוב)
        xs, xe = int(w * x_start), int(w * x_end)
        ys, ye = int(h * y_start), int(h * y_end)
        handwriting = page_cv[ys:ye, xs:xe]

        # הלבנת רצועה צרה בקצה השמאלי — מסירה את קו ההפרדה האנכי האפור
        # (קו אנכי ש-clean_image לא מסיר, כי היא מסירה רק קווים אופקיים)
        if clean_left_strip > 0:
            strip = int(handwriting.shape[1] * clean_left_strip)
            if strip > 0:
                handwriting[:, :strip] = 255

        # חיתוך לשורות
        lines = space_detector.detect_lines(handwriting)

        # דילוג על שורות כותרת בעמוד הראשון בלבד
        skip = header_lines if page_num == 0 else 0
        if skip:
            lines = lines[skip:]

        print(f"[{os.path.basename(pdf_path)}] עמוד {page_num+1}: {len(lines)} שורות")

        for line_img in lines:
            if label_index >= len(all_labels):
                break
            all_lines.append(line_img)
            used_labels.append(all_labels[label_index])
            label_index += 1

    # אזהרה אם מספר השורות שנחתכו לא תואם למספר התוויות
    if len(all_lines) != len(all_labels):
        print("=" * 60)
        print(f"!!! אזהרה [{os.path.basename(pdf_path)}] !!!")
        print(f"    נחתכו {len(all_lines)} שורות אבל יש {len(all_labels)} תוויות.")
        print(f"    אם המספרים לא תואמים - התוויות יזוזו! בדקי את החיתוך.")
        print("=" * 60)

    # קביעת נקודת ההתחלה למספור (לצבירה בין דפים)
    if append:
        existing = [f for f in os.listdir(output_dir)
                    if f.startswith("line_") and f.endswith(".png")]
        start_index = len(existing)
        mode = "a"
    else:
        start_index = 0
        mode = "w"

    # שמירה
    labels_file = open(f"{output_dir}/labels.txt", mode, encoding='utf-8')
    for i, (line_image, label) in enumerate(zip(all_lines, used_labels)):
        idx = start_index + i + 1
        filename = f"line_{idx:03d}.png"
        cv2.imencode('.png', line_image)[1].tofile(f"{output_dir}/{filename}")
        labels_file.write(f"{filename}|{label}\n")
    labels_file.close()

    print(f"[{os.path.basename(pdf_path)}] הסתיים! {len(all_lines)} זוגות "
          f"(מספור {start_index+1}..{start_index+len(all_lines)}) ב: {output_dir}\n")
    return len(all_lines)


# הרצה ישירה מהטרמינל — מעבד את שני הדפים לאותה תיקייה
if __name__ == "__main__":
    USER = "שבי שפירא"

    # דף 1 — גרסה ישנה: סרוק לרוחב, כתב-יד בחצי שמאלי, 2 כותרות לדלג
    process_training_pdf(
        USER,
        pdf_path="training_page_1.pdf",
        labels_path="training_text.txt",
        rotate=True,
        x_start=0.0, x_end=0.5,
        header_lines=2,
        append=False,   # מתחיל תיקייה חדשה
    )

    # דף 2 — אחרי הסיבוב הדף חוזר לכיוון הנכון (portrait): טקסט מודפס
    # בצד שמאל, כתב-יד בצד ימין. לכן חותכים לפי x — הצד הימני (~42%-100%).
    process_training_pdf(
        USER,
        pdf_path="training_page_2.pdf",
        labels_path="training_text_2.txt",
        rotate=True,
        x_start=0.43, x_end=1.0,
        y_start=0.0, y_end=1.0,
        header_lines=1,          # מדלג על שורה מודפסת אחת בראש הדף
        clean_left_strip=0.025,  # מלבין את קו ההפרדה האנכי האפור בקצה שמאל
        append=True,             # מוסיף לדף 1 במקום לדרוס
    )

    print("סיום — שני הדפים עובדו לתיקייה אחת מאוחדת.")
