# -*- coding: utf-8 -*-
# סקריפט בדיקה בלבד — לא נוגע בדאטה.
# שומר תמונות כדי לראות איפה הכתב-יד אחרי הסיבוב, ומה החיתוך הנוכחי לוקח.

import cv2
import numpy as np
from pdf2image import convert_from_path

POPPLER_PATH = r"C:\poppler\poppler-23.08.0\Library\bin"
PDF = "training_page_2.pdf"

# טוענים רק את עמוד 1
pages = convert_from_path(PDF, poppler_path=POPPLER_PATH, first_page=1, last_page=1)
page = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

# סיבוב 90° נגד כיוון השעון (כמו בקוד)
rotated = cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)
h, w = rotated.shape[:2]
print(f"גודל אחרי סיבוב: רוחב={w}, גובה={h}")

# שומרים את הדף המלא אחרי הסיבוב — לראות איפה הכתב-יד
cv2.imencode('.png', rotated)[1].tofile("debug_full.png")
print("נשמר debug_full.png — הדף המלא אחרי הסיבוב")

# מציירים קווים אנכיים כל 10% כדי לדעת לאיזה אחוז x להגדיר
marked = rotated.copy()
for pct in range(10, 100, 10):
    x = int(w * pct / 100)
    cv2.line(marked, (x, 0), (x, h), (0, 0, 255), 2)
    cv2.putText(marked, f"{pct}%", (x + 5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
cv2.imencode('.png', marked)[1].tofile("debug_marked.png")
print("נשמר debug_marked.png — עם קווי אחוז x (אדום)")

# החיתוך הנוכחי: x 42%-100% (צד ימין = הכתב-יד)
xs, xe = int(w * 0.42), int(w * 1.0)
crop = rotated[:, xs:xe]
cv2.imencode('.png', crop)[1].tofile("debug_crop.png")
print(f"נשמר debug_crop.png — החיתוך הנוכחי (x {xs}..{xe})")
