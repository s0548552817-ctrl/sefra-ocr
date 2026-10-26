# -*- coding: utf-8 -*-
# מכין דאטה לאימות (validation) מדף הסרוק של המשתמש.
# דף ה-validation נוצר ונסרק באותו אופן כמו דף אימון 2:
#   - ה-HTML נוצר לרוחב, אז מסובבים 90°.
#   - אחרי הסיבוב: טקסט מודפס בצד שמאל, כתב-יד בצד ימין.
# לכן משתמשים באותה פונקציה גנרית עם אותם פרמטרים, רק שומרים
# ל-validation_data (לא training_data) ובלי append.

from prepare_training_data import process_training_pdf

if __name__ == "__main__":
    USER = "פיני שפירא"

    count = process_training_pdf(
        USER,
        pdf_path="validation_pini.pdf",
        labels_path="validation_text.txt",
        rotate=True,
        x_start=0.43, x_end=1.0,
        y_start=0.0, y_end=1.0,
        header_lines=2,          # מדלג על הכותרת ושורת ההוראות המודפסות
        clean_left_strip=0.025,  # מלבין את קו ההפרדה האנכי האפור
        append=False,            # קובץ עצמאי — לא מצטרף לאימון
        subdir="validation_data",
    )
    print(f"סה\"כ {count} זוגות validation")
