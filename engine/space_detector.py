# מדידת מרווחים, החלטה מתי מילה חדשה
import cv2
import numpy as np
from image_processor import crop_to_content


def clean_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # הפיכה - עכשיו הכתב לבן והרקע שחור
    inverted = cv2.bitwise_not(binary)
    # kernel אופקי ארוך - מוצא קווים אופקיים רצופים
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detected_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    # מחיקת הקווים מהתמונה המקורית
    inverted = cv2.subtract(inverted, detected_lines)
    # חזרה לשחור על לבן
    result = cv2.bitwise_not(inverted)
    # לאחר החיסור, מיזוג אנכי קטן לניקוי שאריות
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    result = cv2.erode(result, vertical_kernel)
    return result


# def clean_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh

def detect_lines(image):
    # גרסת גווני אפור - להחזרה בסוף (איכות טובה ל-Mishkefet)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # גרסת binary מנוקה - רק לחישוב איפה השורות
    binary = clean_image(image)
    black_pixels_per_row = np.sum(binary == 0, axis=1)

    threshold = image.shape[1] * 0.02

    in_line = False
    segments = []
    start = 0

    for i, count in enumerate(black_pixels_per_row):
        if not in_line and count > threshold:
            in_line = True
            start = i
        elif in_line and count <= threshold:
            in_line = False
            segments.append((start, i))

    if in_line:
        segments.append((start, i))

    heights = [end - start for start, end in segments]
    avg_height = sum(heights) / len(heights)

    lines = []
    for start, end in segments:
        if (end - start) >= avg_height * 0.5:
            h = end - start
            # padding אסימטרי: יותר מלמעלה (לאותיות עולות כמו ל')
            # וקצת מלמטה (לאותיות סופיות יורדות כמו ך, ף, ץ)
            pad_top = int(h * 0.5)
            pad_bottom = int(h * 0.35)
            top = max(0, start - pad_top)
            bottom = min(gray.shape[0], end + pad_bottom)
            lines.append(gray[top:bottom, :])  # חותך מגווני אפור

    return lines


def content_width(line_image):
    """
    מחזירה את רוחב התוכן (הדיו) בפועל בשורה — מהדיו השמאלי ביותר
    עד הימני ביותר. משמש לזיהוי שורה קצרה (סוף פסקה).
    line_image מגיע בגווני אפור.
    """
    # המרה ל-binary כדי לזהות היכן יש דיו (Otsu בוחר סף אוטומטית)
    _, binary = cv2.threshold(line_image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # עמודות שיש בהן דיו (פיקסלים שחורים)
    ink_per_column = np.sum(binary == 0, axis=0)
    ink_cols = np.where(ink_per_column > 0)[0]
    if len(ink_cols) == 0:
        return 0
    # רוחב התוכן = מהעמודה השמאלית ביותר עם דיו עד הימנית ביותר
    return int(ink_cols[-1] - ink_cols[0] + 1)


def detect_words(line_image):
    black_pixels_per_column = np.sum(line_image == 0, axis=0)
    threshold = line_image.shape[0] * 0.02

    # איסוף רווחים
    in_gap = False
    gaps = []
    gap_start = 0

    for i, count in enumerate(black_pixels_per_column):
        if not in_gap and count <= threshold:
            in_gap = True
            gap_start = i
        elif in_gap and count > threshold:
            in_gap = False
            gaps.append((gap_start, i, i - gap_start))  # התחלה, סוף, רוחב

    # הסרת רווחי שוליים לפני חישוב הממוצע
    inner_gaps = [g for g in gaps if g[0] > 0 and g[1] < line_image.shape[1]]

    if not inner_gaps:
       return [line_image]

    avg_gap = sum(g[2] for g in inner_gaps) / len(inner_gaps)

    word_boundaries = [g for g in inner_gaps if g[2] >= avg_gap * 1.2]


    # חיתוך לפי גבולות
    words = []
    prev = 0
    for gap in word_boundaries:
        words.append(line_image[:, prev:gap[0]])
        prev = gap[1]
    words.append(line_image[:, prev:])
    words = [crop_to_content(w) for w in words if w.shape[1] > 0]
    words.reverse()
    return words


def find_cut_points(word_image):
    # פרופיל אנכי - כמה פיקסלים שחורים בכל עמודה
    black_pixels_per_col = np.sum(word_image == 0, axis=0)

    # מנרמל לאחוזים מגובה התמונה
    profile = black_pixels_per_col / word_image.shape[0]

    cut_points = [0]  # תמיד מתחילים מ-0

    for x in range(1, len(profile) - 1):
        # עמק מקומי - קטן משכניו
        if profile[x] < profile[x-1] and profile[x] < profile[x+1]:
            # רק עמקים משמעותיים - פחות מ-30% מהגובה
            if profile[x] < 0.3:
                cut_points.append(x)

    cut_points.append(word_image.shape[1])  # תמיד מסיימים בסוף
    return cut_points

def segment_word(word_image):
    cut_points = find_cut_points(word_image)
    segments = []
    for i in range(len(cut_points) - 1):
        x_start = cut_points[i]
        x_end = cut_points[i + 1]
        segment = word_image[:, x_start:x_end]
        segments.append((x_start, x_end, segment))
    return segments

