# פיענוח דף האימון וחיתוך האותיות לפי מיקום קבוע
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from layout import LAYOUT
import user_profile
from PIL import Image



BOX_W = 0.08  # רוחב ריבוע - 8% מרוחב התמונה
BOX_H = 0.05  # גובה ריבוע - 5% מגובה התמונה

def load_image(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".pdf"):
        from pdf2image import convert_from_path
        images = convert_from_path(path, poppler_path=r"C:\poppler\poppler-23.08.0\Library\bin")
        import numpy as np
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    return cv2.imread(path)



def crop_letter(image, x_pct, y_pct):
    x = int(image.shape[1] * x_pct)
    y = int(image.shape[0] * y_pct)
    w = int(image.shape[1] * BOX_W)
    h = int(image.shape[0] * BOX_H)
    margin = 20
    cropped = image[y:y+h, x:x+w]
    return cropped[margin:-margin, margin:-margin]

def process_training_page(image_path, user):
    image = load_image(image_path)
    if image is None:
        return None
    # המרה לשחור-לבן
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for item in LAYOUT:
        letter = item["letter"]
        x = item["x"]
        y = item["y"]
        letter_image = crop_letter(image, x, y)
        user_profile.save_letter(user, letter, letter_image)
    return True




