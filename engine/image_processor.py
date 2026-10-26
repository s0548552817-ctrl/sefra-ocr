# עיבוד תמונה, ניקוי רעש, חיתוך שורות
import cv2

def crop_to_content(image):
    coords = cv2.findNonZero(cv2.bitwise_not(image))
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


