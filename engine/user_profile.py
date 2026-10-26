# שמירה וטעינה של האותיות של כל משתמש
import os
import cv2
import sys
import numpy as np
sys.path.append("engine")
import image_processor


INVALID_CHARS = ['?', '/', ':', '*', '"']

def safe_name(letter):
    if letter in INVALID_CHARS:
        return f"letter_{ord(letter)}"
    return f"letter_{letter}"


def create_user(name):
    path = f"data/users/{name}"
    if os.path.exists(path):
        return False  # משתמש כבר קיים
    os.makedirs(path)
    return True

def load_user(name):
    path = f"data/users/{name}"
    if os.path.exists(path):
        return path 
    return None

def load_letter(user, letter):
    path = f"data/users/{user}/{safe_name(letter)}"
    if not os.path.exists(path):
        return None
    images = []
    for f in os.listdir(path):
        img = cv2.imdecode(np.fromfile(f"{path}/{f}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images




def save_letter(user, letter, image):
    image = image_processor.crop_to_content(image)  
    path = f"data/users/{user}/{safe_name(letter)}"
    os.makedirs(path, exist_ok=True)
    count = len(os.listdir(path))
    file_path = f"{path}/{count+1}.png"
    cv2.imencode('.png', image)[1].tofile(file_path)
    return True


