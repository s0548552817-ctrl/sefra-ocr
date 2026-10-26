# -*- coding: utf-8 -*-
import sys
import os
import json
import io
import pickle
sys.path.append("engine")

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from docx import Document
import space_detector
import decoder

app = Flask(__name__)
app.secret_key = "sifra_secret_2026"

POPPLER_PATH = r"C:\poppler\poppler-23.08.0\Library\bin"
USERS_FILE = "data/users.json"
ADMIN_PASSWORD = "4850"  # סיסמת מנהל - שנה לפי הצורך

# משתנה גלובלי לעדכוני התקדמות
progress_message = ""

# =====================
# ניהול משתמשים
# =====================
def load_users():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, encoding='utf-8') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

# =====================
# דפי ניווט
# =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", error=None, success=None)

    first_name = request.form.get("first_name", "").strip()
    last_name  = request.form.get("last_name", "").strip()
    id_number  = request.form.get("id_number", "").strip()
    phone      = request.form.get("phone", "").strip()
    password   = request.form.get("password", "").strip()

    if not all([first_name, last_name, id_number, phone, password]):
        return render_template("register.html", error="יש למלא את כל השדות", success=None)

    users = load_users()
    username = f"{first_name} {last_name}"

    if username in users:
        return render_template("register.html", error="שם משתמש זה כבר קיים", success=None)

    users[username] = {"password": password, "id": id_number, "phone": phone}
    save_users(users)

    os.makedirs(f"data/users/{username}", exist_ok=True)

    # עיבוד דף האימון אם הועלה
    training_pdf = request.files.get("training_pdf")
    training_lines = 0
    if training_pdf and training_pdf.filename:
        try:
            import prepare_training_data
            pdf_bytes = training_pdf.read()
            training_lines = prepare_training_data.process_training_pdf(
                username, pdf_bytes=pdf_bytes
            )
        except Exception as e:
            print(f"שגיאה בעיבוד דף אימון: {e}")

    msg = f"נרשמת בהצלחה! שם המשתמש שלך: {username}"
    if training_lines > 0:
        msg += f" | דף האימון עובד: {training_lines} שורות נשמרו"

    return render_template("register.html", error=None, success=msg)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    users = load_users()
    if username not in users or users[username]["password"] != password:
        return render_template("login.html", error="שם משתמש או סיסמה שגויים")

    session["username"] = username
    return redirect(url_for("dashboard"))

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "GET":
        return render_template("forgot.html", error=None, result=None)

    phone = request.form.get("phone", "").strip()
    users = load_users()

    found = [(username, data) for username, data in users.items()
             if data.get("phone") == phone]

    if not found:
        return render_template("forgot.html", error="מספר הטלפון לא נמצא במערכת", result=None)

    results = []
    for username, data in found:
        results.append(f"שם משתמש: <b>{username}</b><br>סיסמה: <b>{data['password']}</b>")

    return render_template("forgot.html", error=None, result="<br><br>".join(results))


# =====================
# הגדרות מנהל
# =====================
@app.route("/admin", methods=["GET", "POST"])
def admin():
    # אם כבר מחובר כמנהל
    if session.get("is_admin"):
        return render_template("admin.html", boost=500, merge_message=None)

    if request.method == "GET":
        return render_template("admin_login.html", error=None)

    # בדיקת סיסמה
    password = request.form.get("admin_password", "").strip()
    if password == ADMIN_PASSWORD:
        session["is_admin"] = True
        return render_template("admin.html", boost=500, merge_message=None)
    else:
        return render_template("admin_login.html", error="סיסמה שגויה")

@app.route("/admin/merge", methods=["POST"])
def admin_merge():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    boost = int(request.form.get("boost", 500))
    import merge_vocabularies
    total = merge_vocabularies.merge(boost=boost)

    msg = f"המיזוג הושלם — {total} מילים (BOOST={boost}). הפעל מחדש את השרת כדי שהשינוי ישפיע."
    return render_template("admin.html", boost=boost, merge_message=msg)

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))

@app.route("/admin/users")
def admin_users():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))
    users = load_users()
    return render_template("admin_users.html", users=users, message=None)

@app.route("/admin/users/edit", methods=["POST"])
def admin_users_edit():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    original_name = request.form.get("original_name", "").strip()
    new_name      = request.form.get("username", "").strip()
    id_number     = request.form.get("id_number", "").strip()
    phone         = request.form.get("phone", "").strip()
    password      = request.form.get("password", "").strip()

    users = load_users()
    if original_name in users:
        # אם השם השתנה - מוחקים את הישן ויוצרים חדש
        del users[original_name]
        users[new_name] = {"password": password, "id": id_number, "phone": phone}
        save_users(users)
        # שינוי שם תיקיית המשתמש
        old_dir = f"data/users/{original_name}"
        new_dir = f"data/users/{new_name}"
        if os.path.exists(old_dir) and original_name != new_name:
            os.rename(old_dir, new_dir)

    return render_template("admin_users.html", users=load_users(), message="המשתמש עודכן בהצלחה")

@app.route("/admin/users/delete", methods=["POST"])
def admin_users_delete():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    username = request.form.get("username", "").strip()
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        # מחיקת תיקיית המשתמש
        import shutil
        user_dir = f"data/users/{username}"
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)

    return render_template("admin_users.html", users=load_users(), message="המשתמש נמחק")

# =====================
# מילונים
# =====================
DICT_PATHS = {
    "hebrew":  "models/hebrew_words.pkl",
    "aramaic": "models/aramaic_words.pkl",
    "merged":  "models/merged_words.pkl",
}
DICT_NAMES = {"hebrew": "עברי", "aramaic": "ארמי", "merged": "ממוזג"}

def load_dict_counts(dict_key):
    """טוען את מילון התדירויות — מחזיר dict של {מילה: תדירות}"""
    path = DICT_PATHS[dict_key]
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    # הפורמט: {counts: {...}, total, oov_logprob, prefix_discount}
    return data.get("counts", {})

def save_aramaic_counts(counts):
    """שומר את המילון הארמי בפורמט הנכון"""
    path = DICT_PATHS["aramaic"]
    with open(path, "rb") as f:
        data = pickle.load(f)
    data["counts"] = counts
    data["total"] = sum(counts.values())
    with open(path, "wb") as f:
        pickle.dump(data, f)

@app.route("/admin/dictionary")
def admin_dictionary():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    current = request.args.get("dict", "aramaic")
    search  = request.args.get("search", "").strip()

    counts = load_dict_counts(current)

    # סינון לפי חיפוש
    if search:
        items = [(w, f) for w, f in counts.items() if search in w]
    else:
        items = list(counts.items())

    # מיון לפי תדירות, הגבלה ל-300 הראשונים לביצועים
    items = sorted(items, key=lambda x: x[1], reverse=True)[:300]

    return render_template("admin_dictionary.html",
        current=current,
        current_name=DICT_NAMES[current],
        total=len(counts),
        words=items,
        search=search,
        message=None)

@app.route("/admin/dictionary/add", methods=["POST"])
def admin_dict_add():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    word = request.form.get("new_word", "").strip()
    freq = int(request.form.get("new_freq", 10))

    counts = load_dict_counts("aramaic")
    counts[word] = counts.get(word, 0) + freq
    save_aramaic_counts(counts)

    return redirect(url_for("admin_dictionary", dict="aramaic"))

@app.route("/admin/dictionary/delete", methods=["POST"])
def admin_dict_delete():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    word = request.form.get("word", "").strip()
    counts = load_dict_counts("aramaic")
    if word in counts:
        del counts[word]
        save_aramaic_counts(counts)

    return redirect(url_for("admin_dictionary", dict="aramaic"))

@app.route("/admin/add_text", methods=["GET", "POST"])
def admin_add_text():
    if not session.get("is_admin"):
        return redirect(url_for("admin"))

    if request.method == "GET":
        return render_template("admin_add_text.html", message=None)

    import re
    from collections import Counter

    # איסוף הטקסט - מקובץ Word או מהדבקה
    text = ""
    docx_file = request.files.get("docx_file")
    pasted = request.form.get("pasted_text", "").strip()

    if docx_file and docx_file.filename:
        from docx import Document
        import tempfile
        # שמירה זמנית וקריאה
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(docx_file.read())
        tmp.close()
        doc = Document(tmp.name)
        text = "\n".join(p.text for p in doc.paragraphs)
        os.unlink(tmp.name)
    elif pasted:
        text = pasted
    else:
        return render_template("admin_add_text.html", message="לא התקבל טקסט")

    # חילוץ מילים עבריות
    words = re.findall(r"[א-ת][א-ת\-'״\"]*", text)
    new_counter = Counter(words)

    # מיזוג למילון הארמי
    counts = load_dict_counts("aramaic")
    for word, freq in new_counter.items():
        counts[word] = counts.get(word, 0) + freq
    save_aramaic_counts(counts)

    msg = f"נוספו {len(new_counter)} מילים ייחודיות ({len(words)} סה\"כ) למילון הארמי"
    return render_template("admin_add_text.html", message=msg)

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

# =====================
# עדכון התקדמות
# =====================
@app.route("/progress")
def progress():
    return jsonify({"message": progress_message})

# =====================
# עיבוד קובץ
# =====================
@app.route("/process", methods=["POST"])
def process():
    global progress_message

    if "username" not in session:
        return jsonify({"error": "לא מחובר"}), 401

    if "file" not in request.files:
        return jsonify({"error": "לא הועלה קובץ"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    file_bytes = file.read()

    # שם המסמך שהמשתמש נתן (ברירת מחדל אם ריק)
    doc_name = (request.form.get("doc_name") or "תוצאת_פיענוח").strip()

    progress_message = "טוען קובץ..."

    # המרה לתמונות
    pages = []
    if filename.endswith(".pdf"):
        pil_pages = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
        for pil_page in pil_pages:
            page_cv = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
            pages.append(page_cv)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        page_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if page_cv is None:
            return jsonify({"error": "לא ניתן לקרוא את הקובץ"}), 400
        pages.append(page_cv)

    # חיתוך לשורות
    progress_message = "מנתח שורות..."
    all_lines = []
    for page in pages:
        lines = space_detector.detect_lines(page)
        all_lines.extend(lines)

    if not all_lines:
        return jsonify({"error": "לא נמצאו שורות בדף"}), 400

    total = len(all_lines)

    # רוחב התוכן של כל שורה — לזיהוי סוף פסקה (שורה קצרה)
    widths = [space_detector.content_width(line) for line in all_lines]
    avg_width = (sum(widths) / len(widths)) if widths else 1
    PARAGRAPH_THRESHOLD = 0.92  # שורה מתחת ל-80% מהרוחב הממוצע = סוף פסקה

    # פיענוח שורה שורה עם עדכון התקדמות
    user = session.get("username")
    print(f"=== פיענוח עבור: '{user}' ===")
    results = []
    for i, line in enumerate(all_lines):
        progress_message = f"מפענח שורה {i+1} מתוך {total}..."
        text = decoder.decode_line(line, user=user)
        results.append(text)

    progress_message = "הסתיים!"

    # הרכבת הטקסט כזרימת פסקאות:
    # מצרפים שורות רצופות לטקסט אחד (עם רווח ביניהן), ורק כששורה קצרה
    # (< 80% מהרוחב הממוצע = סוף פסקה) יורדים שורה ומתחילים פסקה חדשה.
    paragraphs = []
    current = []
    for i, text in enumerate(results):
        current.append(text.strip())
        is_short = widths[i] < avg_width * PARAGRAPH_THRESHOLD
        is_last = (i == total - 1)
        if is_short or is_last:
            paragraphs.append(" ".join(current))  # סוגרים פסקה
            current = []
    full_text = "\n".join(paragraphs)

    return jsonify({"text": full_text, "lines": total, "doc_name": doc_name})

# =====================
# הורדה כ-Word
# =====================
@app.route("/download_word", methods=["POST"])
def download_word():
    data = request.get_json()
    text = data.get("text", "")

    doc = Document()
    doc.add_heading("תוצאת פיענוח", 0)
    for line in text.split("\n"):
        p = doc.add_paragraph(line)
        p.alignment = 2  # ימין

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="תוצאת_פיענוח.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

# =====================
# הפעלה
# =====================
if __name__ == "__main__":
    print("השרת פועל על http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
