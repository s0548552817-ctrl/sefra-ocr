# -*- coding: utf-8 -*-
# מייצר דף אימון 2 (HTML להדפסה) מתוך training_text_2.txt
# פורמט: כתב יד בימין, טקסט מודפס בשמאל, שורות מילוי ארוכות.
# כותרות (שורות שמסתיימות ב-":") מודגשות ונשארות שורה נפרדת.

TEXT_PATH = "training_text_2.txt"
OUT_PATH  = "training_page_2.html"

with open(TEXT_PATH, encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]

rows = []
for ln in lines:
    esc = (ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    is_heading = ln.strip().endswith(":")
    cls = "text heading" if is_heading else "text"
    rows.append(f'    <tr><td class="write"></td><td class="{cls}">{esc}</td></tr>')

rows_html = "\n".join(rows)

html = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
  <meta charset="UTF-8">
  <title>דף אימון 2 - ספרא</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; direction: rtl; }}
    h2 {{ text-align: center; margin: 6px 0; }}
    .instructions {{ background: #f0f0f0; padding: 8px; margin-bottom: 12px; border-radius: 5px; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ padding: 3px 8px; vertical-align: middle; height: 34px; }}
    td.write {{ width: 62%; border-bottom: 1px solid #aaa; }}
    td.text {{ width: 38%; font-size: 13px; color: #333; border-right: 2px solid #999; white-space: nowrap; }}
    td.heading {{ font-weight: bold; color: #000; }}
  </style>
</head>
<body>
  <h2>דף אימון 2 - ספרא</h2>
  <div class="instructions">
    <b>הוראות:</b> העתק כל שורה מהעמודה השמאלית (טקסט מודפס) לעמודה הימנית בכתב ידך. כתוב גם את הכותרות.
  </div>
  <table>
{rows_html}
  </table>
</body>
</html>
"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"נוצר {OUT_PATH} עם {len(lines)} שורות ({sum(1 for l in lines if l.strip().endswith(':'))} כותרות)")
