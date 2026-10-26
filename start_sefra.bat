@echo off
cd /d "d:\פרויקט-לפיני"
start "Sefra Server" /min python app.py
timeout /t 4 /nobreak >nul
start "" "http://localhost:5000"
