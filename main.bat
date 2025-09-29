@echo off

cls
set "BAT_DIR=%~dp0"
pip install -r %BAT_DIR%requirements.txt > nul
cls
python "%BAT_DIR%src/main.py"
pause
cls
exit /b 0
