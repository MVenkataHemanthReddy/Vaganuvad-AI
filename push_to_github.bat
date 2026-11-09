@echo off
title Push Vaganuvad AI to GitHub
cd /d "%~dp0"
echo ==============================================================
echo Pushing to https://github.com/MVenkataHemanthReddy/Vaganuvad-AI
echo ==============================================================
echo.
"C:\Users\mitta\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe" push -u origin main
echo.
echo ==============================================================
echo Pushed successfully to https://github.com/MVenkataHemanthReddy/Vaganuvad-AI
echo ==============================================================
pause
