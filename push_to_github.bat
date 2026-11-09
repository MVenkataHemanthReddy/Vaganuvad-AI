@echo off
title Push Telugu-Hindi Speech Translation to GitHub
cd /d "%~dp0"
echo ==============================================================
echo Pushing to https://github.com/MVenkataHemanthReddy/Telugu-Hindi-Speech-Translation
echo ==============================================================
echo.
"C:\Users\mitta\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe" push -u origin main
echo.
echo ==============================================================
echo If you saw "Repository not found", make sure you created the
echo repository named "Telugu-Hindi-Speech-Translation" at:
echo https://github.com/new first!
echo ==============================================================
pause
