@echo off
REM Lanceur AEGIS (Windows).
REM   run.bat                 menu interactif
REM   run.bat setup^|regen^|train^|eval^|backend^|frontend^|demo
setlocal
set "DIR=%~dp0"
set "PY=%DIR%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%DIR%run.py" %*
endlocal
