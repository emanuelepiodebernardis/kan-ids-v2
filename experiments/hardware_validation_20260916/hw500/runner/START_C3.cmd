@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
py -3.11 "%~dp0bootstrap.py" --board c3 %*
set "KANIDS_EXIT_CODE=%ERRORLEVEL%"
if not "%KANIDS_EXIT_CODE%"=="0" echo Stopped. Keep logs; no subsequent measurement was started.
exit /b %KANIDS_EXIT_CODE%
