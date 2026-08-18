@echo off
set PYTHON="D:\tools\Anaconda3\envs\checkin_detection\python.exe"
set SCRIPT="%~dp0src\checkin_system.py"

if not exist %PYTHON% (
    echo [ERROR] python not found: %PYTHON%
    echo Please rebuild conda env 'checkin_detection' per requirements.txt
    pause
    exit /b 1
)

if not exist %SCRIPT% (
    echo [ERROR] script not found: %SCRIPT%
    pause
    exit /b 1
)

echo Starting Checkin Detection System (env: checkin_detection) ...
echo.

%PYTHON% %SCRIPT%

echo.
echo [EXIT] code %errorlevel%
pause
