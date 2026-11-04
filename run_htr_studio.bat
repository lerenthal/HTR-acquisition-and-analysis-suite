@echo off
setlocal EnableExtensions
rem ---------------------------------------------------------------------------
rem  Starts HTR Studio. Drag a video onto this file to open it in Review.
rem
rem  Interpreter search order:
rem    1. .venv next to this file        - what install.bat creates
rem    2. ..\.venv                       - the first-generation pipeline's
rem                                        environment, if this folder sits
rem                                        inside HTR_setup. It already has
rem                                        numpy, OpenCV, SciPy, matplotlib,
rem                                        PyQt5 and pyqtgraph, so the app runs
rem                                        unchanged. Only "move to recycle bin"
rem                                        is unavailable (no Send2Trash there),
rem                                        and it says so rather than failing.
rem    3. nothing                        - tell the user to run install.bat
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

set "PY="
set "PYW="
set "WHICH="

if exist "%CD%\.venv\Scripts\python.exe" (
    set "PY=%CD%\.venv\Scripts\python.exe"
    set "PYW=%CD%\.venv\Scripts\pythonw.exe"
    set "WHICH=own"
    goto found
)

if exist "%CD%\..\.venv\Scripts\python.exe" (
    set "PY=%CD%\..\.venv\Scripts\python.exe"
    set "PYW=%CD%\..\.venv\Scripts\pythonw.exe"
    set "WHICH=parent"
    goto found
)

echo.
echo  HTR Studio has no Python environment to run in.
echo.
echo  Run install.bat first. It builds .venv here and downloads FFmpeg;
echo  it needs internet access once and takes a few minutes.
echo.
pause
exit /b 1

:found
if "%WHICH%"=="parent" (
    echo.
    echo  Using the existing environment one folder up ^(..\.venv^).
    echo  That works for everything except "move to recycle bin".
    echo  Run install.bat when convenient to give HTR Studio its own.
    echo.
)

rem pythonw has no console window; fall back to python.exe if it is missing.
if exist "%PYW%" (
    start "" "%PYW%" -m htr_studio %*
) else (
    "%PY%" -m htr_studio %*
)
exit /b 0
