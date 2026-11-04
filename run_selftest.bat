@echo off
setlocal EnableExtensions
rem ---------------------------------------------------------------------------
rem  Verifies the installation: dependencies, FFmpeg, the GUI, and an
rem  end-to-end detection test against a clip it renders with known twitch
rem  times. Run this after installing and after copying the folder to another
rem  machine.
rem
rem    run_selftest.bat            everything
rem    run_selftest.bat --quick    skip the clip rendering and detection test
rem
rem  Falls back to ..\.venv (the first-generation pipeline's environment) when
rem  this folder has no .venv of its own, so the checks can be run before
rem  install.bat.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

set "PY="
if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "%CD%\..\.venv\Scripts\python.exe" (
    set "PY=%CD%\..\.venv\Scripts\python.exe"
    echo  Using the existing environment one folder up ^(..\.venv^).
    echo.
)

if not defined PY (
    echo  No Python environment found - run install.bat first.
    pause
    exit /b 1
)

"%PY%" "%CD%\tools\selftest.py" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo  Some checks FAILED. docs\TROUBLESHOOTING.md explains each one.
) else (
    echo  All checks passed.
)
echo.
pause
exit /b %RC%
