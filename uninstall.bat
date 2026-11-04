@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Removes the installed environment and downloaded tools from this folder.
rem
rem  It does NOT touch:
rem    data\recordings   your trials
rem    data\exports      your CSVs and figures
rem    config\settings.json
rem
rem  So after running this you can re-run install.bat and pick up where you
rem  left off. To remove everything, delete the windows_HTR folder yourself -
rem  but check data\recordings first, because that is where the trials live.
rem ===========================================================================

cd /d "%~dp0"
echo.
echo  This removes from %CD%:
echo     .venv            the Python environment
echo     .venv-dlc        the DeepLabCut environment, if present
echo     tools\bin        uv
echo     tools\ffmpeg     the FFmpeg build
echo     data\cache       thumbnails and probe scratch files
echo.
echo  It keeps your recordings, exports, logs and settings.
echo.
choice /c YN /n /m "Proceed? [Y/N] "
if errorlevel 2 exit /b 0
echo.

for %%D in (".venv" ".venv-dlc" "tools\bin" "tools\ffmpeg" "tools\_download" "data\cache") do (
    if exist "%CD%\%%~D" (
        echo   removing %%~D
        rmdir /s /q "%CD%\%%~D"
    )
)

if exist "%USERPROFILE%\Desktop\HTR Studio.lnk" (
    echo   removing the Desktop shortcut
    del "%USERPROFILE%\Desktop\HTR Studio.lnk"
)

echo.
echo  Done. Re-run install.bat to reinstall.
echo.
if exist "%CD%\data\recordings" (
    echo  Your recordings are still in:
    echo     %CD%\data\recordings
    echo.
)
pause
exit /b 0
