@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  HTR Studio - Windows installer
rem
rem  Creates a self-contained installation inside this folder:
rem
rem    tools\bin\uv.exe          downloads and builds the Python environment
rem    tools\ffmpeg\bin\         static FFmpeg + FFprobe (recording, frame counts)
rem    .venv\                    Python 3.11 with the packages in requirements.txt
rem    data\                     recordings, exports, logs (created on first run)
rem
rem  Nothing is written outside this folder, nothing goes on PATH, and nothing
rem  touches the registry, so the whole folder can be copied to another Windows
rem  machine and run there. Needs internet once.
rem
rem  Usage:   install.bat                 normal install
rem           install.bat /no-selftest    skip the verification step
rem           install.bat /force          rebuild the virtual environment
rem ===========================================================================

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "UV=%ROOT%\tools\bin\uv.exe"
set "RUN_SELFTEST=1"
set "FORCE=0"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="/no-selftest" set "RUN_SELFTEST=0"
if /i "%~1"=="/force" set "FORCE=1"
shift
goto parse
:parsed

echo.
echo ===========================================================================
echo  HTR Studio installer
echo  Target folder: %ROOT%
echo ===========================================================================
echo.

rem --- a path with no write access, or one inside Program Files, breaks this ---
echo test> "%ROOT%\.write_test" 2>nul
if not exist "%ROOT%\.write_test" (
    echo [FAIL] This folder is not writable.
    echo        HTR Studio keeps its environment, settings and logs inside its own
    echo        folder, so install it somewhere you can write - your user folder,
    echo        a data drive, or an external disk. Not Program Files.
    goto fail
)
del "%ROOT%\.write_test" >nul 2>&1

echo [1/5] Fetching uv and FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\bootstrap_tools.ps1" -Root "%ROOT%"
if errorlevel 1 (
    echo [warn] The tool bootstrap reported a problem. Carrying on - the next step
    echo        will fall back to a system Python if uv is missing.
)
echo.

echo [2/5] Creating the Python environment...
if "%FORCE%"=="1" if exist "%VENV%" (
    echo   removing the existing .venv as requested
    rmdir /s /q "%VENV%"
)

if exist "%PY%" (
    echo   .venv already exists - reusing it. Pass /force to rebuild.
    goto deps
)

if exist "%UV%" (
    echo   using uv; it downloads a managed Python 3.11 if this machine has none
    "%UV%" venv "%VENV%" --python 3.11
    if errorlevel 1 (
        echo   [warn] uv could not build the environment; trying the Windows
        echo          Python launcher instead
        goto try_py
    )
    goto deps
)

:try_py
if exist "%PY%" goto deps
echo   looking for a system Python 3.10-3.12
set "SYSPY="
for %%V in (3.12 3.11 3.10) do (
    if not defined SYSPY (
        py -%%V -c "import sys" >nul 2>&1 && set "SYSPY=py -%%V"
    )
)
if not defined SYSPY (
    python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,14) else 1)" >nul 2>&1 && set "SYSPY=python"
)
if not defined SYSPY (
    echo.
    echo [FAIL] No usable Python and uv is not available either.
    echo.
    echo        Either:
    echo          * get this machine online briefly and re-run install.bat, or
    echo          * install Python 3.11 from https://www.python.org/downloads/
    echo            ^(tick "Add python.exe to PATH"^) and re-run install.bat, or
    echo          * copy uv.exe into tools\bin by hand - see
    echo            docs\INSTALLATION.md, "Installing without internet access".
    goto fail
)
echo   using %SYSPY%
%SYSPY% -m venv "%VENV%"
if errorlevel 1 (
    echo [FAIL] Could not create the virtual environment with %SYSPY%.
    goto fail
)

:deps
if not exist "%PY%" (
    echo [FAIL] %PY% is missing, so the environment was not created.
    goto fail
)
echo.
echo [3/5] Installing Python packages ^(about 250 MB, two to three minutes^)...
if exist "%UV%" (
    "%UV%" pip install --python "%PY%" -r "%ROOT%\requirements.txt"
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r "%ROOT%\requirements.txt"
)
if errorlevel 1 (
    echo.
    echo [FAIL] Package installation failed.
    echo        The usual cause is no internet access or a proxy that needs
    echo        configuring. Set HTTPS_PROXY and re-run, or see
    echo        docs\INSTALLATION.md for the offline route.
    goto fail
)

echo.
echo [4/5] Creating folders and default settings...
rem Run with this folder as the working directory so '.' finds the package.
rem Embedding %ROOT% in the Python source would break if the install sits at a
rem drive root, where %CD% ends with a backslash.
"%PY%" -c "import sys; sys.path.insert(0, '.'); from htr_studio import paths, config; paths.ensure_dirs(); c = config.Config.load(); c.write_defaults_reference(); config.write_capture_presets_reference(); c.save(); print('   settings:', c.file)"
if errorlevel 1 (
    echo [FAIL] The package did not import. The folder may be incomplete -
    echo        re-copy windows_HTR and try again.
    goto fail
)

echo.
if "%RUN_SELFTEST%"=="0" (
    echo [5/5] Self-test skipped at your request. Run run_selftest.bat later.
    goto done
)
echo [5/5] Running the self-test ^(renders a short clip and scores it^)...
echo.
"%PY%" "%ROOT%\tools\selftest.py"
set "ST=%ERRORLEVEL%"
if not "%ST%"=="0" (
    echo.
    echo [warn] The self-test reported failures. The app will still start, but read
    echo        the lines marked FAIL above before recording a real trial.
    echo        docs\TROUBLESHOOTING.md covers each one.
)

:done
echo.
echo ===========================================================================
echo  Installed.
echo.
echo    Start the app          run_htr_studio.bat
echo    Re-check the install   run_selftest.bat
echo    Read first             docs\USER_MANUAL.md
echo    At the rig             docs\ACQUISITION_QUICKCARD.md
echo.
echo  Optional, later: install_dlc.bat adds DeepLabCut and PyTorch in a separate
echo  environment. The shipped detector needs no trained model, so this is only
echo  worth doing if you want pose keypoints for other behaviours.
echo ===========================================================================
echo.
choice /c YN /n /m "Create a Desktop shortcut? [Y/N] "
if errorlevel 2 goto end
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'),'HTR Studio.lnk')); $s.TargetPath=[IO.Path]::Combine('%ROOT%','run_htr_studio.bat'); $s.WorkingDirectory='%ROOT%'; $s.Description='HTR Studio - head-twitch response acquisition and analysis'; $s.Save(); Write-Host '  shortcut created on the Desktop'"

:end
echo.
pause
exit /b 0

:fail
echo.
echo ===========================================================================
echo  Installation did not complete. Nothing outside this folder was changed.
echo  docs\INSTALLATION.md has the manual steps and the offline route.
echo ===========================================================================
echo.
pause
exit /b 1
