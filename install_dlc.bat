@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Optional: DeepLabCut + PyTorch, in a SEPARATE environment (.venv-dlc).
rem
rem  Separate on purpose. DeepLabCut pins numpy and several other packages, and
rem  those pins have conflicted with what the rest of this app wants. Keeping it
rem  in its own environment means a DLC dependency problem can never stop HTR
rem  Studio from opening and scoring a trial.
rem
rem  You probably do not need this. The shipped detector uses no trained model
rem  and scores F1 = 1.00 on both synthetic benchmarks. A pose model earns its
rem  keep when you want keypoints for other behaviours (grooming, rearing,
rem  locomotion), or if real footage defeats blob tracking - very low contrast,
rem  loose bedding, or two animals in frame.
rem
rem  Size and time: about 3 GB and 10 to 20 minutes on a reasonable connection.
rem  This is a CPU build, as the reference host has no CUDA GPU. Training a
rem  ResNet-50 on 200-400 labelled frames takes roughly 1.5 to 3 hours on CPU,
rem  against 10 to 15 minutes on an NVIDIA RTX 3060 or better.
rem ===========================================================================

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv-dlc"
set "PY=%VENV%\Scripts\python.exe"
set "UV=%ROOT%\tools\bin\uv.exe"

echo.
echo ===========================================================================
echo  DeepLabCut / PyTorch add-on
echo  Installs into: %VENV%
echo  About 3 GB, 10-20 minutes.
echo ===========================================================================
echo.
choice /c YN /n /m "Continue? [Y/N] "
if errorlevel 2 exit /b 0
echo.

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo [FAIL] Run install.bat first - this add-on sits beside the main install.
    pause
    exit /b 1
)

if exist "%PY%" (
    echo   .venv-dlc already exists; installing into it.
) else (
    echo [1/3] Creating the environment...
    if exist "%UV%" (
        "%UV%" venv "%VENV%" --python 3.11
    ) else (
        py -3.11 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
    )
    if errorlevel 1 (
        echo [FAIL] Could not create %VENV%.
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Installing PyTorch ^(CPU^) and DeepLabCut...
if exist "%UV%" (
    "%UV%" pip install --python "%PY%" -r "%ROOT%\requirements-dlc.txt"
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r "%ROOT%\requirements-dlc.txt"
)
if errorlevel 1 (
    echo.
    echo [FAIL] Installation failed. This stack is large and its pins change
    echo        often; the DeepLabCut install guide is at
    echo        https://deeplabcut.github.io/DeepLabCut/docs/installation.html
    echo.
    echo        HTR Studio itself is unaffected - .venv was not touched.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying...
"%PY%" -c "import torch, deeplabcut, numpy; print('   torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('   deeplabcut', deeplabcut.__version__); print('   numpy', numpy.__version__)"
if errorlevel 1 (
    echo [warn] The packages installed but did not import cleanly. See the output above.
)

echo.
echo   Recording the interpreter path in config\settings.json...
"%ROOT%\.venv\Scripts\python.exe" -c "import sys, os; sys.path.insert(0, '.'); from htr_studio import config; c = config.Config.load(); c.set('paths.dlc_python', os.path.join(os.getcwd(), '.venv-dlc', 'Scripts', 'python.exe')); c.save(); print('   paths.dlc_python =', c.get('paths.dlc_python'))"

echo.
echo ===========================================================================
echo  Done. The DLC environment is at %VENV%.
echo.
echo  To work with it, open a command prompt here and run:
echo      .venv-dlc\Scripts\python.exe -m deeplabcut
echo.
echo  docs\USER_MANUAL.md, section "Adding a pose model later", covers how to
echo  feed DeepLabCut keypoints into this pipeline: the detector chain after the
echo  head-angle trace is unchanged, so only the angle source is replaced.
echo ===========================================================================
echo.
pause
exit /b 0
