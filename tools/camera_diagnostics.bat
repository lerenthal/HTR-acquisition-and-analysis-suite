@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  Camera diagnostics.
rem
rem  Double-click this when the Acquisition tab cannot find the camera. It
rem  collects everything needed to work out why and writes it to
rem
rem      data\logs\camera_diagnostics.txt
rem
rem  Nothing is changed and no camera is opened for longer than a moment.
rem  Close the Windows Camera app first - it holds the device while it is open.
rem ===========================================================================

cd /d "%~dp0\.."
set "OUT=%CD%\data\logs\camera_diagnostics.txt"
if not exist "%CD%\data\logs" mkdir "%CD%\data\logs"

set "FF=%CD%\tools\ffmpeg\bin\ffmpeg.exe"
if not exist "%FF%" set "FF=ffmpeg"
set "PY=%CD%\.venv\Scripts\python.exe"

echo.
echo  Collecting camera diagnostics. This takes about 20 seconds.
echo  Writing to: %OUT%
echo.

> "%OUT%" echo HTR Studio camera diagnostics
>>"%OUT%" echo generated %DATE% %TIME%
>>"%OUT%" echo install  %CD%
>>"%OUT%" echo ffmpeg   %FF%
>>"%OUT%" echo.

echo [1/5] ffmpeg version
>>"%OUT%" echo ===========================================================================
>>"%OUT%" echo [1] ffmpeg -version
>>"%OUT%" echo ===========================================================================
"%FF%" -version >>"%OUT%" 2>&1

echo [2/5] DirectShow device list ^(raw, unparsed^)
>>"%OUT%" echo.
>>"%OUT%" echo ===========================================================================
>>"%OUT%" echo [2] ffmpeg -list_devices true -f dshow -i dummy
>>"%OUT%" echo     ^("Immediate exit requested" at the end is normal^)
>>"%OUT%" echo ===========================================================================
"%FF%" -hide_banner -list_devices true -f dshow -i dummy >>"%OUT%" 2>&1

echo [3/5] Windows device database
>>"%OUT%" echo.
>>"%OUT%" echo ===========================================================================
>>"%OUT%" echo [3] Windows PnP cameras
>>"%OUT%" echo ===========================================================================
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "Get-PnpDevice -Class Camera,Image -EA SilentlyContinue | Select-Object Status,Class,FriendlyName | Format-Table -AutoSize | Out-String -Width 200" >>"%OUT%" 2>&1

echo [4/5] Desktop-app camera privacy setting
>>"%OUT%" echo.
>>"%OUT%" echo ===========================================================================
>>"%OUT%" echo [4] Camera privacy ^(Settings ^> Privacy ^> Camera^)
>>"%OUT%" echo     "Value" must be Allow for FFmpeg to see the camera at all.
>>"%OUT%" echo ===========================================================================
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "$k='HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam'; foreach ($p in @($k, ($k + '\NonPackaged'))) { $v = (Get-ItemProperty -Path $p -Name Value -EA SilentlyContinue).Value; Write-Output ($p + ' = ' + $(if ($v) { $v } else { '(not set - default Allow)' })) }" >>"%OUT%" 2>&1

echo [5/5] OpenCV device probe
>>"%OUT%" echo.
>>"%OUT%" echo ===========================================================================
>>"%OUT%" echo [5] OpenCV index probe
>>"%OUT%" echo ===========================================================================
if exist "%PY%" (
    "%PY%" -c "import sys; sys.path.insert(0,'.'); from htr_studio import camera, ffmpeg_tools as F; [print(' ', p.label) for p in camera.enumerate_indices(6,'dshow')] or print('  (no indices opened with the dshow backend)'); print(); print('  parsed by HTR Studio:', [d.name for d in F.list_dshow_devices()]); print('  from Windows PnP   :', F.list_pnp_cameras())" >>"%OUT%" 2>&1
) else (
    >>"%OUT%" echo .venv not found - run install.bat first
)

>>"%OUT%" echo.
>>"%OUT%" echo === end ===

echo.
echo  Done.
echo.
echo  Written to:
echo     %OUT%
echo.
echo  Open it and send it over - section [2] is the one that matters.
echo.
start "" notepad "%OUT%"
pause
exit /b 0
