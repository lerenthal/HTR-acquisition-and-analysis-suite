<#
.SYNOPSIS
    Downloads the two external tools HTR Studio needs: uv and FFmpeg.

.DESCRIPTION
    install.bat calls this. It is idempotent: anything already present is left
    alone, so re-running after a partial install only fetches what is missing.

    uv      -> tools\bin\uv.exe
              Builds the virtual environment and, if the machine has no suitable
              Python, downloads a managed CPython 3.11 itself. That is why uv is
              used rather than plain venv: the analysis computer does not need
              Python installed beforehand.

    FFmpeg  -> tools\ffmpeg\bin\ffmpeg.exe and ffprobe.exe
              A static build, so there is nothing to install system-wide and
              nothing to put on PATH. FFmpeg does the recording (it is the only
              path that sustains 240 fps on this camera) and FFprobe provides
              exact frame counts.

.PARAMETER Root
    The installation folder (the one holding install.bat).

.PARAMETER SkipFfmpeg
    Leave FFmpeg alone, e.g. when pointing the app at an existing build.
#>
[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipFfmpeg
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # a progress bar makes downloads slower

# Some Windows 10 builds still default to TLS 1.0, which every CDN now refuses.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
} catch {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}

$ToolsBin  = Join-Path $Root 'tools\bin'
$FfmpegDir = Join-Path $Root 'tools\ffmpeg'
$TempDir   = Join-Path $Root 'tools\_download'

function Write-Step([string]$Message) { Write-Host "  $Message" }
function Write-Ok  ([string]$Message) { Write-Host "  [ ok ] $Message" -ForegroundColor Green }
function Write-Warn([string]$Message) { Write-Host "  [warn] $Message" -ForegroundColor Yellow }
function Write-Err ([string]$Message) { Write-Host "  [FAIL] $Message" -ForegroundColor Red }

function Get-File {
    param([string[]]$Urls, [string]$OutFile)
    foreach ($url in $Urls) {
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            try {
                Write-Step "downloading $url"
                Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing `
                    -TimeoutSec 900 -MaximumRedirection 5
                if ((Get-Item $OutFile).Length -gt 100000) { return $true }
                Write-Warn "the file came back suspiciously small; retrying"
            } catch {
                Write-Warn "attempt $attempt failed: $($_.Exception.Message)"
                Start-Sleep -Seconds 3
            }
        }
    }
    return $false
}

New-Item -ItemType Directory -Force -Path $ToolsBin, $TempDir | Out-Null

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------
$uvExe = Join-Path $ToolsBin 'uv.exe'
if (Test-Path $uvExe) {
    Write-Ok "uv already present at $uvExe"
} else {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) {
        Copy-Item $existing.Source $uvExe -Force
        Write-Ok "copied the uv already on PATH into tools\bin"
    } else {
        $zip = Join-Path $TempDir 'uv.zip'
        $urls = @(
            'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
        )
        if (Get-File -Urls $urls -OutFile $zip) {
            $extract = Join-Path $TempDir 'uv'
            Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
            Expand-Archive -Path $zip -DestinationPath $extract -Force
            $found = Get-ChildItem $extract -Recurse -Filter 'uv.exe' |
                     Select-Object -First 1
            if ($found) {
                Copy-Item $found.FullName $uvExe -Force
                $uvx = Get-ChildItem $extract -Recurse -Filter 'uvx.exe' |
                       Select-Object -First 1
                if ($uvx) { Copy-Item $uvx.FullName (Join-Path $ToolsBin 'uvx.exe') -Force }
                Write-Ok "uv installed to $uvExe"
            } else {
                Write-Err 'the uv archive did not contain uv.exe'
            }
        } else {
            Write-Err @'
could not download uv.

Options:
  * If this machine has Python 3.10-3.12 installed, install.bat will fall back
    to the built-in venv module, so you can ignore this.
  * Behind a proxy: set HTTPS_PROXY before running install.bat, or download
    https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
    on another machine and put uv.exe in tools\bin yourself.
'@
        }
    }
}

# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------
if ($SkipFfmpeg) {
    Write-Step 'skipping FFmpeg as requested'
} else {
    $ffExe = Join-Path $FfmpegDir 'bin\ffmpeg.exe'
    if (Test-Path $ffExe) {
        Write-Ok "FFmpeg already present at $ffExe"
    } else {
        $zip = Join-Path $TempDir 'ffmpeg.zip'
        $urls = @(
            'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
        )
        if (Get-File -Urls $urls -OutFile $zip) {
            $extract = Join-Path $TempDir 'ffmpeg'
            Remove-Item $extract -Recurse -Force -ErrorAction SilentlyContinue
            Expand-Archive -Path $zip -DestinationPath $extract -Force
            $ff = Get-ChildItem $extract -Recurse -Filter 'ffmpeg.exe' |
                  Select-Object -First 1
            if ($ff) {
                $srcBin = Split-Path -Parent $ff.FullName
                New-Item -ItemType Directory -Force -Path (Join-Path $FfmpegDir 'bin') | Out-Null
                Copy-Item (Join-Path $srcBin '*.exe') (Join-Path $FfmpegDir 'bin') -Force
                # Keep the licence text beside the binaries.
                Get-ChildItem (Split-Path -Parent $srcBin) -Filter 'LICENSE*' -ErrorAction SilentlyContinue |
                    ForEach-Object { Copy-Item $_.FullName $FfmpegDir -Force }
                $names = (Get-ChildItem (Join-Path $FfmpegDir 'bin') -Filter '*.exe' |
                          ForEach-Object { $_.Name }) -join ', '
                Write-Ok "FFmpeg installed to $FfmpegDir\bin ($names)"
            } else {
                Write-Err 'the FFmpeg archive did not contain ffmpeg.exe'
            }
        } else {
            Write-Err @'
could not download FFmpeg.

HTR Studio cannot record or verify frame rates without it. Analysis of existing
videos still works.

To fix by hand: download a Windows build from https://ffmpeg.org/download.html,
and put ffmpeg.exe and ffprobe.exe in tools\ffmpeg\bin - or set paths.ffmpeg and
paths.ffprobe in config\settings.json to wherever they already live.
'@
        }
    }
}

Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '  Tool status:'
foreach ($pair in @(
        @{ Name = 'uv';      Path = (Join-Path $ToolsBin 'uv.exe') },
        @{ Name = 'ffmpeg';  Path = (Join-Path $FfmpegDir 'bin\ffmpeg.exe') },
        @{ Name = 'ffprobe'; Path = (Join-Path $FfmpegDir 'bin\ffprobe.exe') })) {
    if (Test-Path $pair.Path) {
        Write-Ok "$($pair.Name): $($pair.Path)"
    } else {
        Write-Warn "$($pair.Name): not installed"
    }
}
exit 0
