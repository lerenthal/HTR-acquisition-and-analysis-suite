# Installing HTR Studio

HTR Studio is self-contained. Everything it needs lives inside the
`windows_HTR` folder: the Python environment, FFmpeg, the settings, the logs and
(by default) the recordings. Nothing is written to the registry, nothing goes on
PATH, and there is no system-wide installer. Copy the folder, run `install.bat`,
and you have a working installation.

- [Requirements](#requirements)
- [The normal install](#the-normal-install)
- [Moving it to the analysis computer](#moving-it-to-the-analysis-computer)
- [What ends up where](#what-ends-up-where)
- [Verifying the install](#verifying-the-install)
- [Installing without internet access](#installing-without-internet-access)
- [Behind a proxy](#behind-a-proxy)
- [The optional DeepLabCut add-on](#the-optional-deeplabcut-add-on)
- [Updating, reinstalling, removing](#updating-reinstalling-removing)

---

## Requirements

| | Minimum | Comment |
|---|---|---|
| Windows | 10 or 11, 64-bit | Acquisition uses DirectShow, which is Windows-only. |
| Disk | 2 GB for the install | Plus about **2.5 GB per 70-minute trial** at 320×240 / 240 fps. |
| RAM | 8 GB | Analysis streams the video, so memory use does not grow with recording length. |
| CPU | any 64-bit x86 | More cores means faster analysis and nothing else; see the note below. |
| Python | not required | `install.bat` downloads a managed Python 3.11 if the machine has none. |
| Internet | once, for about 5 minutes | See [Installing without internet access](#installing-without-internet-access) otherwise. |
| GPU | not needed | Nothing in the core pipeline uses CUDA. |

Install it somewhere **writable**: your user folder, a data drive, an external
disk. Not `C:\Program Files` — the app keeps its environment, settings and logs
beside itself, and Program Files is not writable without elevation.

> **On cores.** Analysis runs at roughly 300–500 frames per second per thread.
> A 70-minute trial is about a million frames, so single-threaded that is 35–55
> minutes; with 4–6 worker threads on an 8-core machine, 8–15 minutes. It is
> parallel over ranges of the file and the parallel result is verified
> bit-identical to the single-threaded one, so the only thing more cores change
> is how long you wait.

---

## The normal install

> **Shortcut if this folder is sitting inside the original `HTR_setup`.** That
> folder's existing `.venv` already contains numpy, OpenCV, SciPy, matplotlib,
> PyQt5 and pyqtgraph — everything HTR Studio needs. `run_htr_studio.bat` and
> `run_selftest.bat` fall back to it automatically, so you can start the app and
> test a camera **before** running `install.bat`. The only thing missing there is
> Send2Trash, so "move to recycle bin" is disabled and says why. Run `install.bat`
> when convenient to give HTR Studio its own environment and a bundled FFmpeg.

1. Copy the whole `windows_HTR` folder wherever you want it.
2. Double-click **`install.bat`**.
3. Wait. It prints five steps and takes three to five minutes.
4. When it offers a Desktop shortcut, say yes if you want one.
5. Start the app with **`run_htr_studio.bat`**.

What the five steps do:

| Step | What happens |
|---|---|
| 1 — tools | Downloads `uv` (the environment builder) into `tools\bin`, and a static FFmpeg build into `tools\ffmpeg`. Anything already there is left alone. |
| 2 — environment | Creates `.venv` with Python 3.11. If the machine has no suitable Python, `uv` fetches a managed one; if `uv` is unavailable, it falls back to the Windows `py` launcher. |
| 3 — packages | Installs `requirements.txt`: numpy, OpenCV, SciPy, matplotlib, PyQt5, pyqtgraph, Send2Trash. About 250 MB. **No PyTorch, no DeepLabCut.** |
| 4 — folders | Creates `data\` and writes `config\settings.json` plus a read-only copy of the shipped defaults. |
| 5 — self-test | Renders a short video with known twitch times, analyses it, and reports precision and recall. This is the step that tells you the install actually works. |

Options:

```
install.bat                 normal
install.bat /force          rebuild .venv from scratch
install.bat /no-selftest    skip step 5 (run run_selftest.bat later)
```

---

## Moving it to the analysis computer

This is what the folder is designed for.

**If the analysis computer has internet:** copy `windows_HTR` across (a USB
stick, a network share, anything) and run `install.bat` there. Five minutes.

**If it does not:** install on a networked machine first, then copy the folder
*including* `.venv` and `tools`. The environment is relocatable in practice —
`uv`-built virtual environments do not bake in absolute paths the way an older
`venv` does — but verify it after copying:

```
run_selftest.bat
```

If the self-test reports import failures after the copy, rebuild the environment
in place: `install.bat /force`. If that machine has no internet either, use the
offline route below.

**What to copy, and what not to:**

| Copy | Leave behind |
|---|---|
| everything in `windows_HTR\` | `data\cache\` — thumbnails and probe scratch, regenerated on demand |
| `config\settings.json` if you want the same settings | `data\logs\` — not worth carrying |
| `.venv` and `tools\` for an offline target | |

`data\recordings\` is where trials live by default. If the recordings are on a
different drive — which they usually should be — set **Settings → Recordings
root** to that path and the folder itself stays small.

---

## What ends up where

```
windows_HTR\
  install.bat                  run this first
  install_dlc.bat              optional pose-estimation add-on
  run_htr_studio.bat           start the app (drag a video onto it to open it)
  run_selftest.bat             verify the installation
  uninstall.bat                remove the environment, keep the data
  requirements.txt             core dependencies
  requirements-dlc.txt         optional add-on dependencies
  VERSION

  htr_studio\                  the application
    analysis.py                signal extraction and burst detection
    camera.py                  preview, UVC controls, lighting and flicker checks
    recorder.py                FFmpeg recording, pre-flight, sidecars
    library.py                 recordings, metadata, sessions
    report.py                  CSV and figure export
    ffmpeg_tools.py            tool discovery, device enumeration, probing
    theme.py, config.py, paths.py, logging_setup.py
    ui\                        the six tabs

  tools\
    bin\uv.exe                 downloaded by install.bat
    ffmpeg\bin\                downloaded by install.bat
    selftest.py
    make_synthetic_benchmark.py
    bootstrap_tools.ps1

  config\
    settings.json              live settings (created on first run)
    default_settings.json      the shipped defaults, for reference
    capture_presets.json       camera modes, editable

  docs\
    INSTALLATION.md            this file
    USER_MANUAL.md             the full manual
    ACQUISITION_QUICKCARD.md   one page to print and keep at the rig
    TROUBLESHOOTING.md         symptom to cause to fix
    PIPELINE_REVIEW.md         what changed from the first pipeline, and why

  data\                        created on first run
    recordings\                default recordings root
    exports\                   CSVs and figures
    logs\htr_studio.log        rotating log, first place to look
    cache\                     thumbnails, probe scratch
    benchmark\                 synthetic benchmark, if you generate one

  .venv\                       Python environment (created by install.bat)
  .venv-dlc\                   only if you run install_dlc.bat
```

---

## Verifying the install

```
run_selftest.bat
```

Nineteen checks across six groups. The ones that matter most:

* **detect planted twitches** — renders a 20-second clip with three twitches at
  known times and scores detection. Expect `precision=1.000 recall=1.000
  F1=1.000`. Anything less means a detector default has been changed or a
  dependency is behaving differently.
* **ignore grooming and fast turns** — the same clip plants a grooming bout and a
  fast turn. Neither should be detected. This is the check that distinguishes a
  working detector from one that merely fires on movement.
* **parallel extraction** — confirms the multi-threaded path gives signals
  identical to the single-threaded path, so there is no reason to prefer the slow
  one.
* **extract signals** — reports frames per second per thread, and from that how
  long a 70-minute trial will take on this machine.

`camera enumeration` warning on an analysis-only computer is expected and fine.

For a longer end-to-end check, use **Tools → Make a synthetic benchmark video**
in the app. It writes a 45-second clip with six twitches and three classes of
confounder (grooming, fast turns, a stretch of simulated PWM lighting flicker)
plus a ground-truth CSV. Analyse it, then use **Analysis → Validate against
ground truth** and point it at the CSV.

---

## Installing without internet access

The installer needs the network for three things: `uv`, FFmpeg, and the Python
packages. All three can be staged in advance on a machine that does have access.

### Option A — copy a finished installation (simplest)

Install on a networked Windows machine, then copy the whole folder including
`.venv` and `tools`. Run `run_selftest.bat` on the target to confirm. This is
usually the right answer.

### Option B — stage the parts by hand

On a networked machine:

1. **uv** — download
   `https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip`,
   extract, and put `uv.exe` in `windows_HTR\tools\bin\`.
2. **FFmpeg** — download a Windows build from <https://ffmpeg.org/download.html>
   (the gyan.dev "release essentials" build is the one the installer uses) and
   put `ffmpeg.exe` and `ffprobe.exe` in `windows_HTR\tools\ffmpeg\bin\`.
3. **Python wheels** — with Python 3.11 available, make a wheelhouse:

   ```
   mkdir wheelhouse
   py -3.11 -m pip download -r requirements.txt -d wheelhouse ^
       --only-binary=:all: --platform win_amd64 --python-version 311
   ```

   Copy `wheelhouse` into `windows_HTR\`.
4. **A Python 3.11 installer** if the target machine has no Python at all
   (python.org, "Windows installer (64-bit)").

On the offline machine:

```
rem install Python 3.11 first if the machine has none
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -r requirements.txt
run_selftest.bat
```

`install.bat` will then find the existing `.venv` and leave it alone on any
future run.

---

## Behind a proxy

Set the environment variables before running the installer, in the same console:

```
set HTTPS_PROXY=http://user:password@proxy.example.org:8080
set HTTP_PROXY=%HTTPS_PROXY%
install.bat
```

`uv`, `pip` and PowerShell's `Invoke-WebRequest` all honour these. If downloads
still fail, use the offline route above — that is usually faster than arguing
with a corporate proxy.

---

## The optional DeepLabCut add-on

**You probably do not need it.** The shipped detector uses no trained model and
scores F1 = 1.00 on both synthetic benchmarks. A pose model earns its keep when
you want keypoints for other behaviours — grooming, rearing, locomotion — or if
real footage defeats blob tracking (very low contrast, loose bedding, two animals
in frame).

```
install_dlc.bat
```

It creates a **separate** environment, `.venv-dlc`, and that separation is
deliberate: DeepLabCut pins numpy and several other packages, and those pins have
conflicted with what this app wants. In its own environment, a DLC dependency
problem can never stop HTR Studio from opening and scoring a trial.

About 3 GB and 10–20 minutes. CPU build, because the reference host has no CUDA
GPU; training a ResNet-50 on 200–400 labelled frames takes roughly 1.5–3 hours on
CPU against 10–15 minutes on an NVIDIA RTX 3060 or better. The model files are
portable, so training on a GPU machine and running inference here works fine.

`USER_MANUAL.md`, under *Adding a pose model later*, covers how keypoints feed
into this pipeline. Only the angle source changes; everything after it is
unchanged.

---

## Updating, reinstalling, removing

**Updating the app:** replace the `htr_studio\`, `tools\` and `docs\` folders
with the new versions and leave `.venv`, `config` and `data` alone. Then run
`run_selftest.bat`. If the new version needs a package the old one did not,
`install.bat` picks that up without rebuilding anything.

**Rebuilding the environment:** `install.bat /force`. Recordings, metadata,
analysis results and settings are untouched.

**Removing it:** `uninstall.bat` deletes `.venv`, `.venv-dlc`, the downloaded
tools and the cache, and keeps your recordings, exports and settings — so you can
reinstall into the same folder. To remove everything, delete the folder yourself,
**after checking `data\recordings\`**, because that is where the trials are
unless you moved the recordings root.
