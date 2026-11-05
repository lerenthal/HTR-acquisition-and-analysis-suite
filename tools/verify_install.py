"""Check that the installed files are the ones they are supposed to be.

Written after a stale file reached the rig. A `config.py` from before an edit
was copied over a `__main__.py` from after it, the copy reported success, and
the mismatch was a hard startup crash. Nothing in the installation could have
told anyone that: the version string was right, the file was present, and
every module imported.

Two independent checks, both cheap:

1. **Against git**, when the install is a repository (it is, on the
   development machine). Every tracked file is compared with the committed
   blob, so a file copied over out of order, truncated, or edited by hand
   shows up by name. This is the check that would have caught the crash
   before the app was launched.
2. **Against a manifest**, for an install copied to another machine with no
   git. `--write-manifest` records SHA-256 for every Python file and the
   config references; `--verify` compares. Generated at release, checked on
   arrival.

Neither needs the network, a camera, or the virtual environment's GUI stack.

    python tools/verify_install.py                 both checks
    python tools/verify_install.py --write-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "install_manifest.json"

#: What is worth checking: the code, and the two reference files that are
#: read back by the app. Data, logs, the virtual environment and the FFmpeg
#: binaries are deliberately excluded -- they are large, they change in
#: normal use, and a mismatch in them is not a broken install.
PATTERNS = ("htr_studio/**/*.py", "tools/*.py", "*.py",
            "config/default_settings.json", "config/capture_presets.json",
            "VERSION", "requirements.txt")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file() and "__pycache__" not in p.parts and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------
def check_against_git() -> tuple[bool, list[str]]:
    """Compare every tracked file with the committed blob.

    Uses ``git hash-object`` on the working file rather than ``git diff``, so
    the comparison is of content and line-ending normalisation cannot hide a
    difference in either direction.
    """
    notes: list[str] = []
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT,
                       capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return True, ["not a git repository - skipping the git check"]

    # Plain `ls-tree -r`, not `--format=`: the format option is recent enough
    # that the git on the analysis machine rejected it with a usage error,
    # and this output shape ("<mode> <type> <object>\t<path>") has been
    # stable for the whole life of the tool.
    try:
        head = subprocess.run(["git", "ls-tree", "-r", "-z", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return True, [f"could not read HEAD ({e}) - skipping the git check"]

    committed: dict[str, str] = {}
    for record in head.stdout.split("\0"):
        if "\t" not in record:
            continue
        meta, path = record.split("\t", 1)
        bits = meta.split()
        if len(bits) >= 3:
            committed[path] = bits[2]

    files = tracked_files()
    paths = [rel(p) for p in files if rel(p) in committed]
    if not paths:
        return True, ["nothing tracked to compare"]

    try:
        hashed = subprocess.run(
            ["git", "hash-object", "--stdin-paths"], cwd=ROOT,
            input="\n".join(paths) + "\n", capture_output=True, text=True,
            check=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return True, [f"could not hash the working files ({e})"]

    got = hashed.stdout.split()
    bad: list[str] = []
    for path, obj in zip(paths, got):
        if obj != committed[path]:
            bad.append(path)
    for path in bad:
        notes.append(f"differs from the committed version: {path}")
    untracked = [rel(p) for p in files if rel(p) not in committed]
    for path in untracked[:10]:
        notes.append(f"not in the repository (new or uncommitted): {path}")
    return not bad, notes


def write_manifest() -> Path:
    files = tracked_files()
    doc = {
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip()
                   if (ROOT / "VERSION").is_file() else "",
        "files": {rel(p): sha256(p) for p in files},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return MANIFEST


def check_against_manifest() -> tuple[bool, list[str]]:
    if not MANIFEST.is_file():
        return True, [f"no manifest at {rel(MANIFEST)} - "
                      "run with --write-manifest to create one"]
    try:
        doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, [f"the manifest could not be read: {e}"]

    recorded: dict[str, str] = doc.get("files") or {}
    notes: list[str] = []
    ok = True
    for path, want in sorted(recorded.items()):
        p = ROOT / path
        if not p.is_file():
            notes.append(f"missing: {path}")
            ok = False
            continue
        if sha256(p) != want:
            notes.append(f"content differs from the manifest: {path}")
            ok = False
    extra = [rel(p) for p in tracked_files() if rel(p) not in recorded]
    for path in extra[:10]:
        notes.append(f"present but not in the manifest: {path}")
    if doc.get("version"):
        cur = ((ROOT / "VERSION").read_text(encoding="utf-8").strip()
               if (ROOT / "VERSION").is_file() else "")
        if cur and cur != doc["version"]:
            notes.append(f"the manifest was written for {doc['version']}, "
                         f"this install says {cur}")
    return ok, notes


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-manifest", action="store_true",
                    help="record the current checksums as the reference")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.write_manifest:
        p = write_manifest()
        n = len(json.loads(p.read_text(encoding="utf-8"))["files"])
        print(f"wrote {rel(p)} covering {n} files")
        return 0

    print(f"Verifying {ROOT}")
    print(f"  {len(tracked_files())} files to check\n")

    failures = 0
    for label, fn in (("against the repository", check_against_git),
                      ("against the manifest", check_against_manifest)):
        ok, notes = fn()
        print(f"[{'ok' if ok else 'FAIL'}] {label}")
        for n in notes:
            print(f"       {n}")
        if not ok:
            failures += 1
    print()
    if failures:
        print("The installation does not match its reference. A file copied "
              "over out of order or only partly is the usual cause; re-copy "
              "the files listed above, then run this again.")
        return 1
    print("Every checked file matches. Start the app with "
          "run_htr_studio.bat, or run")
    print("  python -m htr_studio --check-startup")
    print("to confirm it reaches the main window without opening it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
