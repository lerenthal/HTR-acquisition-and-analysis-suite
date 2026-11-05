"""Start the application the way run_htr_studio.bat does, and check it starts.

This test exists because of a specific failure. Version 1.3.0-dev shipped a
hard startup crash -- ``_migrate`` was changed to return a tuple and one of
its two call sites was not updated, so ``_deep_merge`` received a tuple and
died before the window could appear -- with **144 passing checks behind it**.
Every one of those checks constructed widgets directly. Not one came through
``__main__``, so not one touched the settings load, the migration, the
defaults write, the Qt attribute setup or the import chain in the order the
real launcher does.

So this runs ``python -m htr_studio --check-startup`` in a fresh subprocess,
against a settings file in each of the states a real installation can be in,
and asserts it exits 0 having built the main window. A subprocess rather than
an in-process import, because a module already imported by the test harness
would hide exactly the import-order faults this is looking for.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.json"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def run_startup(timeout: float = 240.0) -> tuple[int, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Keep the parent's sys.path out of it: the subprocess must resolve the
    # package the same way the launcher does, from the install folder.
    env.pop("PYTHONPATH", None)
    try:
        p = subprocess.run([sys.executable, "-m", "htr_studio", "--check-startup"],
                           cwd=str(ROOT), env=env, capture_output=True,
                           text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timed out: startup did not finish"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def case(name: str, settings: dict | str | None) -> None:
    """Run a startup with ``config/settings.json`` in a given state."""
    backup = None
    if SETTINGS.exists():
        backup = SETTINGS.with_suffix(".json.testbak")
        shutil.copy2(SETTINGS, backup)
    try:
        if settings is None:
            SETTINGS.unlink(missing_ok=True)
        elif isinstance(settings, str):
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(settings, encoding="utf-8")
        else:
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        rc, out = run_startup()
        ok = rc == 0 and "startup check passed" in out
        detail = f"exit {rc}"
        if not ok:
            tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
            detail = f"exit {rc}: " + " | ".join(tail)
        check(f"starts with {name}", ok, detail)
    finally:
        if backup is not None:
            shutil.copy2(backup, SETTINGS)
            backup.unlink(missing_ok=True)
        elif SETTINGS.exists():
            SETTINGS.unlink(missing_ok=True)


def test_migration_persists() -> None:
    """A schema-2 file is corrected, saved once, and not re-migrated."""
    backup = None
    if SETTINGS.exists():
        backup = SETTINGS.with_suffix(".json.testbak")
        shutil.copy2(SETTINGS, backup)
    try:
        SETTINGS.write_text(json.dumps({
            "schema_version": 2,
            "analysis": {"max_area_frac": 0.45, "core_kernel": 9,
                         "threshold_k": 4.25}}, indent=2), encoding="utf-8")
        rc, out = run_startup()
        check("a stale settings file still starts", rc == 0, f"exit {rc}")
        check("the migration is announced",
              "migrated from schema 2" in out or "settings migration" in out,
              " | ".join(ln for ln in out.splitlines()
                         if "migrat" in ln.lower())[:110])
        stored = json.loads(SETTINGS.read_text(encoding="utf-8"))
        a = stored.get("analysis", {})
        check("the corrected ceiling is written to disk",
              a.get("max_area_frac") == 0.25, str(a.get("max_area_frac")))
        check("the corrected tail kernel is written to disk",
              a.get("core_kernel") == 0, str(a.get("core_kernel")))
        check("the operator's own threshold survives",
              a.get("threshold_k") == 4.25, str(a.get("threshold_k")))

        rc2, out2 = run_startup()
        check("a second start does not migrate again",
              rc2 == 0 and "migrated from schema" not in out2, f"exit {rc2}")
    finally:
        if backup is not None:
            shutil.copy2(backup, SETTINGS)
            backup.unlink(missing_ok=True)
        else:
            SETTINGS.unlink(missing_ok=True)


def test_every_module_imports() -> None:
    """Import every module in the package in a fresh interpreter.

    A circular import or a reference to a name that moved shows up here even
    if no test happens to construct the object involved. ``config`` imports
    ``analysis`` at module level to derive its defaults, which is exactly the
    kind of edge that can turn into a cycle later.
    """
    mods = sorted(p.stem for p in (ROOT / "htr_studio").glob("*.py")
                  if p.stem not in ("__init__", "__main__"))
    uimods = sorted(f"ui.{p.stem}" for p in (ROOT / "htr_studio" / "ui").glob("*.py")
                    if p.stem != "__init__")
    for m in mods + uimods:
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [sys.executable, "-c",
             f"import htr_studio.{m} as m; print(getattr(m, '__name__', '?'))"],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            errors="replace", timeout=120)
        tail = [ln for ln in (p.stderr or "").splitlines() if ln.strip()][-1:]
        check(f"htr_studio.{m} imports alone", p.returncode == 0,
              " ".join(tail)[:110])


def main() -> int:
    print("1. the real entry point, against each settings-file state")
    case("no settings file at all", None)
    case("an empty settings file", "{}")
    case("a current settings file", {"schema_version": 3})
    case("a schema-2 file holding the stale ceiling",
         {"schema_version": 2, "analysis": {"max_area_frac": 0.45}})
    case("a file with no schema_version", {"analysis": {"threshold_k": 5.0}})
    case("a file with unknown keys", {"schema_version": 3, "nonsense": {"a": 1},
                                      "analysis": {"not_a_field": 7}})
    case("a corrupt settings file", "{ this is not json")

    print("2. the migration is persisted, once")
    test_migration_persists()

    print("3. every module imports on its own")
    test_every_module_imports()

    print(f"\n{'STARTUP OK' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
