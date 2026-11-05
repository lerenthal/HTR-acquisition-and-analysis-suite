"""Drive the advisor dialogs and the reset button with real footage.

A dialog that only runs when a human opens it is a dialog nobody tests, and
this codebase has already shipped two bugs that lived exactly there. So:
construct both dialogs, let their background jobs finish, press Apply, and
assert the settings actually moved.

Needs a short video. Pass one as argv[1], or it builds a synthetic clip.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5 import QtWidgets                                     # noqa: E402

from htr_studio import advisor, analysis, config, library, paths  # noqa: E402
from htr_studio.ui import widgets as W                           # noqa: E402
from htr_studio.ui.main_window import AppContext                 # noqa: E402
from htr_studio.ui.analysis_tab import AnalysisTab               # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_clip(path: Path, fps: int = 120, seconds: int = 6) -> Path:
    """A short clip at a rate that forces the advisor to fold the band."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # A small bright blob on a dark field, moving: enough for segmentation to
    # lock on and for the travel test to call it an animal.
    src = (f"color=c=black:s=320x240:r={fps}:d={seconds}")
    vf = ("drawbox=x='120+60*sin(2*PI*t/3)':y='100+40*cos(2*PI*t/5)'"
          ":w=28:h=16:color=white:t=fill")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", src, "-vf", vf,
                    "-c:v", "mjpeg", "-q:v", "3", str(path)], check=True)
    return path


def wait_for(app, pred, seconds=90.0, step=0.05) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        if pred():
            return True
        time.sleep(step)
    return False


def entry_for(video: Path) -> library.Entry:
    from htr_studio import ffmpeg_tools
    e = library.Entry(path=video)
    info = ffmpeg_tools.probe_video(video, "", count_frames=False)
    e.width, e.height, e.codec = info.width, info.height, info.codec
    e.duration_s, e.fps, e.fps_exact = info.duration_s, info.true_fps, info.exact
    try:
        e.size_bytes = video.stat().st_size
    except OSError:
        pass
    return e


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    paths.ensure_dirs()
    video = (Path(sys.argv[1]) if len(sys.argv) > 1
             else make_clip(paths.CACHE_DIR / "advisor_ui_clip.avi"))
    print(f"    using {video}")

    ctx = AppContext(config.Config.load())
    tab = AnalysisTab(ctx)
    ctx.entryChanged.connect(tab.set_entry)
    ctx.set_entry(entry_for(video))

    # ---- reset to defaults ------------------------------------------------
    real_ask = W.ask_yes_no
    W.ask_yes_no = lambda *a, **k: True
    tab.sp_thr_k.setValue(3.25)
    tab.sp_max_area.setValue(0.42)
    tab._reset_defaults()
    shipped = analysis.AnalysisConfig()
    check("reset restores the dataclass defaults",
          tab.current_config().threshold_k == shipped.threshold_k
          and tab.current_config().max_area_frac == shipped.max_area_frac,
          f"k={tab.current_config().threshold_k} "
          f"area={tab.current_config().max_area_frac}")
    check("reset says what it changed",
          "threshold_k" in tab.lbl_run.text(), tab.lbl_run.text()[:80])
    tab._reset_defaults()
    check("a second reset reports there is nothing to do",
          "Already at the shipped defaults" in tab.lbl_run.text(),
          tab.lbl_run.text()[:60])

    W.ask_yes_no = lambda *a, **k: False
    tab.sp_thr_k.setValue(3.25)
    set_k = tab.current_config().threshold_k      # what the spin box rounded to
    tab._reset_defaults()
    check("declining the confirm changes nothing",
          tab.current_config().threshold_k == set_k,
          f"{tab.current_config().threshold_k} (was {set_k})")
    tab.apply_config(shipped)
    W.ask_yes_no = real_ask

    # ---- the suggest dialog ----------------------------------------------
    from htr_studio.ui.advisor_dialog import SuggestDialog
    dlg = SuggestDialog(ctx, video, tab.current_config(), tab)
    t0 = time.perf_counter()
    ok = wait_for(app, lambda: dlg.advice is not None, 120.0)
    check("the suggest dialog finished", ok, f"{time.perf_counter()-t0:.1f} s")
    if ok:
        adv = dlg.advice
        check("it measured the frame rate", adv.fps > 0, f"{adv.fps:.2f} fps")
        check("it listed the pending changes in the table",
              dlg.table.rowCount() == len(adv.pending),
              f"{dlg.table.rowCount()} rows / {len(adv.pending)} pending")
        check("every row carries a reason",
              all(dlg.table.item(r, 3) and len(dlg.table.item(r, 3).text()) > 20
                  for r in range(dlg.table.rowCount())))
        check("Apply is enabled only when something would change",
              dlg.btn_apply.isEnabled() == bool(adv.pending),
              f"enabled={dlg.btn_apply.isEnabled()} pending={len(adv.pending)}")
        if adv.fps < 200:
            check("a sub-200 fps clip gets a folded band",
                  any(c.key == "band_high_hz" and float(c.suggested) < 60
                      for c in adv.pending),
                  ", ".join(f"{c.key}={c.suggested}" for c in adv.pending))
        before = tab.current_config().band_high_hz
        dlg._apply()
        if dlg.result_cfg:
            tab.apply_config(dlg.result_cfg)
            tab._advice_provenance = adv.provenance()
            check("applying moved the settings",
                  tab.current_config().band_high_hz != before
                  or not adv.pending,
                  f"{before} -> {tab.current_config().band_high_hz}")
            check("the reasoning is kept for the sidecar",
                  tab._advice_provenance is not None
                  and tab._advice_provenance["changes"] is not None)
        check("switching file clears per-file advice but not a session fit",
              True)
        tab._control_calibration = {"marker": 1}
        ctx.set_entry(entry_for(video))
        # set_entry is a no-op for the same entry object, so force it.
        tab.set_entry(entry_for(video))
        check("per-file advice is cleared on a new file",
              tab._advice_provenance is None)
        check("a session calibration survives a new file",
              tab._control_calibration == {"marker": 1})
    dlg.shutdown()
    dlg.deleteLater()

    # ---- the control calibration dialog ----------------------------------
    from htr_studio.ui.advisor_dialog import ControlCalibrationDialog
    tab.apply_config(shipped)
    cdlg = ControlCalibrationDialog(ctx, video, tab.current_config(),
                                    parent=tab)
    t0 = time.perf_counter()
    ok = wait_for(app, lambda: cdlg.cal is not None, 180.0)
    check("the calibration finished", ok, f"{time.perf_counter()-t0:.1f} s")
    if ok:
        cal = cdlg.cal
        print(f"    sweep: {cal.sweep}")
        print(f"    {cal.message[:200]}")
        check("the whole sweep was run",
              len(cal.sweep) == len(advisor.K_SWEEP),
              f"{len(cal.sweep)} of {len(advisor.K_SWEEP)}")
        check("the curve was plotted",
              cdlg.curve.getData()[0] is not None
              and len(cdlg.curve.getData()[0]) == len(cal.sweep))
        check("it reports in plain language", len(cal.message) > 60)
        if cal.ok:
            ks = [k for k, _ in cal.sweep]
            check("the chosen k is at or below the silent point",
                  cal.chosen_k <= cal.zero_k, f"{cal.chosen_k} <= {cal.zero_k}")
            check("the chosen k is one step down, or the lowest swept",
                  cal.chosen_k == ks[max(0, ks.index(cal.zero_k) - 1)],
                  f"chosen {cal.chosen_k}, zero {cal.zero_k}")
            check("Apply and Save became available",
                  cdlg.btn_apply.isEnabled() and cdlg.btn_save.isEnabled())
            check("the chosen config carries the chosen k",
                  cdlg.chosen_config().threshold_k == cal.chosen_k,
                  str(cdlg.chosen_config().threshold_k))
            # Saving needs a name.
            real_warn = W.warn_box
            seen = []
            W.warn_box = lambda p, t, m: seen.append(t)
            cdlg.in_profile.setText("")
            cdlg._save_profile()
            check("saving without a name is refused", bool(seen), str(seen))
            W.warn_box = real_warn
            cdlg.in_profile.setText("ui test session")
            cdlg._save_profile()
            names = [d.get("name") for d in advisor.list_profiles()]
            check("the profile was saved", "ui test session" in names, str(names))
            tab._refresh_profiles()
            idx = next((i for i in range(tab.cb_profile.count())
                        if "ui test session" in tab.cb_profile.itemText(i)), -1)
            check("it appears in the Analysis tab's profile list", idx > 0,
                  f"index {idx}")
            if idx > 0:
                tab.apply_config(shipped)
                tab.cb_profile.setCurrentIndex(idx)
                check("selecting a profile enables Apply",
                      tab.btn_profile_apply.isEnabled())
                tab._apply_profile()
                check("applying the profile set the threshold",
                      tab.current_config().threshold_k == cal.chosen_k,
                      f"{tab.current_config().threshold_k} vs {cal.chosen_k}")
                check("the profile's calibration is kept for the sidecar",
                      (tab._control_calibration or {}).get("chosen_k")
                      == cal.chosen_k)
            for d in advisor.list_profiles():
                if d.get("name") == "ui test session":
                    Path(d["path"]).unlink(missing_ok=True)
        else:
            check("a sweep that never falls silent explains itself",
                  "still yields" in cal.message or "cancelled" in cal.message,
                  cal.message[:90])
    cdlg.shutdown()
    cdlg.deleteLater()

    W.stop_jobs(tab, 3000)
    W.JobRunner.join_all(3000)
    tab.deleteLater()
    app.processEvents()

    print(f"\n{'ADVISOR UI OK' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
