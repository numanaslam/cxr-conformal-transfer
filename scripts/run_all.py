"""Run the whole paper pipeline in order, from downloaded raw data to every result file.

One command, from the repo root (the folder holding `ovcbmr/`, `scripts/`, `configs/`):

    python scripts/run_all.py                 # everything, stops at the first failure
    python scripts/run_all.py --list          # show the stages and exit
    python scripts/run_all.py --check         # only check data + environment
    python scripts/run_all.py --from s16      # resume at a stage
    python scripts/run_all.py --only s31 s33  # run just these
    python scripts/run_all.py --no-gpu        # skip the stages that need the GPU

It does NOT download data (that needs your own Kaggle login; see RUN_FROM_SCRATCH.md). It starts
by checking that the raw files are where the configs expect them and that the GPU works, and
refuses to start if something is missing. Each stage's console output is shown live and saved to
experiments/logs/NN_<stage>.log. At the end the result files and logs are zipped into
experiments/paper6_results_<timestamp>.zip, which is the one file to send back.
"""
from __future__ import annotations
import argparse
import codecs
import csv
import datetime
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIH_IMAGES_EXPECTED = 112120
VINDR_IMAGES_EXPECTED = 15000
_W_KEYS = ("width", "original_width", "dim1", "w", "cols", "Width")
_H_KEYS = ("height", "original_height", "dim0", "h", "rows", "Height")


def build_stages(a):
    """Ordered stage list. Every command runs with the repo root as working directory."""
    N, V, B = a.nih_config, a.vindr_config, a.boxes
    loc = ["--boxes", B, "--max-images", str(a.max_images)] + (["--dims", a.dims] if a.dims else [])
    res = a.results
    S = []

    def add(name, group, argv, what, gpu=False):
        S.append({"name": name, "group": group, "argv": [sys.executable] + argv, "what": what, "gpu": gpu})

    def mod(name, group, module, cfg, what, extra=(), gpu=False):
        add(name, group, ["-m", f"scripts.{module}", "--config", cfg] + list(extra), what, gpu)

    # ---- source hospital (NIH): data prep, embeddings, source-site experiments
    add("nih_manifest", "nih", ["scripts/build_manifest_nih.py", "--csv", a.nih_csv, "--image-root", a.nih_images],
        "NIH label CSV -> manifest + concept bank")
    add("nih_check", "nih", ["scripts/s00_check_data.py", "--manifest", "data/raw/nih/labels.csv",
                             "--image-root", a.nih_images], "manifest and image sanity check")
    mod("nih_split", "nih", "s01_prepare_cxrlt", N, "patient-level train/calib/val/test split")
    mod("nih_embed", "nih", "s02_extract_embeddings", N, "frozen BiomedCLIP embeddings (long, quiet)", gpu=True)
    mod("nih_counts", "nih", "count_manifest", N, "image and patient counts per split")
    mod("s03", "nih", "s03_calibrate_concepts", N, "zero-shot concept calibration")
    mod("s04", "nih", "s04_fit_conformal", N, "zero-shot valve threshold")
    mod("s05", "nih", "s05_run_baselines", N, "novelty-detection baselines (negative result)")
    mod("s06", "nih", "s06_evaluate", N, "zero-shot valve evaluation")
    mod("s08", "nih", "s08_selective", N, "selective prediction, zero-shot predictor")
    mod("s09", "nih", "s09_train_heads", N, "train the nine linear heads + Platt scalars")
    mod("s10", "nih", "s10_interpretability", N, "per-concept AUROC / ECE, attribution")
    mod("s07", "nih", "s07_shift_demo", N, "PA vs AP view shift, weighted conformal")
    mod("s11", "nih", "s11_multigroup", N, "group-conditional coverage by view / sex / age")

    # ---- deployment hospital (VinDr): data prep, embeddings
    add("vindr_manifest", "vindr", ["scripts/build_manifest_vindr.py", "--csv", B, "--image-root", a.vindr_images,
                                    "--out", "data/raw/vindr/labels.csv"], "VinDr boxes -> per-image labels")
    add("vindr_check", "vindr", ["scripts/s00_check_data.py", "--manifest", "data/raw/vindr/labels.csv",
                                 "--image-root", a.vindr_images], "manifest and image sanity check")
    mod("vindr_split", "vindr", "s01_prepare_cxrlt", V, "study-level calib/test split")
    mod("vindr_embed", "vindr", "s02_extract_embeddings", V, "frozen BiomedCLIP embeddings", gpu=True)
    mod("vindr_counts", "vindr", "count_manifest", V, "study counts per split")

    # ---- cross-hospital experiments in the paper (cached embeddings, CPU)
    mod("s12", "cross", "s12_cross_hospital", V, "cross-hospital coverage, any-abnormal denominator")
    mod("s12b", "cross", "s12b_coverage_ci", V, "bootstrap intervals for s12")
    mod("s16", "cross", "s16_crc_cross_hospital", V, "G2 selective risk across hospitals")
    mod("s17", "cross", "s17_known_abnormal", V, "G1 on the in-vocabulary denominator", extra=["--boxes", B])
    mod("s18", "cross", "s18_sample_efficiency", V, "local calibration sample size for G1")
    mod("s19", "cross", "s19_vindr_concepts", V, "concept AUROC / ECE transfer")
    mod("s20", "cross", "s20_blackbox_baseline", V, "direct linear head vs bottleneck")
    mod("s23", "cross", "s23_review_addenda", V, "reverse G1, weighted conformal, G2 decomposition and sample size")

    # ---- analyses added in review (cached embeddings, CPU)
    mod("s24", "review", "s24_valve_ablation", V, "valve statistic vs its components")
    mod("s25", "review", "s25_fn_crc", V, "one-sided (missed-finding) risk control, both directions")
    mod("s26", "review", "s26_local_platt", V, "local Platt refit on the deployment site")
    mod("s27", "review", "s27_weighted_sweep", V, "weighted-conformal sweep + quantile alignment")
    mod("s31", "review", "s31_clean_protocol", V, "disjoint calibration data, all four settings")
    mod("s32", "review", "s32_mlp_comparator", V, "nonlinear same-feature comparator")
    mod("s33", "review", "s33_multiseed", V, "repeat over split seeds", extra=["--seeds"] + [str(s) for s in a.seeds])

    # ---- stages that re-run a network on images (GPU)
    mod("s14", "gpu", "s14_concept_saliency", V, "concept saliency figure", gpu=True)
    mod("s15", "gpu", "s15_localization", V, "pointing game vs radiologist boxes", extra=loc, gpu=True)
    mod("fig_example", "gpu", "make_fig_example", V, "real example for the method figure", gpu=True)
    mod("s30", "gpu", "s30_densenet_auroc", V, "trained DenseNet-121 AUROC (needs torchxrayvision)", gpu=True)
    mod("s29", "gpu", "s29_gradcam_baseline", V, "Grad-CAM++ localization baseline (needs torchxrayvision)",
        extra=loc, gpu=True)

    # ---- derived tables and figures
    add("s28", "figs", ["scripts/s28_localization_bonferroni.py", "--results",
                        os.path.join(res, "s15_localization.json")], "multiplicity-corrected localization table")
    mod("s13", "figs", "s13_figures", V, "coverage and score-shift figures")
    return S


# ------------------------------------------------------------------ pre-flight
def _count_png(d):
    if not os.path.isdir(d):
        return -1
    return sum(1 for e in os.scandir(d) if e.is_file() and e.name.lower().endswith(".png"))


def preflight(a, need_gpu, need_xrv, need_raw):
    """Print a checklist; return True when nothing blocking is missing."""
    ok, lines = True, []

    def item(good, text, block=True):
        nonlocal ok
        lines.append(("  ok   " if good else ("  FAIL " if block else "  warn ")) + text)
        if not good and block:
            ok = False

    for cfg in (a.nih_config, a.vindr_config):
        item(os.path.exists(os.path.join(ROOT, cfg)), f"config {cfg}")

    if need_raw:
        csv_path = os.path.join(ROOT, a.nih_csv)
        alt = os.path.join(os.path.dirname(csv_path), "Data_Entry_2017_v2020.csv")
        if not os.path.exists(csv_path) and os.path.exists(alt):
            shutil.copyfile(alt, csv_path)                       # s11 looks for this exact name
            lines.append(f"  note copied {os.path.basename(alt)} -> {os.path.basename(csv_path)}")
        item(os.path.exists(csv_path), f"NIH label CSV at {a.nih_csv}")
        n = _count_png(os.path.join(ROOT, a.nih_images))
        item(n >= NIH_IMAGES_EXPECTED, f"NIH images in {a.nih_images}: {max(n, 0)} PNG (need {NIH_IMAGES_EXPECTED})")
        v = _count_png(os.path.join(ROOT, a.vindr_images))
        item(v >= VINDR_IMAGES_EXPECTED, f"VinDr images in {a.vindr_images}: {max(v, 0)} PNG (need {VINDR_IMAGES_EXPECTED})")
        box = os.path.join(ROOT, a.boxes)
        item(os.path.exists(box), f"VinDr box CSV at {a.boxes}")
        if os.path.exists(box):
            with open(box, newline="") as f:
                header = next(csv.reader(f), [])
            need = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
            item(need.issubset(header), f"box CSV has columns {sorted(need)}")
            has_dims = any(k in header for k in _W_KEYS) and any(k in header for k in _H_KEYS)
            if a.dims:
                item(os.path.exists(os.path.join(ROOT, a.dims)), f"dims CSV at {a.dims}")
            else:
                item(has_dims, "box CSV carries original width/height (else pass --dims <csv with image_id,dim0,dim1>)",
                     block=False)
    free = shutil.disk_usage(ROOT).free / 1e9
    item(free > 5, f"free disk on this drive: {free:.0f} GB", block=False)

    try:
        import torch
        item(True, f"torch {torch.__version__}")
        if need_gpu:
            cuda = torch.cuda.is_available()
            item(cuda, "CUDA is available to torch")
            if cuda:
                name = torch.cuda.get_device_name(0)
                try:
                    x = torch.randn(8, 3, 32, 32, device="cuda")
                    torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda")).sum().item()
                    item(True, f"GPU runs a test convolution: {name}")
                except Exception as e:                            # e.g. card newer than this torch build
                    item(False, f"GPU {name} cannot run this torch build ({str(e).splitlines()[0][:90]}). "
                                "Install a torch build for your card (RTX 50 series: the cu128 wheels).")
    except Exception as e:
        item(False, f"torch import failed: {e}")
    for modname in ("open_clip", "sklearn", "yaml", "PIL", "matplotlib"):
        try:
            __import__(modname)
            item(True, f"python package {modname}")
        except Exception:
            item(False, f"python package {modname} is missing (pip install -r requirements.txt)")
    if need_xrv:
        try:
            __import__("torchxrayvision")
            item(True, "python package torchxrayvision (for s29, s30)")
        except Exception:
            item(False, "torchxrayvision missing: pip install torchxrayvision scikit-image  "
                        "(or run with --skip s29 s30)")
    print("\n".join(lines))
    return ok


# ------------------------------------------------------------------ running
def run_stage(st, log_path, env):
    """Run one stage, passing its output straight through and saving it. Returns (exit code, seconds)."""
    t0 = time.time()
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        lf.write("$ " + " ".join(st["argv"]) + "\n")
        lf.flush()
        p = subprocess.Popen(st["argv"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk:
                break
            text = dec.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            lf.write(text)
        rc = p.wait()
    return rc, time.time() - t0


def _fmt(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Run the full NIH + VinDr pipeline in order.")
    ap.add_argument("--nih-config", default="configs/nih.yaml")
    ap.add_argument("--vindr-config", default="configs/vindr.yaml")
    ap.add_argument("--nih-csv", default="data/raw/nih/Data_Entry_2017.csv")
    ap.add_argument("--nih-images", default="data/raw/nih/images")
    ap.add_argument("--vindr-images", default="data/raw/vindr/vinbigdata/train")
    ap.add_argument("--boxes", default="data/raw/vindr/vinbigdata/train.csv")
    ap.add_argument("--dims", default="", help="CSV of original image sizes, only if the box CSV lacks width/height")
    ap.add_argument("--max-images", type=int, default=1500, help="localization cohort for s15/s29 (paper: 1500)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--from", dest="start", default="", help="first stage to run")
    ap.add_argument("--to", dest="stop", default="", help="last stage to run")
    ap.add_argument("--only", nargs="+", default=[], help="run only these stages")
    ap.add_argument("--skip", nargs="+", default=[], help="skip these stages")
    ap.add_argument("--groups", nargs="+", default=[], help="run only these groups: nih vindr cross review gpu figs")
    ap.add_argument("--no-gpu", action="store_true", help="skip every stage that needs the GPU")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the pre-flight check only")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    stages = build_stages(a)
    names = [s["name"] for s in stages]
    for n in [a.start, a.stop] + a.only + a.skip:
        if n and n not in names:
            sys.exit(f"unknown stage '{n}'. Use --list to see the stage names.")

    if a.list:
        print(f"{'#':>3}  {'stage':<15}{'group':<8}{'GPU':<5}what it does")
        for i, s in enumerate(stages, 1):
            print(f"{i:>3}  {s['name']:<15}{s['group']:<8}{('yes' if s['gpu'] else ''):<5}{s['what']}")
        return

    sel = stages
    if a.only:
        sel = [s for s in sel if s["name"] in a.only]
    else:
        if a.start:
            sel = sel[names.index(a.start):]
        if a.stop:
            sel = [s for s in sel if names.index(s["name"]) <= names.index(a.stop)]
    if a.groups:
        sel = [s for s in sel if s["group"] in a.groups]
    if a.no_gpu:
        sel = [s for s in sel if not s["gpu"]]
    sel = [s for s in sel if s["name"] not in a.skip]
    if not sel:
        sys.exit("no stages selected.")

    if not a.skip_preflight:
        print("=== pre-flight check ===")
        good = preflight(a, need_gpu=any(s["gpu"] for s in sel),
                         need_xrv=any(s["name"] in ("s29", "s30") for s in sel),
                         need_raw=any(s["name"] in ("nih_manifest", "vindr_manifest", "nih_embed", "vindr_embed")
                                      for s in sel))
        if a.check:
            print("\npre-flight " + ("passed." if good else "FAILED: fix the lines marked FAIL, then rerun."))
            sys.exit(0 if good else 1)
        if not good and not a.dry_run:
            sys.exit("\npre-flight FAILED: fix the lines marked FAIL (see RUN_FROM_SCRATCH.md), then rerun.")
        print()

    logs = os.path.join(ROOT, "experiments", "logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    summary, failed = [], False
    t_all = time.time()
    for s in sel:
        idx = names.index(s["name"]) + 1
        print(f"\n{'=' * 78}\n[{idx}/{len(names)}] {s['name']}: {s['what']}\n$ python {' '.join(s['argv'][1:])}\n{'=' * 78}")
        if a.dry_run:
            summary.append((s["name"], "dry-run", 0.0))
            continue
        rc, dt = run_stage(s, os.path.join(logs, f"{idx:02d}_{s['name']}.log"), env)
        summary.append((s["name"], "ok" if rc == 0 else f"FAILED (exit {rc})", dt))
        if rc != 0:
            failed = True
            print(f"\n[run_all] stage '{s['name']}' failed. Log: experiments/logs/{idx:02d}_{s['name']}.log")
            if not a.keep_going:
                print(f"[run_all] after fixing it, resume with:  python scripts/run_all.py --from {s['name']}")
                break

    print(f"\n{'=' * 78}\nsummary ({_fmt(time.time() - t_all)} total)")
    text = "\n".join(f"  {n:<15}{status:<22}{_fmt(dt)}" for n, status, dt in summary)
    print(text)
    if a.dry_run:
        return
    with open(os.path.join(logs, "run_summary.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n--- {datetime.datetime.now():%Y-%m-%d %H:%M} ---\n{text}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(ROOT, "experiments", f"paper6_results_{stamp}.zip")
    res_dir = os.path.join(ROOT, a.results)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for label, base in (("results", res_dir), ("logs", logs)):
            if not os.path.isdir(base):
                continue
            for dp, _, fs in os.walk(base):
                for fn in fs:
                    full = os.path.join(dp, fn)
                    z.write(full, os.path.join(label, os.path.relpath(full, base)))
    print(f"\n[run_all] results + logs zipped -> {os.path.relpath(out, ROOT)}  (send this one file back)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
