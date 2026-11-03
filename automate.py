"""
automate.py - run the whole project with one command.

    python automate.py all                 # everything below, in order, skipping finished steps
    python automate.py all --zip ~/Downloads/LCORE-D.zip
    python automate.py fetch  [--zip PATH] # get the dataset into dataset/lcore_d/
    python automate.py configure [--yes]   # inspect columns, propose dataset_config.json
    python automate.py train  [--force]    # train + evaluate (+ run tests)
    python automate.py serve  [--auto-monitor 30] [--open]

What is automated, and what deliberately is NOT:
  * Download: best effort. Mendeley's file API is not officially documented,
    so if it fails the script looks for an already-downloaded zip in your
    Downloads folder, and finally prints the manual steps.
  * Label column: the script PROPOSES one and shows you its values. It only
    writes it without asking when you pass --yes AND there is exactly one
    obvious candidate. Picking the wrong label column silently would give
    you a model that predicts nonsense with a confident accuracy number.
  * Training is skipped when the model is newer than the data + config
    (like `make`). Use --force to retrain.
  * No real data? The pipeline falls back to the SYNTHETIC demo, loudly,
    and the dashboard shows the synthetic banner. --no-demo disables that.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

DATASET_ID = "77sztrg5ks"
DATASET_PAGE = f"https://data.mendeley.com/datasets/{DATASET_ID}"
TARGET_DIR = os.path.join("dataset", "lcore_d")
CONFIG = "dataset_config.json"
REAL_MODEL = os.path.join("model", "fault_model.joblib")
MAX_EXTRACT_BYTES = 8 * 1024 ** 3  # refuse zip bombs larger than 8 GB unpacked
PY = sys.executable


def banner(msg, char="="):
    line = char * 72
    print(f"\n{line}\n{msg}\n{line}")


def run(cmd, check=True):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check)


# --------------------------------------------------------------------------- #
# 0. Environment check
# --------------------------------------------------------------------------- #
def check_env():
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10+ required, you have {sys.version.split()[0]}")
    missing = []
    for mod in ("flask", "sklearn", "pandas", "numpy", "joblib", "matplotlib", "psutil"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"Missing packages: {missing}\nRun:  {PY} -m pip install -r requirements.txt")
    print("Environment OK:", sys.version.split()[0])


# --------------------------------------------------------------------------- #
# 1. Fetch
# --------------------------------------------------------------------------- #
def real_csvs():
    files = glob.glob(os.path.join("dataset", "**", "*.csv"), recursive=True)
    return sorted(f for f in files if f"{os.sep}synthetic{os.sep}" not in f)


def extract_once(zip_path, dest):
    """Extract unless this exact zip (path, size, mtime) was already extracted.
    Re-extracting would refresh file times and force needless retraining."""
    st = os.stat(zip_path)
    stamp = f"{os.path.abspath(zip_path)}|{st.st_size}|{int(st.st_mtime)}"
    marker = os.path.join(dest, ".extracted_from")
    if os.path.exists(marker) and open(marker, encoding="utf-8").read() == stamp:
        print(f"{zip_path} already extracted - skipping")
        return
    os.makedirs(dest, exist_ok=True)
    safe_extract(zip_path, dest)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(stamp)


def safe_extract(zip_path, dest):
    """Extract a zip without allowing '../' paths or absurd sizes."""
    dest_abs = os.path.abspath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > MAX_EXTRACT_BYTES:
            raise RuntimeError(f"zip would unpack to {total / 1e9:.1f} GB - refusing")
        for info in zf.infolist():
            target = os.path.abspath(os.path.join(dest, info.filename))
            if not target.startswith(dest_abs + os.sep) and target != dest_abs:
                raise RuntimeError(f"unsafe path in zip: {info.filename}")
        zf.extractall(dest)
    # nested zips (Mendeley often zips per folder)
    for inner in glob.glob(os.path.join(dest, "**", "*.zip"), recursive=True):
        sub = os.path.splitext(inner)[0]
        os.makedirs(sub, exist_ok=True)
        safe_extract(inner, sub)
        os.remove(inner)


def _http_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "network-fault-detection-student-project"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def try_mendeley_api(version):
    """Best-effort download via Mendeley Data's public API (undocumented -> may change)."""
    url = (f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
           f"?folder_id=root&version={version}")
    print("Trying Mendeley API:", url)
    items = _http_json(url)
    if not isinstance(items, list) or not items:
        raise RuntimeError("API answered but returned no file list")
    os.makedirs(TARGET_DIR, exist_ok=True)
    got = 0
    for item in items:
        name = os.path.basename(str(item.get("filename", "")))
        link = (item.get("content_details") or {}).get("download_url")
        if not name or not link:
            continue
        out = os.path.join(TARGET_DIR, name)
        print(f"  downloading {name} ...")
        req = urllib.request.Request(link, headers={"User-Agent": "network-fault-detection-student-project"})
        with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as fh:
            shutil.copyfileobj(r, fh)
        got += 1
        if name.lower().endswith(".zip"):
            safe_extract(out, os.path.splitext(out)[0])
            os.remove(out)
    if not got:
        raise RuntimeError("file list had no downloadable entries (sub-folders are not followed)")


def find_downloaded_zip():
    """Look for a zip the user already downloaded from the Mendeley page."""
    home = os.path.expanduser("~")
    pats = ["*LCORE*.zip", "*lcore*.zip", f"*{DATASET_ID}*.zip"]
    for folder in (os.path.join(home, "Downloads"), home, "dataset"):
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(folder, pat)), key=os.path.getmtime, reverse=True)
            if hits:
                return hits[0]
    return None


def manual_instructions():
    banner("MANUAL DOWNLOAD NEEDED", "!")
    print(f"""1. Open {DATASET_PAGE} in your browser (use the newest version).
2. Click 'Download All' - you get a .zip file.
3. Run:   {os.path.basename(PY)} automate.py all --zip PATH/TO/THAT.zip
   (or extract the CSV files into  {TARGET_DIR}/  yourself)
4. Read the dataset description: note the label column and what its values mean.""")


def fetch(zip_path=None, version=2, allow_network=True):
    if zip_path:
        if not os.path.exists(zip_path):
            sys.exit(f"zip not found: {zip_path}")
        print(f"Extracting {zip_path} -> {TARGET_DIR}")
        try:
            extract_once(zip_path, TARGET_DIR)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            sys.exit(f"Refusing to use {zip_path}: {exc}")
    elif real_csvs():
        print(f"Real CSV data already present ({len(real_csvs())} file(s)) - skipping download")
        return True
    else:
        local = find_downloaded_zip()
        if local:
            print(f"Found a downloaded zip: {local}")
            try:
                extract_once(local, TARGET_DIR)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                print(f"Could not use {local}: {exc}")
        elif allow_network:
            try:
                try_mendeley_api(version)
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
                print(f"Automatic download failed: {exc}")
    csvs = real_csvs()
    others = [f for f in glob.glob(os.path.join(TARGET_DIR, "**", "*"), recursive=True)
              if os.path.isfile(f) and not f.lower().endswith(".csv")]
    if others:
        print(f"Note: {len(others)} non-CSV file(s) present (only CSV is used), e.g. {others[:3]}")
    if csvs:
        print(f"Real data ready: {len(csvs)} CSV file(s)")
        return True
    manual_instructions()
    return False


# --------------------------------------------------------------------------- #
# 2. Configure
# --------------------------------------------------------------------------- #
def load_cfg():
    with open(CONFIG, encoding="utf-8") as fh:
        return json.load(fh)


def ask(prompt, default=False):
    if not sys.stdin.isatty():
        return default
    ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not ans else ans.startswith("y")


def choose(prompt, options):
    if not sys.stdin.isatty():
        return None
    for i, o in enumerate(options, 1):
        print(f"   {i}. {o}")
    ans = input(f"{prompt} (number, or Enter to skip): ").strip()
    return options[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(options) else None


def configure(assume_yes=False, reconfigure=False):
    from inspect_dataset import LABEL_HINTS, inspect
    from preprocess import _parse_time

    cfg = load_cfg()
    if cfg.get("label_column") and not reconfigure:
        print(f"dataset_config.json already has label_column='{cfg['label_column']}' "
              "(use --reconfigure to redo)")
        return True

    report = inspect("dataset")
    df = report.pop("_df")
    sug = report["suggestions"]

    # --- label ------------------------------------------------------------- #
    strong = [c for c in sug["label_candidates"] if LABEL_HINTS.search(c)]
    label = None
    if len(strong) == 1:
        label = strong[0]
        print(f"\nProposed label column: '{label}'  values: {report['columns'][label].get('value_counts')}")
        if not assume_yes and not ask("Is this the fault label described in the dataset docs?", False):
            label = None
    if label is None and sug["label_candidates"]:
        if assume_yes:
            print(f"\n--yes given but the label is ambiguous: {strong or sug['label_candidates']}. "
                  "Not guessing. Set label_column in dataset_config.json by hand.")
            return False
        print("\nWhich column is the fault label?")
        label = choose("Label column", sug["label_candidates"])
    if not label:
        print("No label column chosen - edit dataset_config.json by hand, then run  automate.py train")
        return False

    # --- time / group -------------------------------------------------------- #
    time_col = None
    for c in sug["time_candidates"]:
        try:
            _parse_time(df[c])
            time_col = c
            break
        except ValueError:
            continue
    group_col = None
    if not time_col:
        groups = [c for c in sug["group_candidates"] if c != label]
        group_col = groups[0] if groups else None

    # --- leakage guard: other label-like columns get dropped ---------------- #
    leaky = [c for c in strong if c != label]
    drop = sorted(set(sug["id_like_columns"]) | set(leaky))

    proposal = dict(cfg, label_column=label, time_column=time_col, group_column=group_col,
                    drop_columns=drop, data_path="dataset", split_strategy="auto")
    banner("PROPOSED dataset_config.json")
    print(json.dumps({k: v for k, v in proposal.items() if not k.startswith("_")}, indent=2))
    if leaky:
        print(f"\nDropping {leaky}: they look like other label columns -> would leak the answer.")
    if not time_col and not group_col:
        print("\nWARNING: no time or group column found -> random split -> accuracy may be inflated.")
    if assume_yes or ask("Write this config?", True):
        with open(CONFIG, "w", encoding="utf-8") as fh:
            json.dump(proposal, fh, indent=2)
        print(f"Saved {CONFIG}")
        return True
    return False


# --------------------------------------------------------------------------- #
# 3. Train
# --------------------------------------------------------------------------- #
def model_is_stale():
    if not os.path.exists(REAL_MODEL):
        return True
    newest_input = max([os.path.getmtime(f) for f in real_csvs()] + [os.path.getmtime(CONFIG)])
    return newest_input > os.path.getmtime(REAL_MODEL)


def train(force=False, synthetic=False, tests=True):
    if synthetic:
        if not os.path.exists("dataset/synthetic/SYNTHETIC_network_data.csv"):
            run([PY, "generate_synthetic.py"])
        if force or not os.path.exists("model/synthetic/fault_model.joblib"):
            run([PY, "train_model.py", "--synthetic"])
        else:
            print("Synthetic model already trained - skipping")
    elif force or model_is_stale():
        run([PY, "train_model.py"])
    else:
        print("Model is up to date with data + config - skipping training (use --force to retrain)")
    if tests:
        run([PY, "-m", "pytest", "-q", "tests"], check=False)


# --------------------------------------------------------------------------- #
# 4. Serve
# --------------------------------------------------------------------------- #
def serve(port=5000, auto_monitor=30, host_target="8.8.8.8", open_browser=False):
    cmd = [PY, "app.py", "--port", str(port), "--auto-monitor", str(auto_monitor),
           "--host-target", host_target]
    url = f"http://127.0.0.1:{port}"
    banner(f"Dashboard: {url}   (Ctrl+C to stop)")
    proc = subprocess.Popen(cmd)
    if open_browser:
        for _ in range(30):
            try:
                urllib.request.urlopen(url, timeout=1)
                webbrowser.open(url)
                break
            except OSError:
                time.sleep(0.5)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_fetch(p):
        p.add_argument("--zip", help="path to the dataset zip you downloaded")
        p.add_argument("--version", type=int, default=2, help="Mendeley dataset version")
        p.add_argument("--offline", action="store_true", help="don't try the network")

    def common_serve(p):
        p.add_argument("--port", type=int, default=5000)
        p.add_argument("--auto-monitor", type=int, default=30, metavar="SECONDS", help="0 = off")
        p.add_argument("--host-target", default="8.8.8.8")
        p.add_argument("--open", action="store_true", help="open the browser")

    p_all = sub.add_parser("all", help="fetch -> configure -> train -> test -> serve")
    common_fetch(p_all)
    common_serve(p_all)
    p_all.add_argument("--yes", action="store_true", help="accept an unambiguous config proposal")
    p_all.add_argument("--force", action="store_true", help="retrain even if up to date")
    p_all.add_argument("--no-demo", action="store_true", help="don't fall back to synthetic data")
    p_all.add_argument("--no-serve", action="store_true", help="stop after training (e.g. for CI)")

    p_fetch = sub.add_parser("fetch")
    common_fetch(p_fetch)
    p_conf = sub.add_parser("configure")
    p_conf.add_argument("--yes", action="store_true")
    p_conf.add_argument("--reconfigure", action="store_true")
    p_train = sub.add_parser("train")
    p_train.add_argument("--force", action="store_true")
    p_train.add_argument("--synthetic", action="store_true")
    p_train.add_argument("--no-tests", action="store_true")
    p_serve = sub.add_parser("serve")
    common_serve(p_serve)
    args = ap.parse_args()

    check_env()
    if args.cmd == "fetch":
        sys.exit(0 if fetch(args.zip, args.version, not args.offline) else 1)
    if args.cmd == "configure":
        sys.exit(0 if configure(args.yes, args.reconfigure) else 1)
    if args.cmd == "train":
        train(args.force, args.synthetic, not args.no_tests)
        return
    if args.cmd == "serve":
        serve(args.port, args.auto_monitor, args.host_target, args.open)
        return

    # ---- all ---- #
    banner("STEP 1/4  dataset")
    have_real = fetch(args.zip, args.version, not args.offline)
    configured = False
    if have_real:
        banner("STEP 2/4  configure")
        configured = configure(args.yes)
    if have_real and configured:
        banner("STEP 3/4  train on REAL data")
        train(args.force)
    elif args.no_demo:
        sys.exit("Stopping: real data not ready and --no-demo was given.")
    else:
        why = "not downloaded" if not have_real else "label column not configured"
        banner(f"STEP 3/4  REAL DATA {why.upper()} -> SYNTHETIC DEMO ONLY\n"
               "Results below are NOT project results. Fix the steps above and re-run.", "!")
        train(args.force, synthetic=True)
    if args.no_serve:
        return
    banner("STEP 4/4  serve")
    serve(args.port, args.auto_monitor, args.host_target, args.open)


if __name__ == "__main__":
    main()
