"""Parallel file downloader with resume + retry + validation (stdlib only, cross-platform).

Downloads every entry in a URL list concurrently. Reusable for any large multi-file dataset
(NIH tarballs now; PadChest zip parts once you have the authenticated URLs).

List format (blank lines and '#' comments ignored):
    <url>                     # filename inferred from the URL
    <url>  <filename>         # explicit output filename

    python scripts/pdownload.py --list scripts/nih_urls.txt --out data/raw/nih --workers 4 --extract
    python scripts/pdownload.py --list scripts/nih_urls.txt --out data/raw/nih --dry-run

Notes: resumes partial files via HTTP Range; validates that archives are real gzip/zip (an
HTML error page from a stale link is caught, not silently saved); re-run to resume failures.
"""
from __future__ import annotations
import argparse
import os
import sys
import tarfile
import threading
import time
import zipfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(msg, flush=True)


def parse_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            url = parts[0]
            name = parts[1] if len(parts) > 1 else (os.path.basename(url.split("?")[0]) or "download.bin")
            items.append((url, name))
    return items


def _looks_valid(path):
    """Reject HTML error pages saved under an archive name; allow known/unknown binary."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return True
    if head[:2] == b"\x1f\x8b" or head[:2] == b"PK":   # gzip / zip
        return True
    if head[:1] == b"<":                                # HTML error page
        return False
    return True


def download_one(url, name, out, retries=4, chunk=1 << 20):
    dst = os.path.join(out, name)
    tmp = dst + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ovcbmr-pdownload"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resuming = have and getattr(r, "status", 200) == 206
                mode = "ab" if resuming else "wb"
                got = have if resuming else 0
                t0 = time.time()
                with open(tmp, mode) as fh:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        got += len(buf)
            if not _looks_valid(tmp):
                os.replace(tmp, dst + ".INVALID")
                return name, False, "content looks like HTML (stale/expired link?)"
            os.replace(tmp, dst)
            return name, True, f"{got/1e6:.0f} MB in {time.time()-t0:.0f}s"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            log(f"  [retry {attempt}/{retries}] {name}: {e}")
            time.sleep(min(30, 2 ** attempt))
    return name, False, "exhausted retries"


def extract(path, out):
    if path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as t:
            t.extractall(out)
    elif path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (3-6 is polite)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--extract", action="store_true", help="extract archives after download")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = parse_list(args.list)
    os.makedirs(args.out, exist_ok=True)
    log(f"[pdownload] {len(items)} file(s) -> {args.out} | workers={args.workers}")

    if args.dry_run:
        for url, name in items:
            have = os.path.exists(os.path.join(args.out, name))
            log(f"  {'[have] ' if have else '[queue]'} {name}  <-  {url[:64]}")
        return

    todo = [(u, n) for (u, n) in items if not os.path.exists(os.path.join(args.out, n))]
    log(f"[pdownload] {len(items) - len(todo)} already present; downloading {len(todo)}")
    ok, fail = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, u, n, args.out, args.retries): n for (u, n) in todo}
        for fut in as_completed(futs):
            name, success, msg = fut.result()
            log(f"  {'OK  ' if success else 'FAIL'} {name}: {msg}")
            (ok if success else fail).append(name)

    log(f"[pdownload] done: {len(ok)} ok, {len(fail)} failed")
    if fail:
        log(f"[pdownload] failed: {fail}  (re-run to resume; refresh stale URLs in the list)")
        sys.exit(1)
    if args.extract:
        for _, name in items:
            p = os.path.join(args.out, name)
            if os.path.exists(p) and name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                log(f"[pdownload] extracting {name}")
                extract(p, args.out)
        log("[pdownload] extraction done")


if __name__ == "__main__":
    main()
