#!/usr/bin/env python3
"""Download the OceanDepths files used by the ocean analysis agent at a pinned revision, then verify and record them.

Kept: profile indices, metadata, examples, land mask and the full enriched profile store
`data/argo_glors_ostia_ssh.zarr`. Excluded: uint8 GeoTIFF rasters, the compact grid store and loader caches,
which the design rules out as inputs for precise computation.

hf-mirror.com serves only small git files for this repository; data files redirect to HF's Xet storage, so this
script uses the shell's normal (proxied) route to huggingface.co. The store has ~59K small files and the Hub
rate-limits anonymous IPs (429 after ~1.5K files in 3 minutes), so files are fetched one by one at a capped rate,
waiting out 429s and retrying transient network errors (e.g. proxy drops); files already present with the right
size are skipped, so reruns resume. A HF_TOKEN in the
environment raises the limit. Afterwards every expected file's size is checked against the Hub listing and
SOURCE_MANIFEST.json (path, size, sha256) is written for provenance.
"""
import argparse
import fnmatch
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ESA-philab/OceanDepths"
REVISION = "d8c813ebd6126066ebe8a13b4c6bae769c8e0a07"
PATTERNS = ["README.md", "LICENSE", "manifest.yaml", "requirements-loader.txt", "indices/*", "metadata/*",
            "examples/*", "masks/*", "data/argo_glors_ostia_ssh.zarr/*"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/data/anhuang/oceandepths"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--files_per_minute", type=float, default=200)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    import requests
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    listing = args.output / f"expected_files_{REVISION[:7]}.json"  # cached so resumed runs skip ~60 listing calls
    if listing.exists():
        expected = json.loads(listing.read_text())
    else:
        expected = {}
        for item in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, recursive=True):
            if type(item).__name__ == "RepoFile" and any(fnmatch.fnmatch(item.path, p) for p in PATTERNS):
                expected[item.path] = item.size
        args.output.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(expected))
    print(f"{len(expected)} files, {sum(expected.values()) / 1e9:.2f} GB expected", flush=True)
    if not args.verify_only:
        todo = [p for p, size in expected.items()
                if not (args.output / p).is_file() or (args.output / p).stat().st_size != size]
        print(f"{len(todo)} files to fetch", flush=True)
        lock, state = threading.Lock(), {"next": time.monotonic(), "done": 0, "waits": 0}
        started = time.time()

        def transient(path, exc):
            with lock:
                state["errors"] = state.get("errors", 0) + 1
                count = state["errors"]
            if count % 20 == 1:
                print(f"transient error #{count} on {path}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            time.sleep(min(300, 15 * 2 ** min(state.get("errors", 0) % 6, 4)))

        def fetch(path):
            while True:
                with lock:  # global pacing across workers
                    delay = state["next"] - time.monotonic()
                    state["next"] = max(state["next"], time.monotonic()) + 60 / args.files_per_minute
                if delay > 0:
                    time.sleep(delay)
                try:
                    hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=args.output)
                    break
                except HfHubHTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 429:
                        retry = exc.response.headers.get("Retry-After", "")
                        with lock:
                            state["waits"] += 1
                            state["next"] = time.monotonic() + (float(retry) if retry.isdigit() else 120)
                        continue
                    if status is not None and status < 500:
                        raise
                    transient(path, exc)
                except (requests.exceptions.RequestException, OSError) as exc:  # proxy drops, timeouts, resets
                    transient(path, exc)
            with lock:
                state["done"] += 1
                if state["done"] % 500 == 0:
                    print(f"{state['done']}/{len(todo)} files, {(time.time() - started) / 60:.1f} min, "
                          f"{state['waits']} rate-limit waits", flush=True)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(fetch, todo))
        print(f"download finished in {(time.time() - started) / 60:.1f} min", flush=True)

    wrong = [p for p, size in expected.items()
             if not (args.output / p).is_file() or (args.output / p).stat().st_size != size]
    if wrong:
        raise SystemExit(f"{len(wrong)} files missing or with the wrong size, e.g. {wrong[:5]}; rerun to resume")
    with ThreadPoolExecutor(args.workers) as pool:
        hashes = dict(zip(expected, pool.map(lambda p: sha256(args.output / p), expected)))
    manifest = {"repo": REPO, "revision": REVISION, "patterns": PATTERNS, "license": "CC BY 4.0",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {p: {"size": expected[p], "sha256": hashes[p]} for p in sorted(expected)}}
    (args.output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"verified {len(expected)} files; wrote {args.output / 'SOURCE_MANIFEST.json'}", flush=True)


if __name__ == "__main__":
    main()
