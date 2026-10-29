"""Check out each (repo, commit) pair needed by the samples, fetching only that commit.

Usage:
    python src/fetch_repos.py data/processed/idoft_samples.csv
The CSV needs columns: repo_url, sha. Failures are written to results/fetch_failures.csv
(report the count in the paper as an exclusion criterion).
"""
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import REPOS_DIR, RESULTS_DIR


def repo_dir(repo_url: str, sha: str):
    owner_repo = repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    return REPOS_DIR / f"{owner_repo.replace('/', '__')}__{sha[:12]}"


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)


def fetch(repo_url: str, sha: str):
    target = repo_dir(repo_url, sha)
    if (target / ".fetched_ok").exists():
        return repo_url, sha, "cached", ""
    target.mkdir(parents=True, exist_ok=True)
    steps = [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", repo_url],
        ["git", "fetch", "-q", "--depth", "1", "--filter=blob:limit=2m", "origin", sha],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ]
    for cmd in steps:
        r = _run(cmd, target)
        if r.returncode != 0 and "already exists" not in r.stderr:
            return repo_url, sha, "failed", r.stderr.strip()[:300]
    (target / ".fetched_ok").write_text(sha)
    return repo_url, sha, "ok", ""


def main(csv_path: str, workers: int = 4):
    df = pd.read_csv(csv_path)
    pairs = df[["repo_url", "sha"]].drop_duplicates().itertuples(index=False)
    failures, ok = [], 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, p.repo_url, p.sha) for p in pairs]
        for f in as_completed(futs):
            url, sha, status, err = f.result()
            print(f"[{status}] {url} @ {sha[:10]} {err}")
            if status == "failed":
                failures.append((url, sha, err))
            else:
                ok += 1
    out = RESULTS_DIR / "fetch_failures.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repo_url", "sha", "error"])
        w.writerows(failures)
    print(f"\n{ok} checked out, {len(failures)} failed (see {out})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
