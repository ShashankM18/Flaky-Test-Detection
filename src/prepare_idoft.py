"""Build sample lists that carry repository coordinates (repo_url, sha, module_path, test_fqn).

Two modes:

1) pilot   - a small set of IDoFT flaky Java tests (Status=Accepted) straight from
             IDoFT's pr-data.csv. Use it to test fetching/parsing before you have the
             labeled benchmark. It has no non-flaky samples, so no metrics.
                 python src/prepare_idoft.py pilot data/raw/idoft/pr-data.csv 20

2) join    - attach repo coordinates to the labeled benchmark (e.g. the 3813-sample
             IDoFT subset from the base paper's replication package). Pass the labeled
             CSV and the name of its column holding the fully-qualified test name.
                 python src/prepare_idoft.py join data/raw/idoft/pr-data.csv \
                     data/raw/berndt/idoft_subset.csv "<test-name column>"
             Rows that cannot be matched are written to *_unmatched.csv.

LEAKAGE NOTE: context is always retrieved at "SHA Detected" (before the fix). For a fixed
(non-flaky) sample, use the fixed test code from the labeled dataset as test_code, but never
check out the post-fix commit for context.
"""
import sys

import pandas as pd

from config import DATA_PROCESSED

COLS = {
    "Project URL": "repo_url",
    "SHA Detected": "sha",
    "Module Path": "module_path",
    "Fully-Qualified Test Name (packageName.ClassName.methodName)": "test_fqn",
    "Category": "idoft_category",
    "Status": "status",
    "PR Link": "pr_link",
}


def load_idoft(path):
    df = pd.read_csv(path).rename(columns=COLS)
    df["module_path"] = df["module_path"].fillna(".")
    return df[list(COLS.values())]


def pilot(idoft_csv, n=20, seed=42):
    df = load_idoft(idoft_csv)
    df = df[df.status == "Accepted"]
    # one test per project keeps the pilot diverse and cheap to fetch
    df = df.groupby("repo_url").head(1).sample(n=n, random_state=seed)
    df = df.assign(label=1, sample_id=[f"pilot_{i}" for i in range(len(df))])
    out = DATA_PROCESSED / "pilot_samples.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


def join(idoft_csv, labeled_csv, test_col):
    meta = load_idoft(idoft_csv).drop_duplicates("test_fqn")
    lab = pd.read_csv(labeled_csv)
    merged = lab.merge(meta, left_on=test_col, right_on="test_fqn", how="left")
    ok = merged[merged.repo_url.notna()]
    bad = merged[merged.repo_url.isna()]
    if "sample_id" not in ok.columns:
        ok = ok.assign(sample_id=[f"idoft_{i}" for i in range(len(ok))])
    ok.to_csv(DATA_PROCESSED / "idoft_samples.csv", index=False)
    bad.to_csv(DATA_PROCESSED / "idoft_samples_unmatched.csv", index=False)
    print(f"matched {len(ok)}, unmatched {len(bad)} -> {DATA_PROCESSED}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pilot":
        pilot(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif mode == "join":
        join(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit("mode must be 'pilot' or 'join'")
