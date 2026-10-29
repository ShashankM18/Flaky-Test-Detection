"""Run one experiment configuration over a sample CSV.

Examples
  # baseline (test code only), temperature 0, 3 repetitions like the base paper
  python src/classify.py data/processed/idoft_samples.csv --prompt zero_cot --runs 3

  # structural context: fields + setup/teardown + helper methods
  python src/classify.py data/processed/idoft_samples.csv --prompt ctx \
      --context fields,setup_teardown,helpers --runs 3

  # self-consistency: 5 samples at temperature 0.7, majority vote in evaluate.py
  python src/classify.py ... --runs 5 --temperature 0.7

  # see the prompts without calling the API
  python src/classify.py ... --dry-run --limit 2

Input CSV columns: sample_id, repo_url, sha, module_path, test_fqn, [label], [test_code]
If test_code is present (e.g. fixed version of an IDoFT test) it is used as-is;
otherwise the test is extracted from the checked-out repo.
"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

import prompts
from config import RESULTS_DIR
from fetch_repos import repo_dir
from java_parser import extract

TEMPLATES = {"zero": prompts.ZERO_SHOT, "zero_cot": prompts.ZERO_SHOT_COT,
             "ctx": prompts.ZERO_SHOT_COT_CONTEXT}


def build_prompt(row, args):
    parts = None
    if args.prompt == "ctx" or not isinstance(row.get("test_code"), str):
        parts = extract(repo_dir(row["repo_url"], row["sha"]), row["test_fqn"],
                        str(row.get("module_path", ".")))
        if parts is None:
            return None, None
    code = row["test_code"] if isinstance(row.get("test_code"), str) else parts["test_code"]
    if args.prompt == "ctx":
        context = prompts.build_context(parts, tuple(args.context.split(",")))
        return TEMPLATES["ctx"].format(code=code, context=context), parts
    return TEMPLATES[args.prompt].format(code=code), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--prompt", choices=TEMPLATES, default="zero_cot")
    ap.add_argument("--context", default="fields,setup_teardown,helpers")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="", help="experiment name (default: derived)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Name excludes --runs, so raising --runs later reuses the cached runs.
    dataset = Path(args.samples).stem.replace("_samples", "")
    name = args.name or (f"{args.prompt}"
                         + (f"_{args.context.replace(',', '+')}" if args.prompt == "ctx" else "")
                         + f"_t{args.temperature}_{dataset}")
    df = pd.read_csv(args.samples)
    if args.limit:
        df = df.head(args.limit)

    jobs, skipped = [], []
    for row in df.to_dict("records"):
        p, _ = build_prompt(row, args)
        if p is None:
            skipped.append(row["sample_id"])
        else:
            jobs.append((row, p))
    print(f"{len(jobs)} prompts built, {len(skipped)} skipped (test not found in repo)")

    if args.dry_run:
        for row, p in jobs[:args.limit or 2]:
            print("=" * 80, f"\n{row['sample_id']}  ({len(p)} chars)\n", p)
        return

    from llm_client import Cache, ask  # imported here so --dry-run needs no API key
    cache = Cache(name)
    out = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(ask, p, cache, row["sample_id"], args.temperature, r): (row, r)
                for row, p in jobs for r in range(args.runs)}
        for f in tqdm(as_completed(futs), total=len(futs), desc=name):
            row, r = futs[f]
            try:
                rec = f.result()
                out.append({"sample_id": row["sample_id"], "run": r,
                            "label": row.get("label"), "category": row.get("category"),
                            "pred": rec["label"],
                            "prompt_tokens": rec["usage"].get("prompt_tokens"),
                            "completion_tokens": rec["usage"].get("completion_tokens")})
            except Exception as e:  # keep going; failed calls are retried on rerun
                print(f"ERROR {row['sample_id']} run {r}: {e}")

    pred_path = RESULTS_DIR / f"pred_{name}.csv"
    pd.DataFrame(out).sort_values(["sample_id", "run"]).to_csv(pred_path, index=False)
    (RESULTS_DIR / f"skipped_{name}.json").write_text(json.dumps(skipped))
    print(f"predictions -> {pred_path}")


if __name__ == "__main__":
    main()
