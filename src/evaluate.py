"""Metrics for one predictions file, side by side with the base paper.

  python evaluate.py ../results/pred_zero_t0.0_flakebench.csv --paper flakebench
  python evaluate.py ../results/pred_<name>.csv --paper idoft

Prints, per run and for the majority vote:
  prec/rec/f1  - for the flaky class (what Berndt et al. report in Tables III/IV;
                 their "All flaky" row matches these, not weighted scores)
  f1_w         - weighted F1 (what their text mentions, for completeness)
  mcc          - their main metric
plus All-flaky / Random baselines, the base paper's GPT-OSS-120B rows, recall per
flakiness category, the normalized Hamming distance between runs (their RQ2), and
token usage.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

# Berndt et al., SANER 2026, Tables III and IV: GPT-OSS-120B, iteration 1.
PAPER = {
    "flakebench": [
        ("paper GPT-OSS zero", 0.10, 0.53, 0.17, 0.17),
        ("paper GPT-OSS zero-CoT", 0.06, 0.79, 0.11, 0.14),
        ("paper GPT-OSS few-CoT", 0.06, 0.88, 0.11, 0.15),
    ],
    "idoft": [
        ("paper GPT-OSS zero", 0.90, 0.29, 0.44, 0.10),
        ("paper GPT-OSS zero-CoT", 0.87, 0.58, 0.70, 0.06),
        ("paper GPT-OSS few-CoT", 0.91, 0.75, 0.82, 0.27),
    ],
}


def metrics(y, p):
    return {
        "n": len(y),
        "acc": accuracy_score(y, p),
        "prec": precision_score(y, p, pos_label=1, zero_division=0),
        "rec": recall_score(y, p, pos_label=1, zero_division=0),
        "f1": f1_score(y, p, pos_label=1, zero_division=0),
        "f1_w": f1_score(y, p, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv")
    ap.add_argument("--paper", choices=PAPER, help="show the base paper's numbers")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    unparsed = int(df.pred.isna().sum())
    if unparsed:
        print(f"warning: {unparsed} responses had no parsable label (counted as non-flaky)")
    df["pred"] = df.pred.fillna(0).astype(int)
    wide = df.pivot(index="sample_id", columns="run", values="pred").dropna()
    first = df.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index]
    y = first.label.astype(int)
    print(f"{len(y)} tests ({int(y.sum())} flaky, {int((y == 0).sum())} non-flaky), "
          f"{len(wide.columns)} run(s)\n")

    rows = [{"setting": f"run {r}", **metrics(y, wide[r])} for r in wide.columns]
    if len(wide.columns) > 1:
        vote = (wide.mean(axis=1) > 0.5).astype(int)  # ties -> non-flaky
        rows.append({"setting": "majority vote", **metrics(y, vote)})
    rows.append({"setting": "baseline: all flaky", **metrics(y, np.ones(len(y), int))})
    rng = np.random.default_rng(0)
    rows.append({"setting": "baseline: random", **metrics(y, rng.integers(0, 2, len(y)))})
    table = pd.DataFrame(rows)
    if args.paper:
        ref = pd.DataFrame(PAPER[args.paper],
                           columns=["setting", "prec", "rec", "f1", "mcc"])
        table = pd.concat([table, ref], ignore_index=True)
    print(table.round(3).to_string(index=False, na_rep=""))

    if "category" in first.columns and first.category.notna().any():
        print("\nRecall per flakiness category (run 0):")
        flaky = first[y == 1]
        hit = wide.loc[flaky.index, wide.columns[0]] == 1
        for cat, g in hit.groupby(flaky.category):
            print(f"  {cat:<28} {g.mean():.2f}  ({int(g.sum())}/{len(g)})")

    if len(wide.columns) > 1:
        print("\nNormalized Hamming distance between runs (paper RQ2):")
        for a, b in itertools.combinations(wide.columns, 2):
            d = wide[a] != wide[b]
            print(f"  H(run{a}, run{b}) = {d.mean():.3f}  ({int(d.sum())} tests)")

    if "prompt_tokens" in df.columns:
        print(f"\nTokens: {int(df.prompt_tokens.sum()):,} input, "
              f"{int(df.completion_tokens.sum()):,} output over {len(df)} calls")


if __name__ == "__main__":
    main()
