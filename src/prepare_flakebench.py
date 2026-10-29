"""Turn FlakeBench into sample lists for classify.py.

  python prepare_flakebench.py

Reads data/raw/flakylens/FlakeBench/Flakicat_dataset_with_modules.csv and writes:
  data/processed/flakebench_all.csv    all 8,574 tests -> use this for numbers you compare
                                       with the base paper (precision and MCC depend on
                                       the real 3% flaky ratio)
  data/processed/flakebench_smoke.csv  50 flaky + 150 non-flaky -> only for checking the
                                       pipeline and cost; NOT comparable to the paper

label: 1 = flaky (any category), 0 = non-flaky. The flakiness category is kept for
per-category analysis. The test code comes from the dataset, so the test-code-only
baseline needs no repository checkout.
"""
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW

SRC = DATA_RAW / "flakylens" / "FlakeBench" / "Flakicat_dataset_with_modules.csv"
SHAS = DATA_RAW / "flakylens" / "FlakeBench" / "filtered_tests_with_owner_sha.csv"


def main():
    df = pd.read_csv(SRC)
    out = pd.DataFrame({
        "sample_id": [f"fb_{i}" for i in range(len(df))],
        "project": df["project"],
        "module_path": df["module"].fillna("."),
        "test_name": df["test_name"],
        "test_code": df["full_code"],
        "category": df["label"],
        "label": (df["label"] != "non-flaky").astype(int),
    })

    # Repo coordinates for the RAG step later. Some projects list several commits;
    # the right one per test is resolved after checkout by matching the test code.
    shas = pd.read_csv(SHAS, header=None, names=["owner_repo", "shas"])
    shas["project"] = shas.owner_repo.str.replace("/", "_", regex=False)
    out = out.merge(shas, on="project", how="left")
    out["repo_url"] = "https://github.com/" + out.owner_repo
    unmatched = out.owner_repo.isna().sum()

    out.to_csv(DATA_PROCESSED / "flakebench_all.csv", index=False)
    smoke = pd.concat([out[out.label == 1].sample(50, random_state=1),
                       out[out.label == 0].sample(150, random_state=1)])
    smoke.to_csv(DATA_PROCESSED / "flakebench_smoke.csv", index=False)
    print(f"{len(out)} tests ({out.label.sum()} flaky). "
          f"{unmatched} without a repo match (fine for the baseline).")
    print(out.category.value_counts().to_string())


if __name__ == "__main__":
    main()
