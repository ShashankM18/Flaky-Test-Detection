# flaky-rag

Retrieval-augmented LLM classification of flaky tests, with self-consistency.
Extends Berndt et al., "Can We Classify Flaky Tests Using Only Test Code?" (SANER 2026).

## 1. Setup (Ubuntu, or WSL2 on Windows)

```bash
sudo apt update && sudo apt install -y git python3.11 python3.11-venv
cd flaky-rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenRouter key into .env
python src/check_setup.py   # must print two [ok] lines
```

On Windows, keep the project inside the WSL filesystem (`~/flaky-rag`), not under `/mnt/c/`,
or git and file scanning will be very slow.

### OpenRouter settings (`.env`)
- `LLM_MODEL=openai/gpt-oss-120b`: the base paper's best model.
- `LLM_PROVIDER`: pin one hosting provider (slug from the model's Providers tab on
  openrouter.ai) so all experiments run on the same backend. The code disables
  fallbacks, so a provider outage fails loudly instead of silently switching.
- `LLM_REASONING_EFFORT`: keep fixed for all runs; report it in the paper.
- Add a spending limit on your OpenRouter key before running large experiments.

## 2. Datasets

| Dataset | Where | What you need from it |
|---|---|---|
| IDoFT `pr-data.csv` | github.com/TestingResearchIllinois/idoft (already in `data/raw/idoft/`) | repo URL, commit SHA, module path, test name |
| Base paper replication package | Zenodo DOI 10.5281/zenodo.18311270 | the exact 3,813-sample IDoFT subset, prompts, few-shot examples, their predictions |
| FlakeBench | github.com/UT-SE-Research/FlakyLens, Zenodo DOI 10.5281/zenodo.15761937 | 8,574 tests with labels; check that each test has a repo URL + commit SHA |

Put downloads under `data/raw/<name>/`. The labeled IDoFT subset comes from the
replication package; `prepare_idoft.py join` attaches repo URL + SHA to it.

## 3. Pipeline

```bash
cd src
# a) build a sample list
python prepare_idoft.py pilot ../data/raw/idoft/pr-data.csv 20          # quick pilot
python prepare_idoft.py join ../data/raw/idoft/pr-data.csv <labeled.csv> "<test column>"

# b) check out each repo at the right commit (only that commit is downloaded)
python fetch_repos.py ../data/processed/idoft_samples.csv 4

# c) inspect prompts without spending money
python classify.py ../data/processed/idoft_samples.csv --prompt ctx --dry-run --limit 2

# d) run experiments
python classify.py ../data/processed/idoft_samples.csv --prompt zero_cot --runs 3
python classify.py ../data/processed/idoft_samples.csv --prompt ctx \
       --context fields,setup_teardown,helpers --runs 3
python classify.py ../data/processed/idoft_samples.csv --prompt ctx --runs 5 --temperature 0.7  # self-consistency

# e) metrics (per run, majority vote, Hamming distance)
python evaluate.py ../results/pred_<experiment>.csv
```

All LLM responses are cached in `results/llm_cache/`; re-running never pays twice.

## 4. Layout
```
src/config.py         paths + .env settings
src/llm_client.py     OpenRouter calls, provider pinning, retries, cache, label parsing
src/prepare_idoft.py  sample lists with repo coordinates
src/fetch_repos.py    single-commit checkouts, failure log
src/java_parser.py    Tree-sitter: test method, setup/teardown, fields, helpers, callees
src/prompts.py        base-paper style prompts + context prompt
src/classify.py       run one configuration
src/evaluate.py       MCC, weighted P/R/F1, majority vote, Hamming distance
```

## 5. Rules
- Context is always taken at "SHA Detected" (before the fix). Never check out the fix commit.
- Run the baseline yourself with the same model/provider; do not compare against copied numbers.
- Report fetch failures and unparseable tests as exclusions.
