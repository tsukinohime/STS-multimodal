# STS-experiments

Research question: **when measuring Semantic Textual Similarity (STS) between two
texts, can images generated from each text (used as a caption in an image
generation model) provide extra semantic information that helps the STS task?**

This repo starts from a **text-only reference** (Milestone 1) and will later add
the generated-image (visual) modality on top of the same pipeline.

## Milestone 1 — text-only zero-shot STS

Pipeline: `text pair` → pre-trained text encoder → embeddings → **cosine
similarity** → correlate predictions with human gold scores (**Spearman** =
headline, **Pearson** secondary).

Currently wired datasets:

| name       | source                                             | gold           |
|------------|----------------------------------------------------|----------------|
| `sick`     | `data/raw/sick/SICK.txt` (SemEval_set = TEST)       | relatedness 1–5 |
| `cxc-val`  | `data/raw/cxc/data/sts_val.csv`  + Karpathy COCO    | agg_score 0–5  |
| `cxc-test` | `data/raw/cxc/data/sts_test.csv` + Karpathy COCO    | agg_score 0–5  |

CxC caption ids (`COCO_val2014:sentid:190268`) are resolved to text via
`data/raw/coco_karpathy/dataset_coco.json` (`sentid → raw`).

## Setup

Uses the `STS` conda env:

```bash
conda run -n STS pip install -r requirements.txt
```

## Run

```bash
# default encoder (all-mpnet-base-v2) on SICK-R test + CxC val/test
python -m sts.run --datasets sick cxc-val cxc-test

# quick smoke test (subsample 200 pairs/dataset)
python -m sts.run --datasets sick cxc-val --limit 200

# swap encoder
python -m sts.run --datasets sick --encoder sentence-transformers/all-MiniLM-L6-v2

# E5-family (needs a prompt prefix)
python -m sts.run --datasets sick --encoder intfloat/e5-base-v2 --prompt-prefix "query: "
```

Results print as a table and are saved to `outputs/` as JSON. See
`configs/default.yaml` for all knobs and suggested encoders to compare.

## Layout

```
sts/
  config.py            # Config dataclass, device resolution
  data/                # SICK + CxC loaders -> uniform STSData
  encoders/            # TextEncoder interface: SBERT + raw-HF (mean/CLS) backends
  similarity.py        # paired cosine similarity
  metrics.py           # Spearman / Pearson / Kendall
  evaluate.py          # load -> encode (dedup) -> cosine -> correlate; dataset registry
  run.py               # CLI (python -m sts.run)
scripts/run_text_sts.py  # wrapper for `python scripts/...`
configs/default.yaml
```
