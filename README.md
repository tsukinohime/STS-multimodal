# STS-experiments

Research question: **when measuring Semantic Textual Similarity (STS) between
two texts, can images generated from each text (used as a caption for an
image-generation model) provide extra semantic information that helps the STS
task?**

This repository currently holds **Milestone 1 — the text-only baseline** that
every later multimodal result is compared against. No images are involved yet.

## What Milestone 1 does

```
sentence pair → frozen pretrained encoder → L2-normalised embeddings
              → cosine similarity → Spearman (headline) + Pearson vs. gold
```

Nothing is trained, fine-tuned, regressed or calibrated, and no decision is
made on a test split. "Zero-shot" means **not trained on the target datasets in
this project** — it is *not* a claim that the encoders' pretraining corpora are
disjoint from these benchmarks. COCO captions in particular are widely used in
pretraining, so CxC numbers should be read with that in mind.

## Datasets

Reported **separately** and combined with an *unweighted macro average*. They
are never concatenated into one global correlation — a pooled correlation over
corpora with different score scales mostly measures the between-corpus offsets.

| name        | source                                        | gold                    | categories |
|-------------|-----------------------------------------------|-------------------------|------------|
| `cxc-val`   | `cxc/data/sts_val.csv` + Karpathy COCO        | `agg_score` 0–5         | `sampling_method` |
| `cxc-test`  | `cxc/data/sts_test.csv` + Karpathy COCO       | `agg_score` 0–5         | `sampling_method` |
| `sick-test` | `sick/SICK.txt`, `SemEval_set == TEST`        | `relatedness_score` 1–5 | source corpus |
| `sick-all`  | `sick/SICK.txt`, every pair                   | `relatedness_score` 1–5 | source corpus |
| `sts3k`     | `sts3k/STS3k_all.txt`, complete file          | 0–1                     | none in this file |

Notes:

* **CxC** uses only the caption–caption STS files. Caption ids
  (`COCO_val2014:sentid:190268`) resolve through
  `coco_karpathy/dataset_coco.json` to both the caption text and the
  **`cocoid` of the source image**, which is carried into every pair row so the
  visual milestone can reuse the same tables. An unresolvable id is a hard
  error, never a silent drop. The `sis_*` / `sits_*` files (image–image,
  image–text) are intentionally untouched.
  Categories: `c2c_cocaption` (both captions describe the *same* image) and
  `c2c_isim` (different but visually similar images) — a distinction that
  matters directly for the image experiments.
* **SICK** uses the relatedness score only; entailment labels are ignored.
  `sick-all` **contains** `sick-test`, so only `sick-test` feeds the macro
  average. The distributed `SICK.txt` here has 9,840 rows (TEST = 4,906), which
  is the UniTN combined-file variant rather than the 9,927-pair split some
  papers cite — the manifest records the file's SHA-256 either way.
* **STS3k**: the file in `data/raw/sts3k/` is `sentence1;sentence2;score` with
  no category column, so only overall results are reported. The adapter sniffs
  for a category column and switches on per-category reporting automatically if
  a richer official file is dropped in.

Text is normalised by stripping leading/trailing whitespace **only** — no
lowercasing, no punctuation removal, no stop-word removal.

## Models

All frozen, inference only.

| key                  | model                                     | params | dim  | role |
|----------------------|-------------------------------------------|--------|------|------|
| `jina-v5-omni-small` | `jinaai/jina-embeddings-v5-omni-small`    | 1.74B  | 1024 | **main** |
| `jina-v5-text-small` | `jinaai/jina-embeddings-v5-text-small`    | 677M   | 1024 | control |
| `e5-large-v2`        | `intfloat/e5-large-v2`                    | 335M   | 1024 | off by default |
| `minilm-l6-v2`       | `sentence-transformers/all-MiniLM-L6-v2`  | 22M    | 384  | smoke-test only |

The jina-v5 adapter calls the checkpoints' own official entry point,

```python
model.encode(texts=[...], task="text-matching", prompt_name="document")
```

which owns every detail that must not be guessed: the `"Document: "` prefix
(the only prompt the non-retrieval tasks were trained with), the `text-matching`
LoRA adapter and its task-specific token embeddings, last-token pooling, and
L2 normalisation. The adapter adds only batching, dtype/device placement and
truncation accounting. Full 1024-dim embeddings — no Matryoshka truncation.

`jina-v5-omni-small` loads with `modality: text`, which skips the vision and
audio towers. Text embeddings are unaffected (the towers are separate
parameters), and the visual milestone reloads the *same repo* with
`modality: vision` to get image vectors in this same space.

### On pairing `v5-text-small` (text) with `v5-omni-small` (images)

Jina states the two share a vector space, which would let the cheaper 677M
model handle text while omni handles images. That is a claim worth measuring
rather than assuming — the checkpoints have different vocabularies (151,936 vs
151,672), so if they were only *approximately* aligned, any multimodal gain
could come from residual misalignment rather than image semantics.

The `align` command measures it directly:

```bash
python -m sts.cli align --config configs/text_only.yaml \
    --model-a jina-v5-text-small --model-b jina-v5-omni-small \
    --dataset sick-test --sample 512
```

It reports the cosine between the two models' embeddings of the *same*
sentence, each model's own STS Spearman, and the Spearman when sentence 1 is
encoded by one model and sentence 2 by the other — that last gap is the cost of
mixing.

**Measured** (256 pairs each, MPS/float32, `task=text-matching`):

| dataset     | cosine, same text | ρ text-small | ρ omni-small | ρ cross-encoded | penalty |
|-------------|-------------------|--------------|--------------|-----------------|---------|
| `sick-test` | 0.9999 (min 0.9998) | 0.9125 | 0.9126 | 0.9125 | +0.0001 |
| `cxc-val`   | 1.0000 (min 0.9999) | 0.7523 | 0.7516 | 0.7515 | +0.0008 |

The shared-space claim holds to numerical noise: on text the two models are
interchangeable, and prediction agreement is ρ ≈ 1.0000. The omni text tower is
effectively `v5-text-small` locked in place, which is exactly what the
"locked aligned towers" architecture describes.

**Practical consequence:** since the two agree on text, splitting the work
buys nothing — it would just keep two models resident. Use **`v5-omni-small`
alone** for the multimodal milestone (`modality: vision` loads the vision tower
*and* the text tower), and keep `v5-text-small` as the text-only control that
confirms the alignment still holds. Re-run `align` if either checkpoint's
revision changes.

## Setup

```bash
conda create -n STS python=3.11 -y
conda activate STS
pip install -r requirements.txt
```

Datasets are expected under `data/raw/` (gitignored):

```
data/raw/
  cxc/data/sts_val.csv, sts_test.csv
  coco_karpathy/dataset_coco.json
  sick/SICK.txt
  sts3k/STS3k_all.txt
```

Override with `data.root` in the YAML or `--data-root` on the command line.

### Reading straight from `.tar` archives

Cluster filesystems meter inode count as well as bytes, and COCO's ~123k images
make extraction impractical there. Any of the paths above is therefore *also*
satisfied by a matching member of a `.tar` in an ancestor directory:

```
data/raw/
  coco.tar    → coco_karpathy/dataset_coco.json
  cxc.tar     → cxc/data/sts_val.csv, cxc/data/sts_test.csv
  sick/SICK.txt          (plain file; small enough not to bother)
  sts3k/STS3k_all.txt    (plain file)
```

Nothing needs extracting and **the config does not change** — [sts/archive.py](sts/archive.py)
indexes each archive's members once and resolves paths against that index, so
the archive's own filename need not match the directory it contains. Files are
hashed by **content**, so a manifest produced from archives is directly
comparable with one produced from extracted files (verified: identical
SHA-256s and identical correlations either way).

Use **uncompressed** `.tar`. `.tar.gz` works but every member read decompresses
from the start of the archive; the loader warns when it sees one.

## Run

Everything is driven by one YAML file.

```bash
# The full text-only baseline.
python -m sts.cli run --config configs/text_only.yaml

# 32 pairs per dataset, plus every validation check (this is the smoke test).
python scripts/smoke_test.py --config configs/smoke.yaml

# One model only; useful for splitting work across jobs.
python -m sts.cli run --config configs/text_only.yaml --models jina-v5-omni-small

# Rebuild metrics + summary from predictions already on disk (no GPU needed).
python -m sts.cli report --config configs/text_only.yaml

# Correctness checks on their own.
python -m sts.cli validate --config configs/text_only.yaml

# List registered datasets.
python -m sts.cli datasets
```

Useful flags: `--limit N`, `--device {auto,cuda,mps,cpu}`, `--dtype`,
`--batch-size`, `--bootstrap 1000`, `--no-resume`, `--no-cache`,
`--output-dir`, `--run-tag`.

### Inference defaults

| setting | value | why |
|---|---|---|
| dtype | `bfloat16` on CUDA, `float32` otherwise | the checkpoints ship bf16 and H100s run it natively; MPS kernel coverage for bf16 is patchier and these models are small |
| batch size | 64 (32 for omni-small) | conservative for one GPU; lower it if you hit OOM |
| `max_length` | 512 tokens | far above every corpus here — truncation is counted and reported, and is 0 in practice |
| embeddings | full 1024 dim, L2-normalised | no Matryoshka truncation |
| seed | 42 | governs subsampling and the metric bootstrap; inference itself is deterministic |

## LLM-as-judge experiment

The second text-only baseline keeps everything above — datasets, splits,
categories, seed, outputs, metrics — and swaps the scorer: a frozen chat LLM
is shown each pair together with the dataset's **own** rating guidelines and
asked for a single digit.

```bash
python -m sts.cli judge --config configs/llm_judge.yaml                 # full run
python scripts/smoke_test.py --config configs/llm_judge_smoke.yaml --judge   # 32 pairs, laptop
python -m sts.cli report --config configs/llm_judge.yaml                # rebuild summary
```

**Scoring.** No text is generated. The chat template is rendered up to the
assistant's first token (thinking disabled, so nothing precedes it) and the
next-token distribution at that position is read for the allowed digits:

```
E[score] = Σ_d d · p(d) / Σ_d p(d)
```

An argmax digit gives at most 6 distinct values across 44k CxC pairs — massive
rank ties. The expectation is continuous. Both are recorded (`expected_score`,
`argmax_score`), plus the per-digit probabilities, the total mass the model put
on valid digits (`valid_mass`), and the mass on a leading space. Rows whose
digit mass falls below `min_valid_mass` are NaN, counted as degenerate, and
excluded from the correlation rather than silently trusted.

**Prompts** live in [sts/judge/prompts.py](sts/judge/prompts.py), one spec per
dataset family, and follow four rules:

1. *Scale and anchors are copied verbatim from the original annotation
   guidelines* — never invented. Each spec records its source.
   - CxC: the six STS definitions from the CxC paper appendix (Cer et al.
     2017 / Agirre et al. 2012), scale 0–5.
   - SICK: Marelli et al. 2014 describe the task only through one example
     pair per endpoint ("completely unrelated" / "very related"), explicitly
     *avoiding* strict definitions; the prompt does the same. Scale 1–5.
   - STS3k: Fodor et al. 2025, appendix 8.2.1, verbatim; the authors gave
     participants no rating framework on purpose. Scale 1–7.
2. *Symmetric wording*: the question is always about "the two sentences";
   neither sentence is the subject or object of the comparison.
3. *No chain-of-thought*: the answer is one digit, nothing else, and the
   template's thinking mode is switched off.
4. *Hashed*: `prompt_hash` = spec + chat template. It is in every prediction
   row, in the manifest, and in `outputs/<exp>/prompts/<family>__<version>__<hash>.md`
   together with a fully rendered example, so any later prompt edit is
   traceable. Changing a scale means a new spec version — and a new hash.

Scales therefore differ per dataset. Spearman is invariant to the scale, so
this costs nothing for the comparison across datasets or against the embedding
baseline.

**Diagnostics per (judge, dataset)**, recorded in the manifest: Spearman of the
expectation next to Spearman of the argmax (what the weighting buys), the
argmax histogram and largest tie fraction, digit-mass statistics, and an
*order-symmetry check* — a sample re-scored with the two sentences swapped,
reported as mean |Δ| and the Spearman between orders. A symmetric prompt does
not make an LLM order-invariant; this measures how far off it is. Set
`symmetrize: true` to score both orders for every pair and average (2× cost).

**Judge**: `Qwen/Qwen3.5-35B-A3B` — 35B-parameter MoE with 3B active, hybrid
gated-DeltaNet / full attention, 262k context, thinking mode on by default
(disabled here). bf16 weights are ~70 GB; one H100 suffices for prefill-only
scoring. Requires `transformers >= 5.16`:

```bash
pip install --upgrade -r requirements-judge.txt     # into the existing venv
```

The judgment cache reuses the embedding cache machinery (`.cache/judgments/`),
keyed by ordered pair text and the prompt hash, so a killed job resumes and a
prompt edit never reuses stale answers.

## Outputs

Written to `experiment.output_dir` (default `outputs/text_only/`):

```
manifest.json               # config + hash, git commit, library and hardware
                            # versions, dtype, batch size, seed, model revisions,
                            # per-file data SHA-256, per-run timings
pairs/<dataset>.csv         # the unified pair table: pair_id, dataset, split,
                            # domain, sentence 1/2 + ids, gold, image ids
predictions/<model>/<dataset>.csv
                            # per pair: gold, cosine prediction, model, split,
                            # domain, image ids, token counts, truncation flags
metrics.csv / metrics.json  # Spearman + Pearson per dataset x split x domain
                            # x model, plus the macro-average row
summary.md                  # main table, per-dataset detail, category
                            # breakdown, runtime, truncation stats, provenance
validation.json             # results of every correctness check
```

Pair tables carry the text once; prediction files reference pairs by id so
running many models stays cheap on disk.

## Caching and resume

Encoding dominates the runtime, and the corpora overlap heavily (CxC reuses
25,000 captions across ~44,000 pairs per split). The embedding cache is keyed
by **text content** plus the settings that change a vector — model id, task,
prompt, `max_length`, `truncate_dim`, dtype — but deliberately **not** by
device, so a cache warmed on a laptop is valid on the cluster.

Two levels of resume:

* `(model, dataset)` — with `resume: true`, a rerun skips any combination whose
  prediction CSV exists.
* Within a dataset — the cache flushes every `flush_every` unique texts
  (atomic temp-file + rename), so a job killed mid-encode restarts near where
  it stopped.

### Cache file count

The cache stores one shard per `flush_every` texts, never one file per text —
which matters on a cluster that meters inodes. For the full text-only run
(~60k unique texts, 1024-dim float32, `flush_every: 2048`):

| | per model | both models |
|---|---|---|
| shards | 33 | 66 |
| files (+ `meta.json`) | 34 | 68 |
| size | 234 MB | 468 MB |
| mean file size | 7.1 MB | 7.1 MB |

Repeatedly resuming an interrupted job leaves one shard per interrupted block,
so the count can drift upward over time. `cache --compact` merges them back
into a single file:

```bash
python -m sts.cli cache --config configs/text_only.yaml            # show size
python -m sts.cli cache --config configs/text_only.yaml --compact  # merge shards
```

Compaction is crash-safe (merged shard is written and renamed into place before
the originals are removed) and content-preserving — verified bit-identical.
Raising `flush_every` also reduces the file count, at the cost of coarser
resume granularity.

## TSUBAME4

TSUBAME runs the **Grid Engine** scheduler. See **[tsubame/README.md](tsubame/README.md)**
for login-node vs. compute-node usage, resource types and troubleshooting. Short version:

```bash
cp tsubame/env.sh.example tsubame/env.sh   # group, paths, env, resource type
$EDITOR tsubame/env.sh
bash tsubame/prefetch_models.sh            # ON THE LOGIN NODE — needs network
bash tsubame/submit.sh                     # one GPU job per model
python -m sts.cli report --config configs/text_only.yaml   # merge the results
```

Nothing site-specific (username, group, paths, env) is hardcoded — it all lives
in the gitignored `tsubame/env.sh`. Compute nodes run with `HF_HUB_OFFLINE=1`
against the prefetched cache. Jobs are per-model and resumable, so an h_rt
kill costs almost nothing.

## Validation

`python scripts/smoke_test.py` runs the real pipeline twice and asserts:

| check | what it proves |
|---|---|
| `identical_sentences_cosine_is_1` | cos(x, x) = 1 — pooling and determinism are sound |
| `similarity_is_order_invariant` | cos(a, b) = cos(b, a) after re-encoding in swapped batch order |
| `embeddings_are_l2_normalized` | ‖v‖ = 1 and the dimension is what was configured |
| `predictions_have_no_nan` | no NaN in predictions or gold, cosine within [-1, 1] |
| `cxc_caption_ids_resolve` | every CxC caption id maps to COCO text *and* an image id |
| `pair_count_matches_source` | loaded pairs == rows in the source file |
| `gold_scores_have_no_nan` | gold columns are clean |
| `embedding_cache_reused_after_restart` | a second, separate process gets 100% cache hits |
| `rerun_reproduces_identical_predictions` | both runs agree bit for bit |
| `single_command_reproduces_the_run` | one config + one command is the whole experiment |

## Layout

```
sts/
  config.py        one YAML -> typed config (ExperimentConfig / ModelSpec / ...)
  hashing.py       stable content hashes for cache keys and provenance
  data/            dataset adapters -> one uniform pair schema
    base.py        STSPair / STSDataset, the shared schema
    cxc.py  sick.py  sts3k.py  registry.py
  models/          model adapters -> one uniform encoder interface
    base.py        TextEncoder interface + factory
    jina_v5.py     jina-embeddings-v5 (official encode() path)
    sentence_transformer.py   hf_mean.py
  archive.py       transparent reads from .tar archives (cluster inode limits)
  cache.py         content-addressed embedding cache (sharded, atomic, resumable)
  similarity.py    paired cosine similarity
  metrics.py       Spearman / Pearson / bootstrap CI / macro average
  pipeline.py      orchestration
  manifest.py      reproducibility record
  report.py        metrics CSV/JSON + markdown summary
  diagnostics.py   validation checks + cross-model alignment probe
  cli.py           command-line entry point
  judge/           LLM-as-judge: prompts.py (specs + hashes), scorer.py
                   (next-token expectation), pipeline.py (orchestration)
configs/
  text_only.yaml   the full embedding experiment
  smoke.yaml       32 pairs/dataset, laptop-sized
  llm_judge.yaml   the full LLM-as-judge experiment
  llm_judge_smoke.yaml   its 32-pair smoke test (0.5B stand-in judge)
scripts/
  smoke_test.py    run twice + validate everything
tsubame/
  README.md  env.sh.example  activate_env.sh  prefetch_models.sh
  job_sts.sh  submit.sh
```
