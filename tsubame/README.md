# Running on TSUBAME4

TSUBAME4 (Institute of Science Tokyo) uses the **PBS Professional** scheduler:
`qsub` to submit, `qstat` to watch, `qdel` to cancel.

> The resource-type names, storage paths and login hostname below are
> site-specific and do change. Confirm them against the current TSUBAME4 user
> guide and your own `t4-user-info group list` before the first submission.
> Nothing in this repository hardcodes them — they all live in `tsubame/env.sh`.

## Login node vs. compute node

```
your laptop  --ssh-->  login node  --qsub-->  compute node (GPU)
```

**Login node** — shared by everyone. Only for editing files, `git`, installing
packages, and downloading model weights. Never run the experiment here: a long
job on a login node gets killed and is bad manners.

**Compute node** — where the work happens. Two ways in:

```bash
# 1. Batch (what you want for the real run): submit and walk away.
qsub -q gpu_1 -l select=1 -l walltime=4:00:00 -P <group> job.sh

# 2. Interactive: get a shell on a GPU node, for debugging.
qsub -I -q gpu_1 -l select=1 -l walltime=1:00:00 -P <group>
```

`-q` picks the **resource type** (how much of a node you get). This experiment
is single-GPU, so the smallest GPU allocation is right:

| resource type | GPUs | use here                                    |
|---------------|------|---------------------------------------------|
| `gpu_1`       | 1    | **default** — everything in this project    |
| `node_q`      | 1    | 1 GPU plus a quarter node of CPU/RAM        |
| `node_h`      | 2    | not needed                                  |
| `node_f`      | 4    | not needed                                  |

`-P <group>` charges the job to your TSUBAME group. Omit it to get a **trial
run** (お試し実行), which needs no group but is capped in nodes and walltime —
fine for a smoke test, not for the full CxC run.

Compute nodes generally have **no internet access**, which is why model weights
are downloaded on the login node first (step 2 below) and the jobs then run with
`HF_HUB_OFFLINE=1`.

## Setup, once

```bash
# 1. Site config — group, paths, env, resource type.
cp tsubame/env.sh.example tsubame/env.sh
$EDITOR tsubame/env.sh          # at minimum: STS_GROUP, STS_CONDA_SH, STS_DATA_ROOT

# 2. Python env (login node).
conda create -n STS python=3.11 -y
conda activate STS
pip install -r requirements.txt

# 3. Model weights into the shared HF cache (login node — needs network).
bash tsubame/prefetch_models.sh
```

Keep `STS_DATA_ROOT`, `STS_OUTPUT_DIR`, `STS_CACHE_DIR` and `HF_HOME` on group
storage (`/gs/bs/<group>/…` or `/gs/fs/<group>/…`), not in `$HOME` — the home
quota is small and the HF cache alone is several GB.

## Run

```bash
# Sanity check on a small allocation first (32 pairs/dataset, minutes).
qsub -I -q gpu_1 -l select=1 -l walltime=0:30:00 -P <group>
#   ... on the compute node:
cd $STS_REPO && source tsubame/activate_env.sh
python scripts/smoke_test.py --config configs/smoke.yaml --device cuda

# The real thing: one independent job per model.
bash tsubame/submit.sh

# Inspect the qsub lines without submitting.
DRY_RUN=1 bash tsubame/submit.sh

# Just one model, with a longer walltime.
STS_WALLTIME=8:00:00 bash tsubame/submit.sh jina-v5-omni-small
```

Watch and collect:

```bash
qstat -u $USER                 # queued / running
qdel <jobid>                   # cancel
tail -f logs/sts-*.o*          # job output (PBS names it <jobname>.o<jobid>)

# Merge the per-model manifests into one metrics.csv + summary.md.
python -m sts.cli report --config configs/text_only.yaml
```

## Resuming

Jobs are resumable by design, so a walltime kill costs almost nothing:

* Each `(model, dataset)` writes its own `predictions/<model>/<dataset>.csv`.
  With `resume: true` (the default) a resubmitted job skips the finished ones.
* Below that, the embedding cache flushes every `flush_every` unique texts, so
  even a dataset that was interrupted mid-encode restarts near where it stopped.
* The cache is keyed by text content and by the settings that affect a vector —
  **not** by device — so a cache warmed anywhere is valid on the cluster.

To resume, resubmit the identical command. To force a clean recompute, add
`--no-resume` (or delete the prediction CSVs).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `qsub: Unauthorized Request` | `STS_GROUP` wrong or expired — check `t4-user-info group list` |
| job dies instantly, log mentions `CalledProcessError` in `activate_env.sh` | `STS_CONDA_SH` not set; it is `$(conda info --base)/etc/profile.d/conda.sh` |
| `OSError: … not a local folder … HF_HUB_OFFLINE=1` | weights not prefetched, or `HF_HOME` differs between login and compute — rerun `prefetch_models.sh` with the same `env.sh` |
| CUDA OOM | lower `STS_BATCH_SIZE`; these models are small, so 64 is already conservative |
| job runs but is very slow | check `nvidia-smi` in the log; if it reports no GPU the job landed on a CPU resource type |
