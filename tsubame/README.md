# Running on TSUBAME4

TSUBAME uses the **Grid Engine** scheduler: `qsub` to submit a batch job, `qrsh`
for an interactive one, `qstat` to watch, `qdel` to cancel.

> Resource-type names, storage paths and the login hostname are site-specific
> and do change. Confirm them against the current TSUBAME user guide and your
> own account before the first submission — `qconf -sc` lists the schedulable
> resource attributes. Nothing in this repository hardcodes them; they all live
> in `tsubame/env.sh`.

## Login node vs. compute node

```
your laptop  --ssh-->  login node  --qsub/qrsh-->  compute node (GPU)
```

**Login node** — shared by everyone. Only for editing files, `git`, installing
packages, and downloading model weights. Never run the experiment here: a long
job on a login node gets killed and is bad manners.

**Compute node** — where the work happens. Two ways in:

```bash
# 1. Batch (what you want for the real run): submit and walk away.
qsub -g <group> -l <resource>=1 -l h_rt=1:00:00 job.sh

# 2. Interactive: get a shell on a GPU node, for debugging.
qrsh -g <group> -l <resource>=1 -l h_rt=1:00:00
```

`-l <resource>=<count>` picks the **resource type** — how much of a node you
get. This experiment is single-GPU, so the smallest GPU allocation is right.
The usual TSUBAME names are node fractions:

| resource type | GPUs | use here                                 |
|---------------|------|------------------------------------------|
| `s_gpu`       | 1    | **default here** — smallest GPU slice    |
| `q_node`      | 1    | 1 GPU plus a quarter node of CPU/RAM     |
| `h_node`      | 2    | not needed                               |
| `f_node`      | 4    | not needed                               |

`-l h_rt=HH:MM:SS` is the runtime limit (Grid Engine's equivalent of a
walltime). `-g <group>` charges the job to your TSUBAME group; omit it for a
**trial run** (お試し実行), which needs no group but is capped in size and
runtime — fine for a smoke test, not for the full CxC run.

### Grid Engine vs. PBS

If you follow a PBS-flavoured tutorial, these are the things that differ:

| | PBS Pro | **Grid Engine (here)** |
|---|---|---|
| script directive | `#PBS` | `#$` |
| group / account | `-P <group>` | `-g <group>` |
| resources | `-q <queue> -l select=1` | `-l <type>=<n>` |
| runtime limit | `-l walltime=…` | `-l h_rt=…` |
| start in submit dir | `cd $PBS_O_WORKDIR` | `#$ -cwd` |
| job id variable | `$PBS_JOBID` | `$JOB_ID` |
| merge stderr | `-j oe` | `-j y` |
| interactive | `qsub -I` | `qrsh` |

Compute nodes generally have **no internet access**, which is why model weights
are downloaded on the login node first (step 3 below) and the jobs then run with
`HF_HUB_OFFLINE=1`.

## Setup, once

```bash
# 1. Site config — group, paths, env, resource type.
cp tsubame/env.sh.example tsubame/env.sh
$EDITOR tsubame/env.sh          # at minimum: STS_GROUP, STS_CONDA_SH, STS_DATA_ROOT

# 2. Python env (login node). MUST be >= 3.10 — see "Python version" below.
conda create -p $STS_REPO/../venvs/sts python=3.11 -y
conda activate $STS_REPO/../venvs/sts
pip install -r requirements.txt

# 3. Model weights into the shared HF cache (login node — needs network).
bash tsubame/prefetch_models.sh
```

### Python version

**The system Python (3.9) cannot run this project.** Not a preference — the
main model's remote modeling code imports `Qwen3VL` and `Qwen2.5-Omni`
components that exist only in `transformers` 5.x, and every `transformers` 5.x
release declares `requires_python >= 3.10.0`. `torch >= 2.13`, `peft >= 0.20`
and `sentence-transformers >= 6.0` have the same floor. Relaxing the pins in
`requirements.txt` is not an option: `transformers` 4.x has no `Qwen3VL`, so
`jina-v5-omni-small` simply fails to load.

Use **Python 3.11** (what this project is validated against), in this order of
preference:

1. A site module, if one exists — `module avail 2>&1 | grep -iE "python|conda"`.
   If you go this way, the job script must `module load` the *same* module
   before activating the env, or the compute node won't find the interpreter.
2. A site `conda`/`miniforge` module, with `conda create -p <path-on-group-disk>`
   so the env does not eat the small home quota.
3. Miniforge installed into group storage yourself — needs no admin rights and
   no `module load` at job time, which makes it the most robust option:

   ```bash
   curl -L -o miniforge.sh \
     https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash miniforge.sh -b -p <groupdir>/miniforge && rm miniforge.sh
   source <groupdir>/miniforge/etc/profile.d/conda.sh
   conda create -p <groupdir>/venvs/sts python=3.11 -y
   ```

Verify before submitting anything — `transformers` must report **5.x**:

```bash
python -V
python -c "import torch, transformers, peft, sentence_transformers as st; \
print(torch.__version__, transformers.__version__, peft.__version__, st.__version__)"
```

Keep `STS_DATA_ROOT`, `STS_OUTPUT_DIR`, `STS_CACHE_DIR` and `HF_HOME` on group
storage (`/gs/bs/<group>/…` or `/gs/fs/<group>/…`), not in `$HOME` — the home
quota is small and the HF cache alone is several GB.

## LLM-as-judge: direct qsub submission

`job_judge.sh` runs the real judge from `configs/llm_judge.yaml` in either
small-sample or full mode. It defaults to `node_q=1` (one whole GPU, 192 GB
host RAM) and a one-hour time limit. TSUBAME4 uses the resource names `node_q`
and `gpu_1` for whole-GPU allocations; `node_o` and `gpu_h` are half-GPU MIG
allocations and cannot hold the current 35B judge's bf16 weights. See the
[current resource table](https://www.t4.cii.isct.ac.jp/docs/handbook.en/jobs/#511-resource-types).

From the **repository root on a login node**, activate your prepared Python
environment, then submit one of these commands (replace `<group>`):

```bash
# Real-model smoke run: 32 pairs per configured dataset.
qsub -g <group> -v STS_PYTHON="$(command -v python)" tsubame/job_judge.sh smoke

# Larger smoke run: 128 pairs per dataset.
qsub -g <group> -v STS_PYTHON="$(command -v python)" tsubame/job_judge.sh smoke 128

# Full experiment; four hours is a walltime limit, not an estimated duration.
qsub -g <group> -l h_rt=4:00:00 -v STS_PYTHON="$(command -v python)" tsubame/job_judge.sh full
```

`STS_PYTHON` selects the same interpreter you just activated, without relying
on the batch shell to inherit Conda or venv activation. This route bypasses
`env.sh` and `activate_env.sh`; use it for a self-contained Python environment.
If your environment needs site modules or activation hooks, configure those in
`tsubame/env.sh` and **omit `-v STS_PYTHON=...`** to use the shared bootstrap.

Both modes keep the model path, prompts, datasets, seed, and metrics from the
same formal YAML. Set the main judge's `model_id` there to your local model
directory. Batch size comes from the YAML judge entry; this script does not
pass `STS_BATCH_SIZE`, which would not override that entry anyway.

| Mode | Behavior | Default output directory |
|------|----------|--------------------------|
| `smoke` or no argument | 32 pairs/dataset, fresh inference with `--no-resume --no-cache` | `outputs/llm_judge/smoke_32/` |
| `smoke N` | N pairs/dataset, same fresh-inference behavior | `outputs/llm_judge/smoke_N/` |
| `full` | All pairs, honoring the YAML's resume/cache settings | `outputs/llm_judge/full/` |

Smoke mode exercises the pipeline; it does not run the assertions in
`scripts/smoke_test.py`. It never clears or writes the full run's judgment
cache. Full mode uses `.cache/judgments/` and can be resumed by submitting the
same command with the same config and data. Each submission's manifest,
metrics, and summary carry the scheduler job ID, preserving earlier records.
Use a new `STS_JUDGE_OUTPUT_ROOT` when changing data, prompts, or scoring
settings: existing prediction CSVs are reused without checking those changes.

Optional paths can be passed in the same `-v` argument, for example:

```bash
qsub -g <group> \
  -v STS_PYTHON="$(command -v python)",STS_DATA_ROOT=/work/your/data/raw \
  tsubame/job_judge.sh smoke
```

Other optional variables are `STS_REPO`, `STS_JUDGE_CONFIG`,
`STS_JUDGE_OUTPUT_ROOT` (the script appends `smoke_N` or `full`), and
`STS_JUDGE_CACHE_DIR`. Run `bash tsubame/job_judge.sh --help` for details.
Judge-specific defaults do not use the embedding experiment's
`STS_CONFIG`, `STS_OUTPUT_DIR`, or `STS_CACHE_DIR` settings.

Watch with `qstat -u "$USER"` and `tail -f sts-judge.o<jobid>` in the submission
directory. The log first prints the GPU memory visible to PyTorch, then the
exact experiment command. After completion, results are in the mode's output
directory. To rebuild the full report without GPU inference:

```bash
python -m sts.cli report --config configs/llm_judge.yaml --output-dir outputs/llm_judge/full
```

## Run (embedding)

```bash
# Sanity check on a small allocation first (32 pairs/dataset, minutes).
qrsh -g <group> -l s_gpu=1 -l h_rt=0:30:00
#   ... on the compute node:
cd $STS_REPO && source tsubame/activate_env.sh
python scripts/smoke_test.py --config configs/smoke.yaml --device cuda

# The real thing: one independent job per model.
bash tsubame/submit.sh

# Inspect the qsub lines without submitting.
DRY_RUN=1 bash tsubame/submit.sh

# Just one model, with a longer runtime limit.
STS_H_RT=8:00:00 bash tsubame/submit.sh jina-v5-omni-small
```

Watch and collect:

```bash
qstat -u $USER                 # queued / running
qdel <jobid>                   # cancel
tail -f logs/sts-*.o*          # job output (Grid Engine names it <jobname>.o<jobid>)

# Merge the per-model manifests into one metrics.csv + summary.md.
python -m sts.cli report --config configs/text_only.yaml
```

## Resuming

Jobs are resumable by design, so an h_rt kill costs almost nothing:

* Each `(model, dataset)` writes its own `predictions/<model>/<dataset>.csv`.
  With `resume: true` (the default) a resubmitted job skips the finished ones.
* Below that, the embedding cache flushes every `flush_every` unique texts, so
  even a dataset that was interrupted mid-encode restarts near where it stopped.
* The cache is keyed by text content and by the settings that affect a vector —
  **not** by device — so a cache warmed anywhere is valid on the cluster.

To resume, resubmit the identical command. To force a clean recompute, add
`--no-resume` (or delete the prediction CSVs).

### Keeping the inode count down

The embedding cache is sharded, not one-file-per-text: the full text-only run
leaves ~34 files of ~7 MB per model, which is the file profile this filesystem
wants. Each interrupted-and-resumed block does add a shard, so after a run that
was restarted several times, merge them:

```bash
python -m sts.cli cache --config configs/text_only.yaml            # show size
python -m sts.cli cache --config configs/text_only.yaml --compact  # merge shards
```

Datasets stay archived — `.tar` files are read in place, never extracted (see
the main README).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `Unable to run job: ... no suitable queues` | `STS_RESOURCE` name wrong for this site — check `qconf -sc` |
| job rejected on the group | `STS_GROUP` wrong or expired — check your account's group list |
| job dies instantly, log mentions `CalledProcessError` in `activate_env.sh` | `STS_CONDA_SH` not set; it is `$(conda info --base)/etc/profile.d/conda.sh` |
| `OSError: … not a local folder … HF_HUB_OFFLINE=1` | weights not prefetched, or `HF_HOME` differs between login and compute — rerun `prefetch_models.sh` with the same `env.sh` |
| CUDA OOM | lower `STS_BATCH_SIZE`; these models are small, so 64 is already conservative |
| job runs but is very slow | check `nvidia-smi` in the log; if it reports no GPU the job landed on a CPU-only resource type |
