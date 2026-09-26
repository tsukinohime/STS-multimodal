#!/bin/bash
# Run from the repository root. Both modes use configs/llm_judge.yaml.
#   qsub -g <group> -v STS_PYTHON="$(command -v python)" tsubame/job_judge.sh smoke
#   qsub -g <group> -l h_rt=4:00:00 -v STS_PYTHON="$(command -v python)" tsubame/job_judge.sh full
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
# Request a whole GPU: a half-GPU MIG instance cannot hold the bf16 35B weights.
#$ -cwd
#$ -N sts-judge
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> [qsub options] tsubame/job_judge.sh [smoke [N] | full]

  smoke [N]  Score N pairs per dataset (default: 32). Recompute predictions
             without reading or writing the judgment cache.
  full       Score every pair, using the YAML's resume and cache settings.
  No mode    Defaults to smoke, 32 pairs per dataset.

Optional environment variables (pass with qsub -v):
  STS_PYTHON             Absolute path to the prepared Python interpreter.
                        If absent, activate the environment via env.sh.
  STS_REPO               Repository root (default: submission directory).
  STS_DATA_ROOT          Override all dataset paths; otherwise use the YAML
                        or the data root supplied by env.sh during activation.
  STS_JUDGE_CONFIG       Config file (default: configs/llm_judge.yaml).
  STS_JUDGE_OUTPUT_ROOT  Output parent (default: outputs/llm_judge in the repo).
                        Results go into smoke_N/ or full/ under this parent.
  STS_JUDGE_CACHE_DIR    Judgment cache (default: .cache/judgments in the repo).

Batch size is taken from the judge's YAML entry. STS_BATCH_SIZE is not used.
The default allocation is node_q=1 for one hour; qsub -l overrides it.
EOF
}

judge_mode="${1:-smoke}"
case "$judge_mode" in
  -h|--help)
    usage
    exit 0
    ;;
  smoke)
    judge_limit="${2:-32}"
    if [[ $# -gt 2 || ! "$judge_limit" =~ ^[1-9][0-9]*$ ]]; then
      echo "[error] use: job_judge.sh smoke [positive integer]" >&2
      exit 2
    fi
    judge_output_name="smoke_$judge_limit"
    ;;
  full)
    if [[ $# -gt 1 ]]; then
      echo "[error] full mode takes no sample limit" >&2
      exit 2
    fi
    judge_output_name="full"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# Grid Engine spools the script away from the repository, so do not infer the
# repository from BASH_SOURCE. Submit from the repo root or pass STS_REPO.
cd "${STS_REPO:-${SGE_O_WORKDIR:-$PWD}}"
export STS_REPO="$PWD"

judge_config="${STS_JUDGE_CONFIG:-$STS_REPO/configs/llm_judge.yaml}"
judge_output="${STS_JUDGE_OUTPUT_ROOT:-$STS_REPO/outputs/llm_judge}/$judge_output_name"
judge_cache="${STS_JUDGE_CACHE_DIR:-$STS_REPO/.cache/judgments}"

if [[ -n "${STS_PYTHON:-}" ]]; then
  judge_python="$STS_PYTHON"
  if [[ "$judge_python" != /* || ! -x "$judge_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  # Supply judge defaults to the shared bootstrap rather than its embedding
  # defaults. The CLI below uses the judge-specific paths even if env.sh sets
  # STS_CONFIG/STS_OUTPUT_DIR/STS_CACHE_DIR for the embedding experiment.
  export STS_CONFIG="$judge_config"
  export STS_OUTPUT_DIR="$judge_output"
  export STS_CACHE_DIR="$judge_cache"
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  judge_python="$(command -v python)"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$judge_output"

trap 'judge_status=$?; echo "[job] finished with status $judge_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} mode=$judge_mode host=$(hostname)"
echo "[job] python=$judge_python config=$judge_config"
echo "[job] output=$judge_output cache=$judge_cache"

# Report the CUDA device visible to this process, including a MIG partition
# if one was allocated. No model weights are loaded by this check.
"$judge_python" - <<'PYTHON'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable; check the GPU allocation and Python environment")
gpu = torch.cuda.get_device_properties(0)
print(f"[gpu] {gpu.name}: {gpu.total_memory / 1024**3:.2f} GiB visible VRAM", flush=True)
PYTHON

judge_command=(
  "$judge_python" -u -m sts.cli judge
  --config "$judge_config"
  --device cuda --dtype bfloat16
  --output-dir "$judge_output"
  --cache-dir "$judge_cache"
  --run-tag "${JOB_ID:-manual}"
)
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  judge_command+=(--data-root "$STS_DATA_ROOT")
fi
if [[ "$judge_mode" == smoke ]]; then
  judge_command+=(--limit "$judge_limit" --no-resume --no-cache)
else
  # --limit 0 means no subsampling in the pipeline, even if the YAML has a
  # limit left over from a manual test.
  judge_command+=(--limit 0)
fi

printf '[job] command:'
printf ' %q' "${judge_command[@]}"
printf '\n'
"${judge_command[@]}"

echo "[job] results: $judge_output"
