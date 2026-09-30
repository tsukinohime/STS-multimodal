#!/bin/bash
# Post-hoc breakdown of the full LLM-judge run next to the embedding baseline,
# all in one job: SICK by NLI label, CxC by sampling_method (c2c_cocaption /
# c2c_isim). Reads prediction CSVs only; loads no model and needs no GPU.
# Run from the repository root:
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON="$(command -v python)" tsubame/job_breakdown.sh
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
# Choose the (CPU-only) resource explicitly with qsub -l when submitting.
#$ -cwd
#$ -N sts-breakdown
#$ -l h_rt=0:30:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> -l <resource>=<count> [qsub options] tsubame/job_breakdown.sh

Optional environment variables (pass with qsub -v):
  STS_PYTHON               Absolute path to the prepared Python interpreter.
                           If absent, activate the environment via env.sh.
  STS_REPO                 Repository root (default: submission directory).
  STS_DATA_ROOT            Override all dataset paths; otherwise use the YAML
                           or the data root supplied by env.sh during activation.
  STS_JUDGE_CONFIG         Config file (default: configs/llm_judge.yaml).
  STS_JUDGE_OUTPUT_DIR     The judge run to break down
                           (default: outputs/llm_judge/full in the repo).
  STS_BASELINE_OUTPUT_DIR  The embedding run to compare against
                           (default: outputs/text_only in the repo).
  STS_BASELINE_MODELS      Space-separated model keys from the baseline
                           (default: jina-v5-omni-small).
  STS_BREAKDOWN_DATASETS   Space-separated datasets
                           (default: sick-all sick-test cxc-val cxc-test).

Tables go to <judge output>/breakdown/. The default time limit is 30 minutes.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
fi

# Grid Engine spools the script away from the repository, so do not infer the
# repository from BASH_SOURCE. Submit from the repo root or pass STS_REPO.
cd "${STS_REPO:-${SGE_O_WORKDIR:-$PWD}}"
export STS_REPO="$PWD"

breakdown_config="${STS_JUDGE_CONFIG:-$STS_REPO/configs/llm_judge.yaml}"
judge_output="${STS_JUDGE_OUTPUT_DIR:-$STS_REPO/outputs/llm_judge/full}"
baseline_output="${STS_BASELINE_OUTPUT_DIR:-$STS_REPO/outputs/text_only}"
read -r -a baseline_models <<< "${STS_BASELINE_MODELS:-jina-v5-omni-small}"
read -r -a breakdown_datasets <<< "${STS_BREAKDOWN_DATASETS:-sick-all sick-test cxc-val cxc-test}"

# Fail here rather than silently breaking down the judge alone.
for predictions in "$judge_output/predictions" "$baseline_output/predictions"; do
  if [[ ! -d "$predictions" ]]; then
    echo "[error] no $predictions; set STS_JUDGE_OUTPUT_DIR / STS_BASELINE_OUTPUT_DIR" >&2
    exit 2
  fi
done

if [[ -n "${STS_PYTHON:-}" ]]; then
  breakdown_python="$STS_PYTHON"
  if [[ "$breakdown_python" != /* || ! -x "$breakdown_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  breakdown_python="$(command -v python)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-4}}"
export PYTHONUNBUFFERED=1

trap 'breakdown_status=$?; echo "[job] finished with status $breakdown_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} host=$(hostname)"
echo "[job] python=$breakdown_python config=$breakdown_config"
echo "[job] judge=$judge_output baseline=$baseline_output"

breakdown_command=(
  "$breakdown_python" -u -m sts.cli breakdown
  --config "$breakdown_config"
  --output-dir "$judge_output"
  --datasets "${breakdown_datasets[@]}"
  --baseline "$baseline_output"
  --baseline-models "${baseline_models[@]}"
)
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  breakdown_command+=(--data-root "$STS_DATA_ROOT")
fi

printf '[job] command:'
printf ' %q' "${breakdown_command[@]}"
printf '\n'
"${breakdown_command[@]}"

echo "[job] results: $judge_output/breakdown"
