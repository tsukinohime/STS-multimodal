#!/bin/bash
# Late fusion (condition LF) of finished runs' predictions, then the fused
# systems broken down by category: CPU only, no model, about a minute for the
# full datasets (~1.5 GB memory). Run from the repository root.
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON="$(command -v python)" tsubame/job_fusion.sh
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
#$ -cwd
#$ -N sts-fusion
#$ -l h_rt=0:30:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> -l <cpu resource>=1 [qsub options] tsubame/job_fusion.sh

Reads the per-pair predictions of the finished runs named in the config
(default configs/late_fusion.yaml: A0 text_only, B1 llm_judge/full, D1
d1_image_only/full), fuses text and image scores, and writes predictions,
metrics, summary, fusion_comparison and alpha_sweep to the config's output
directory (default outputs/late_fusion). Then `sts.cli breakdown` splits the
fused systems by category, SICK by NLI label and CxC by sampling_method, into
<output directory>/breakdown/. Nothing is resumable or cached: a rerun
recomputes everything in about a minute.

Optional environment variables (pass with qsub -v):
  STS_PYTHON             Absolute path to the prepared Python interpreter.
                         If absent, activate the environment via env.sh.
  STS_REPO               Repository root (default: submission directory).
  STS_DATA_ROOT          Override all dataset paths.
  STS_FUSION_CONFIG      Config file (default: configs/late_fusion.yaml).
  STS_FUSION_OUTPUT_DIR  Output directory (default: the config's).
  STS_FUSION_SYSTEMS     Space-separated fusion keys to run (default: all), e.g.
                         "lf-judge". The breakdown covers the same systems.
  STS_BREAKDOWN_DATASETS Space-separated datasets to break down
                         (default: sick-all sick-test cxc-val cxc-test;
                         STS3k has no categories).
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

# Grid Engine spools the script away from the repository, so do not infer the
# repository from BASH_SOURCE. Submit from the repo root or pass STS_REPO.
cd "${STS_REPO:-${SGE_O_WORKDIR:-$PWD}}"
export STS_REPO="$PWD"

fusion_config="${STS_FUSION_CONFIG:-$STS_REPO/configs/late_fusion.yaml}"

if [[ -n "${STS_PYTHON:-}" ]]; then
  fusion_python="$STS_PYTHON"
  if [[ "$fusion_python" != /* || ! -x "$fusion_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  fusion_python="$(command -v python)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-4}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

trap 'fusion_status=$?; echo "[job] finished with status $fusion_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} host=$(hostname)"
echo "[job] python=$fusion_python config=$fusion_config"

# Shared by both steps, so the breakdown reads exactly what fuse wrote.
fusion_common=(--config "$fusion_config")
if [[ -n "${STS_FUSION_OUTPUT_DIR:-}" ]]; then
  fusion_common+=(--output-dir "$STS_FUSION_OUTPUT_DIR")
fi
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  fusion_common+=(--data-root "$STS_DATA_ROOT")
fi
if [[ -n "${STS_FUSION_SYSTEMS:-}" ]]; then
  read -r -a fusion_systems <<< "$STS_FUSION_SYSTEMS"
  fusion_common+=(--models "${fusion_systems[@]}")
fi
read -r -a breakdown_datasets <<< "${STS_BREAKDOWN_DATASETS:-sick-all sick-test cxc-val cxc-test}"

run() {
  printf '[job] command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run "$fusion_python" -u -m sts.cli fuse "${fusion_common[@]}"
# -text and -image are the two sources on the same pairs, so no --baseline.
run "$fusion_python" -u -m sts.cli breakdown "${fusion_common[@]}" --datasets "${breakdown_datasets[@]}"
