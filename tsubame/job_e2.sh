#!/bin/bash
# Condition E2 (sentence pair + another pair's generated images), one stage per
# Grid Engine job. Run from the repository root. All stages use configs/e2_mismatched.yaml.
#   qsub -g <group> -l <resource>=<count> -v STS_PYTHON="$(command -v python)",STS_IMAGE_DIR=<store> tsubame/job_e2.sh judge smoke
#   qsub -g <group> -l <resource>=<count> -l h_rt=12:00:00 -v STS_PYTHON=...,STS_IMAGE_DIR=<store> tsubame/job_e2.sh judge full
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON=...,STS_IMAGE_DIR=<store> tsubame/job_e2.sh report full
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON=...,STS_IMAGE_DIR=<store> tsubame/job_e2.sh compare full
# The images are D1's: pass the STS_IMAGE_DIR the D1 jobs used. compare reads
# the finished D2 and B1 runs named under `comparison:` in the config.
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
#$ -cwd
#$ -N sts-e2
#$ -l h_rt=1:00:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> -l <resource>=<count> [qsub options] tsubame/job_e2.sh STAGE [smoke [N] | full]

Stages (judge is resumable; resubmit the same command to continue):
  judge    Qwen3.5-35B-A3B shown each sentence pair with the two images
           generated for another pair (D2's prompt, unchanged). GPU, a whole
           H100 (bf16 ~70 GB). Reads D1's image store; generates nothing.
  report   Rebuild the summary from the judge's predictions. CPU only.
  compare  E2 against D2 (own images) and B1 (no image) on the same pairs:
           `sts.cli compare`, then `sts.cli breakdown` by SICK NLI label and
           CxC sampling method. Reads finished runs only. CPU only.

Modes:
  smoke [N]  N pairs per dataset (default 32). judge recomputes without reading
             or writing its cache, into <output root>/smoke_N/; compare sets
             those pairs against the full D2 and B1 predictions.
  full       Every pair, into <output root>/full/, honouring resume and caches.
  No mode    Defaults to smoke, 32 pairs per dataset.

Optional environment variables (pass with qsub -v):
  STS_PYTHON          Absolute path to the prepared Python interpreter.
                      If absent, activate the environment via env.sh.
  STS_REPO            Repository root (default: submission directory).
  STS_DATA_ROOT       Override all dataset paths.
  STS_E2_CONFIG       Config file (default: configs/e2_mismatched.yaml).
  STS_E2_OUTPUT_ROOT  Output parent (default: outputs/e2_mismatched in the repo).
  STS_IMAGE_DIR       Image store (default: .cache/images in the repo). Use the
                      same directory as the D1 jobs: E2 reads D1's images.
  STS_E2_CACHE_DIR    Judgment cache (default: .cache/e2 in the repo).
  STS_E2_DATASETS     Space-separated subset of the config's datasets, to split
                      the judge over several concurrent jobs, e.g.
                      "cxc-val" | "cxc-test" | "sick-all sick-test sts3k".
                      Concurrent judge jobs get a cache directory of their own
                      unless STS_E2_CACHE_DIR is set.
EOF
}

e2_stage="${1:-}"
case "$e2_stage" in
  judge|report|compare) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

e2_mode="${2:-smoke}"
case "$e2_mode" in
  smoke)
    e2_limit="${3:-32}"
    if [[ $# -gt 3 || ! "$e2_limit" =~ ^[1-9][0-9]*$ ]]; then
      echo "[error] use: job_e2.sh $e2_stage smoke [positive integer]" >&2
      exit 2
    fi
    e2_output_name="smoke_$e2_limit"
    ;;
  full)
    if [[ $# -gt 2 ]]; then
      echo "[error] full mode takes no sample limit" >&2
      exit 2
    fi
    e2_output_name="full"
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

e2_config="${STS_E2_CONFIG:-$STS_REPO/configs/e2_mismatched.yaml}"
e2_output="${STS_E2_OUTPUT_ROOT:-$STS_REPO/outputs/e2_mismatched}/$e2_output_name"
e2_images="${STS_IMAGE_DIR:-$STS_REPO/.cache/images}"
e2_cache="${STS_E2_CACHE_DIR:-$STS_REPO/.cache/e2}"
e2_datasets=()
if [[ -n "${STS_E2_DATASETS:-}" ]]; then
  read -r -a e2_datasets <<< "$STS_E2_DATASETS"
  if [[ -z "${STS_E2_CACHE_DIR:-}" ]]; then
    # Two jobs flushing into one cache directory could race on a shard name.
    e2_split="$(IFS=+; echo "${e2_datasets[*]}")"
    e2_cache="$STS_REPO/.cache/e2/split__$e2_split"
  fi
fi

if [[ -n "${STS_PYTHON:-}" ]]; then
  e2_python="$STS_PYTHON"
  if [[ "$e2_python" != /* || ! -x "$e2_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  export STS_CONFIG="$e2_config"
  export STS_OUTPUT_DIR="$e2_output"
  export STS_CACHE_DIR="$e2_cache"
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  e2_python="$(command -v python)"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-8}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$e2_output"

trap 'e2_status=$?; echo "[job] finished with status $e2_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} stage=$e2_stage mode=$e2_mode host=$(hostname)"
echo "[job] python=$e2_python config=$e2_config"
echo "[job] output=$e2_output images=$e2_images cache=$e2_cache"

if [[ "$e2_stage" == judge ]]; then
  # Report the CUDA device visible to this process (a MIG slice included).
  "$e2_python" - <<'PYTHON'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable; check the GPU allocation and Python environment")
gpu = torch.cuda.get_device_properties(0)
print(f"[gpu] {gpu.name}: {gpu.total_memory / 1024**3:.2f} GiB visible VRAM", flush=True)
PYTHON
fi

e2_common=(--config "$e2_config" --output-dir "$e2_output")
if [[ ${#e2_datasets[@]} -gt 0 ]]; then
  e2_common+=(--datasets "${e2_datasets[@]}")
fi
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  e2_common+=(--data-root "$STS_DATA_ROOT")
fi

run() {
  printf '[job] command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

case "$e2_stage" in
  judge)
    e2_command=("$e2_python" -u -m sts.cli judge "${e2_common[@]}" --image-dir "$e2_images"
                --device cuda --dtype bfloat16 --cache-dir "$e2_cache" --run-tag "judge-${JOB_ID:-manual}")
    if [[ "$e2_mode" == smoke ]]; then
      e2_command+=(--limit "$e2_limit" --no-resume --no-cache)
    else
      # --limit 0 means no subsampling, even if the YAML has a leftover limit.
      e2_command+=(--limit 0)
    fi
    run "${e2_command[@]}"
    ;;
  report)
    e2_command=("$e2_python" -u -m sts.cli report "${e2_common[@]}" --image-dir "$e2_images")
    if [[ "$e2_mode" == smoke ]]; then
      e2_command+=(--limit "$e2_limit")
    fi
    run "${e2_command[@]}"
    ;;
  compare)
    # Both read the pairs the judge scored, whatever the mode; no --limit needed.
    run "$e2_python" -u -m sts.cli compare "${e2_common[@]}"
    run "$e2_python" -u -m sts.cli breakdown "${e2_common[@]}"
    ;;
esac

echo "[job] results: $e2_output"
