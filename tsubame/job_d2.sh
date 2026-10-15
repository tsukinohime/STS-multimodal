#!/bin/bash
# Condition D2 (sentence pair + generated images), one stage per Grid Engine job.
# Run from the repository root. All stages use configs/d2_text_image.yaml.
#   qsub -g <group> -l <resource>=<count> -v STS_PYTHON="$(command -v python)",STS_IMAGE_DIR=<store> tsubame/job_d2.sh judge smoke
#   qsub -g <group> -l <resource>=<count> -l h_rt=12:00:00 -v STS_PYTHON=...,STS_IMAGE_DIR=<store> tsubame/job_d2.sh judge full
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON=...,STS_IMAGE_DIR=<store> tsubame/job_d2.sh report full
# The images are D1's: pass the STS_IMAGE_DIR the D1 jobs used. Once D1's
# imagine has covered the datasets, D2 needs no imagine stage at all.
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
#$ -cwd
#$ -N sts-d2
#$ -l h_rt=1:00:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> -l <resource>=<count> [qsub options] tsubame/job_d2.sh STAGE [smoke [N] | full]

Stages (each resumable; resubmit the same command to continue):
  imagine  Generate the images still missing from the image store with
           FLUX.2 [klein]. D2 shares D1's store, so after D1's imagine this
           generates nothing; run it only if D2 goes first. GPU.
  judge    Qwen3.5-35B-A3B shown the sentence pair and its two generated
           images. GPU, a whole H100 (bf16 ~70 GB).
  report   Rebuild the summary from the judge's predictions. CPU only.

Modes:
  smoke [N]  N pairs per dataset (default 32). imagine writes into the shared
             image store; judge recomputes without reading or writing its
             cache, into <output root>/smoke_N/.
  full       Every pair, into <output root>/full/, honouring resume and caches.
  No mode    Defaults to smoke, 32 pairs per dataset.

Optional environment variables (pass with qsub -v):
  STS_PYTHON          Absolute path to the prepared Python interpreter.
                      If absent, activate the environment via env.sh.
  STS_REPO            Repository root (default: submission directory).
  STS_DATA_ROOT       Override all dataset paths.
  STS_D2_CONFIG       Config file (default: configs/d2_text_image.yaml).
  STS_D2_OUTPUT_ROOT  Output parent (default: outputs/d2_text_image in the repo).
  STS_IMAGE_DIR       Image store (default: .cache/images in the repo). Use the
                      same directory as the D1 jobs: D2 reads D1's images.
  STS_D2_CACHE_DIR    Judgment cache (default: .cache/d2 in the repo).
  STS_D2_DATASETS     Space-separated subset of the config's datasets, to split
                      one stage over several concurrent jobs, e.g.
                      "cxc-val" | "cxc-test" | "sick-all sick-test sts3k".
                      Concurrent judge jobs get a cache directory of their own
                      unless STS_D2_CACHE_DIR is set.
EOF
}

d2_stage="${1:-}"
case "$d2_stage" in
  imagine|judge|report) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

d2_mode="${2:-smoke}"
case "$d2_mode" in
  smoke)
    d2_limit="${3:-32}"
    if [[ $# -gt 3 || ! "$d2_limit" =~ ^[1-9][0-9]*$ ]]; then
      echo "[error] use: job_d2.sh $d2_stage smoke [positive integer]" >&2
      exit 2
    fi
    d2_output_name="smoke_$d2_limit"
    ;;
  full)
    if [[ $# -gt 2 ]]; then
      echo "[error] full mode takes no sample limit" >&2
      exit 2
    fi
    d2_output_name="full"
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

d2_config="${STS_D2_CONFIG:-$STS_REPO/configs/d2_text_image.yaml}"
d2_output="${STS_D2_OUTPUT_ROOT:-$STS_REPO/outputs/d2_text_image}/$d2_output_name"
d2_images="${STS_IMAGE_DIR:-$STS_REPO/.cache/images}"
d2_cache="${STS_D2_CACHE_DIR:-$STS_REPO/.cache/d2}"
d2_datasets=()
if [[ -n "${STS_D2_DATASETS:-}" ]]; then
  read -r -a d2_datasets <<< "$STS_D2_DATASETS"
  if [[ -z "${STS_D2_CACHE_DIR:-}" ]]; then
    # Two jobs flushing into one cache directory could race on a shard name.
    d2_split="$(IFS=+; echo "${d2_datasets[*]}")"
    d2_cache="$STS_REPO/.cache/d2/split__$d2_split"
  fi
fi

if [[ -n "${STS_PYTHON:-}" ]]; then
  d2_python="$STS_PYTHON"
  if [[ "$d2_python" != /* || ! -x "$d2_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  export STS_CONFIG="$d2_config"
  export STS_OUTPUT_DIR="$d2_output"
  export STS_CACHE_DIR="$d2_cache"
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  d2_python="$(command -v python)"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-8}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$d2_output" "$d2_images"

trap 'd2_status=$?; echo "[job] finished with status $d2_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} stage=$d2_stage mode=$d2_mode host=$(hostname)"
echo "[job] python=$d2_python config=$d2_config"
echo "[job] output=$d2_output images=$d2_images cache=$d2_cache"

if [[ "$d2_stage" != report ]]; then
  # Report the CUDA device visible to this process (a MIG slice included).
  "$d2_python" - <<'PYTHON'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable; check the GPU allocation and Python environment")
gpu = torch.cuda.get_device_properties(0)
print(f"[gpu] {gpu.name}: {gpu.total_memory / 1024**3:.2f} GiB visible VRAM", flush=True)
PYTHON
fi

d2_command=(
  "$d2_python" -u -m sts.cli "$d2_stage"
  --config "$d2_config"
  --output-dir "$d2_output"
  --image-dir "$d2_images"
)
case "$d2_stage" in
  imagine)
    d2_command+=(--device cuda --run-tag "${JOB_ID:-manual}")
    ;;
  judge)
    d2_command+=(--device cuda --dtype bfloat16 --cache-dir "$d2_cache"
                 --run-tag "judge-${JOB_ID:-manual}")
    ;;
esac
if [[ ${#d2_datasets[@]} -gt 0 ]]; then
  d2_command+=(--datasets "${d2_datasets[@]}")
fi
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  d2_command+=(--data-root "$STS_DATA_ROOT")
fi
if [[ "$d2_mode" == smoke ]]; then
  d2_command+=(--limit "$d2_limit")
  if [[ "$d2_stage" == judge ]]; then
    d2_command+=(--no-resume --no-cache)
  fi
elif [[ "$d2_stage" != report ]]; then
  # --limit 0 means no subsampling, even if the YAML has a leftover limit.
  d2_command+=(--limit 0)
fi

printf '[job] command:'
printf ' %q' "${d2_command[@]}"
printf '\n'
"${d2_command[@]}"

echo "[job] results: $d2_output"
