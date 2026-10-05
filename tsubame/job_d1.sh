#!/bin/bash
# Condition D1 (generated image only), one stage per Grid Engine job. Run from
# the repository root. All stages use configs/d1_image_only.yaml.
#   qsub -g <group> -l <resource>=<count> -v STS_PYTHON="$(command -v python)" tsubame/job_d1.sh imagine smoke
#   qsub -g <group> -l <resource>=<count> -l h_rt=12:00:00 -v STS_PYTHON="$(command -v python)" tsubame/job_d1.sh imagine full
#   qsub -g <group> -l <resource>=<count> -l h_rt=4:00:00 -hold_jid <imagine id> -v STS_PYTHON=... tsubame/job_d1.sh embed full
#   qsub -g <group> -l <resource>=<count> -l h_rt=12:00:00 -hold_jid <imagine id> -v STS_PYTHON=... tsubame/job_d1.sh judge full
#   qsub -g <group> -l <cpu resource>=1 -v STS_PYTHON=... tsubame/job_d1.sh report full
# STS_PYTHON selects an already prepared interpreter. When omitted, use the
# existing tsubame/activate_env.sh and its env.sh configuration instead.
#$ -cwd
#$ -N sts-d1
#$ -l h_rt=1:00:00
#$ -j y

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qsub -g <group> -l <resource>=<count> [qsub options] tsubame/job_d1.sh STAGE [smoke [N] | full]

Stages (each resumable; resubmit the same command to continue):
  imagine  Generate one image per unique sentence with FLUX.2 [klein] into the
           image store. GPU; bf16 weights are ~15 GB, so a half-GPU (MIG)
           allocation should hold it.
  embed    jina-v5-omni cosine between the two generated images. GPU, small.
  judge    Qwen3.5-35B-A3B shown the two images. GPU, a whole H100 (bf16 ~70 GB).
  report   Merge the embed and judge results into one summary. CPU only.

Modes:
  smoke [N]  N pairs per dataset (default 32). imagine writes into the shared
             image store (the same images the full run uses, so nothing is
             wasted); embed and judge recompute without reading or writing
             their caches, into <output root>/smoke_N/.
  full       Every pair, into <output root>/full/, honouring resume and caches.
  No mode    Defaults to smoke, 32 pairs per dataset.

Optional environment variables (pass with qsub -v):
  STS_PYTHON          Absolute path to the prepared Python interpreter.
                      If absent, activate the environment via env.sh.
  STS_REPO            Repository root (default: submission directory).
  STS_DATA_ROOT       Override all dataset paths.
  STS_D1_CONFIG       Config file (default: configs/d1_image_only.yaml).
  STS_D1_OUTPUT_ROOT  Output parent (default: outputs/d1_image_only in the repo).
  STS_IMAGE_DIR       Image store (default: .cache/images in the repo). Put it
                      on group storage: the full run is ~60k images.
  STS_D1_CACHE_DIR    Embedding/judgment cache (default: .cache/d1 in the repo).
  STS_D1_DATASETS     Space-separated subset of the config's datasets, to split
                      one stage over several concurrent jobs, e.g.
                      "cxc-val" | "cxc-test" | "sick-all sick-test sts3k".
                      Concurrent imagine jobs may share one store. Concurrent
                      embed/judge jobs get a cache directory of their own
                      unless STS_D1_CACHE_DIR is set.
EOF
}

d1_stage="${1:-}"
case "$d1_stage" in
  imagine|embed|judge|report) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

d1_mode="${2:-smoke}"
case "$d1_mode" in
  smoke)
    d1_limit="${3:-32}"
    if [[ $# -gt 3 || ! "$d1_limit" =~ ^[1-9][0-9]*$ ]]; then
      echo "[error] use: job_d1.sh $d1_stage smoke [positive integer]" >&2
      exit 2
    fi
    d1_output_name="smoke_$d1_limit"
    ;;
  full)
    if [[ $# -gt 2 ]]; then
      echo "[error] full mode takes no sample limit" >&2
      exit 2
    fi
    d1_output_name="full"
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

d1_config="${STS_D1_CONFIG:-$STS_REPO/configs/d1_image_only.yaml}"
d1_output="${STS_D1_OUTPUT_ROOT:-$STS_REPO/outputs/d1_image_only}/$d1_output_name"
d1_images="${STS_IMAGE_DIR:-$STS_REPO/.cache/images}"
d1_cache="${STS_D1_CACHE_DIR:-$STS_REPO/.cache/d1}"
d1_datasets=()
if [[ -n "${STS_D1_DATASETS:-}" ]]; then
  read -r -a d1_datasets <<< "$STS_D1_DATASETS"
  if [[ -z "${STS_D1_CACHE_DIR:-}" ]]; then
    # Two jobs flushing into one cache directory could race on a shard name.
    d1_split="$(IFS=+; echo "${d1_datasets[*]}")"
    d1_cache="$STS_REPO/.cache/d1/split__$d1_split"
  fi
fi

if [[ -n "${STS_PYTHON:-}" ]]; then
  d1_python="$STS_PYTHON"
  if [[ "$d1_python" != /* || ! -x "$d1_python" ]]; then
    echo "[error] STS_PYTHON must be an absolute path to an executable Python" >&2
    exit 2
  fi
else
  export STS_CONFIG="$d1_config"
  export STS_OUTPUT_DIR="$d1_output"
  export STS_CACHE_DIR="$d1_cache"
  source "$STS_REPO/tsubame/activate_env.sh"
  cd "$STS_REPO"
  d1_python="$(command -v python)"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-8}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$d1_output" "$d1_images"

trap 'd1_status=$?; echo "[job] finished with status $d1_status at $(date +%Y-%m-%dT%H:%M:%S%z)"' EXIT

echo "[job] id=${JOB_ID:-manual} stage=$d1_stage mode=$d1_mode host=$(hostname)"
echo "[job] python=$d1_python config=$d1_config"
echo "[job] output=$d1_output images=$d1_images cache=$d1_cache"

if [[ "$d1_stage" != report ]]; then
  # Report the CUDA device visible to this process (a MIG slice included).
  "$d1_python" - <<'PYTHON'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable; check the GPU allocation and Python environment")
gpu = torch.cuda.get_device_properties(0)
print(f"[gpu] {gpu.name}: {gpu.total_memory / 1024**3:.2f} GiB visible VRAM", flush=True)
PYTHON
fi

case "$d1_stage" in
  imagine) d1_subcommand=imagine ;;
  embed)   d1_subcommand=run ;;
  judge)   d1_subcommand=judge ;;
  report)  d1_subcommand=report ;;
esac

d1_command=(
  "$d1_python" -u -m sts.cli "$d1_subcommand"
  --config "$d1_config"
  --output-dir "$d1_output"
  --image-dir "$d1_images"
)
case "$d1_stage" in
  imagine)
    d1_command+=(--device cuda --run-tag "${JOB_ID:-manual}")
    ;;
  embed|judge)
    d1_command+=(--device cuda --dtype bfloat16 --cache-dir "$d1_cache"
                 --run-tag "$d1_stage-${JOB_ID:-manual}")
    ;;
esac
if [[ ${#d1_datasets[@]} -gt 0 ]]; then
  d1_command+=(--datasets "${d1_datasets[@]}")
fi
if [[ -n "${STS_DATA_ROOT:-}" ]]; then
  d1_command+=(--data-root "$STS_DATA_ROOT")
fi
if [[ "$d1_mode" == smoke ]]; then
  d1_command+=(--limit "$d1_limit")
  if [[ "$d1_stage" == embed || "$d1_stage" == judge ]]; then
    d1_command+=(--no-resume --no-cache)
  fi
elif [[ "$d1_stage" != report ]]; then
  # --limit 0 means no subsampling, even if the YAML has a leftover limit.
  d1_command+=(--limit 0)
fi

printf '[job] command:'
printf ' %q' "${d1_command[@]}"
printf '\n'
"${d1_command[@]}"

echo "[job] results: $d1_output"
