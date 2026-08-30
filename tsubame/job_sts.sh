#!/bin/bash
# -----------------------------------------------------------------------------
# Grid Engine job: run the text-only STS experiment for ONE model.
#
# Submit via tsubame/submit.sh, which supplies the resource type, runtime,
# group and model on the qsub command line — nothing site-specific is baked in
# here. The `#$` directives below are only fallbacks for a bare `qsub`.
#
# Resumable: with `resume: true` in the YAML, a job that hits the h_rt limit can
# simply be resubmitted. Finished (model, dataset) predictions are reused, and
# the embedding cache (flushed every `flush_every` texts) means even a
# partly-encoded dataset restarts near where it stopped.
# -----------------------------------------------------------------------------
#$ -cwd
#$ -N sts-text
#$ -l h_rt=1:00:00
#$ -j y

set -euo pipefail

# `#$ -cwd` already starts the job in the submission directory; SGE_O_WORKDIR is
# the belt-and-braces fallback when this script is run without that directive.
cd "${SGE_O_WORKDIR:-$(pwd)}"

source "${STS_REPO:-$(pwd)}/tsubame/activate_env.sh"
cd "$STS_REPO"

MODEL="${STS_MODEL:?STS_MODEL is not set — submit via tsubame/submit.sh}"

echo
echo "=============================================================================="
echo " job      : ${JOB_ID:-<interactive>} (${JOB_NAME:-?})  on $(hostname)"
echo " model    : $MODEL"
echo " config   : $STS_CONFIG"
echo " started  : $(date -Is)"
echo "=============================================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "(no nvidia-smi on this node)"
echo

# Keep BLAS from oversubscribing the cores the scheduler actually granted.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NSLOTS:-8}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -m sts.cli run \
  --config "$STS_CONFIG" \
  --models "$MODEL" \
  --run-tag "$MODEL" \
  --output-dir "$STS_OUTPUT_DIR" \
  --data-root "$STS_DATA_ROOT" \
  --cache-dir "$STS_CACHE_DIR" \
  --batch-size "$STS_BATCH_SIZE" \
  --dtype "$STS_DTYPE" \
  --device cuda

status=$?
echo
echo "[job] finished at $(date -Is) with status $status"
echo "[job] artifacts: $STS_OUTPUT_DIR/{manifest,metrics,summary}__${MODEL}.*"
echo "[job] merge all models into one report with:"
echo "        python -m sts.cli report --config $STS_CONFIG --output-dir $STS_OUTPUT_DIR"
exit $status
