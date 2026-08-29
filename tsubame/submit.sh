#!/bin/bash
# -----------------------------------------------------------------------------
# Submit one independent PBS job per model.
#
#   bash tsubame/submit.sh                          # every model in STS_MODELS
#   bash tsubame/submit.sh jina-v5-omni-small       # just this one
#   STS_WALLTIME=8:00:00 bash tsubame/submit.sh     # override any env.sh value
#   DRY_RUN=1 bash tsubame/submit.sh                # print the qsub lines only
#
# Separate jobs mean one model failing or running out of walltime never blocks
# the others, and each can be resubmitted on its own to resume.
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `source` is a POSIX special builtin: under `set -e` a missing file aborts the
# shell outright, even on the left of `||`. So test before sourcing.
if [[ -f "$HERE/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HERE/env.sh"
elif [[ -f "$HERE/env.sh.example" ]]; then
  echo "[warn] using env.sh.example — copy it to tsubame/env.sh and set STS_GROUP." >&2
  # shellcheck source=/dev/null
  source "$HERE/env.sh.example"
else
  echo "[error] no tsubame/env.sh or env.sh.example found" >&2
  exit 1
fi

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  read -r -a MODELS <<< "$STS_MODELS"
fi
[[ ${#MODELS[@]} -gt 0 ]] || { echo "[error] no models to submit" >&2; exit 1; }

mkdir -p "$STS_LOG_DIR"

echo "queue=$STS_QUEUE select=$STS_SELECT walltime=$STS_WALLTIME group=${STS_GROUP:-<trial run>}"
echo "config=$STS_CONFIG output=$STS_OUTPUT_DIR"
echo

for model in "${MODELS[@]}"; do
  # Everything the job needs, passed explicitly so the job script stays generic.
  vars="STS_REPO=$STS_REPO"
  vars+=",STS_CONFIG=$STS_CONFIG"
  vars+=",STS_MODEL=$model"
  vars+=",STS_DATA_ROOT=$STS_DATA_ROOT"
  vars+=",STS_OUTPUT_DIR=$STS_OUTPUT_DIR"
  vars+=",STS_CACHE_DIR=$STS_CACHE_DIR"
  vars+=",STS_LOG_DIR=$STS_LOG_DIR"
  vars+=",STS_BATCH_SIZE=$STS_BATCH_SIZE"
  vars+=",STS_DTYPE=$STS_DTYPE"
  vars+=",HF_HOME=$HF_HOME"
  vars+=",HF_HUB_CACHE=$HF_HUB_CACHE"
  vars+=",HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

  args=(
    -N "sts-$model"
    -q "$STS_QUEUE"
    -l "select=$STS_SELECT"
    -l "walltime=$STS_WALLTIME"
    # A trailing slash makes PBS name the file itself: sts-<model>.o<jobid>.
    # PBS does not expand $PBS_JOBID inside -o, so don't try to build the name.
    -o "$STS_LOG_DIR/"
    -j oe
    -v "$vars"
  )
  # Omitting -P submits a trial run, which needs no group.
  [[ -n "${STS_GROUP// }" ]] && args+=(-P "$STS_GROUP")
  args+=("$HERE/job_sts.sh")

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'qsub'; printf ' %q' "${args[@]}"; printf '\n'
  else
    jobid="$(qsub "${args[@]}")"
    echo "submitted $model -> $jobid"
  fi
done

echo
echo "watch:   qstat -u \$USER          (or: qstat -f <jobid>)"
echo "cancel:  qdel <jobid>"
echo "logs:    $STS_LOG_DIR/"
echo "merge:   python -m sts.cli report --config $STS_CONFIG --output-dir $STS_OUTPUT_DIR"
