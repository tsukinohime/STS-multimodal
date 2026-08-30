#!/bin/bash
# -----------------------------------------------------------------------------
# Submit one independent Grid Engine job per model.
#
#   bash tsubame/submit.sh                          # every model in STS_MODELS
#   bash tsubame/submit.sh jina-v5-omni-small       # just this one
#   STS_H_RT=8:00:00 bash tsubame/submit.sh         # override any env.sh value
#   DRY_RUN=1 bash tsubame/submit.sh                # print the qsub lines only
#
# Separate jobs mean one model failing or running out of h_rt never blocks the
# others, and each can be resubmitted on its own to resume.
#
# Grid Engine, not PBS: resources are `-l <type>=<n>`, the runtime limit is
# `-l h_rt=`, the accounting group is `-g`, and stderr merges with `-j y`.
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

echo "resource=$STS_RESOURCE=$STS_RESOURCE_COUNT  h_rt=$STS_H_RT  group=${STS_GROUP:-<trial run>}"
echo "config=$STS_CONFIG  output=$STS_OUTPUT_DIR"
echo

for model in "${MODELS[@]}"; do
  # Everything the job needs, passed explicitly so the job script stays generic.
  # Grid Engine's -v takes a comma-separated list; none of these values may
  # contain a comma (paths do not).
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
    -cwd
    -l "${STS_RESOURCE}=${STS_RESOURCE_COUNT}"
    -l "h_rt=$STS_H_RT"
    # A trailing slash makes Grid Engine name the file itself:
    # sts-<model>.o<jobid>. $JOB_ID is not expanded inside -o, so don't build
    # the name here.
    -o "$STS_LOG_DIR/"
    -j y
    -v "$vars"
  )
  # Omitting -g submits a trial run, which needs no group.
  [[ -n "${STS_GROUP// }" ]] && args+=(-g "$STS_GROUP")
  args+=("$HERE/job_sts.sh")

  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'qsub'; printf ' %q' "${args[@]}"; printf '\n'
  else
    jobid="$(qsub "${args[@]}")"
    echo "submitted $model -> $jobid"
  fi
done

echo
echo "watch:   qstat -u \$USER          (details: qstat -j <jobid>)"
echo "cancel:  qdel <jobid>"
echo "logs:    $STS_LOG_DIR/"
echo "merge:   python -m sts.cli report --config $STS_CONFIG --output-dir $STS_OUTPUT_DIR"
