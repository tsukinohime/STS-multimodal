#!/bin/bash
# Shared bootstrap: load site config, load modules, activate the Python env.
# Sourced (not executed) by prefetch_models.sh, job_sts.sh and submit.sh.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$_here/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_here/env.sh"
elif [[ -f "$_here/env.sh.example" ]]; then
  echo "[warn] tsubame/env.sh not found; falling back to env.sh.example defaults." >&2
  echo "[warn] cp tsubame/env.sh.example tsubame/env.sh and edit it." >&2
  # shellcheck source=/dev/null
  source "$_here/env.sh.example"
else
  echo "[error] no tsubame/env.sh or env.sh.example found" >&2
  return 1 2>/dev/null || exit 1
fi

# ---- modules ----------------------------------------------------------------
if [[ -n "${STS_MODULES// }" ]]; then
  if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $STS_MODULES || { echo "[error] module load $STS_MODULES failed" >&2; exit 1; }
    echo "[env] modules: $STS_MODULES"
  else
    echo "[warn] STS_MODULES set but no 'module' command available; skipping." >&2
  fi
fi

# ---- python environment -----------------------------------------------------
case "${STS_ENV_KIND:-none}" in
  conda)
    conda_sh="$STS_CONDA_SH"
    if [[ -z "$conda_sh" ]]; then
      if command -v conda >/dev/null 2>&1; then
        conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
      fi
    fi
    if [[ -f "$conda_sh" ]]; then
      # shellcheck source=/dev/null
      source "$conda_sh"
    else
      echo "[error] conda profile not found. Set STS_CONDA_SH in tsubame/env.sh" >&2
      echo "        (it is \$(conda info --base)/etc/profile.d/conda.sh)" >&2
      exit 1
    fi
    conda activate "$STS_CONDA_ENV" || { echo "[error] conda activate $STS_CONDA_ENV failed" >&2; exit 1; }
    echo "[env] conda env: $STS_CONDA_ENV"
    ;;
  venv)
    [[ -f "$STS_VENV/bin/activate" ]] || { echo "[error] no venv at $STS_VENV" >&2; exit 1; }
    # shellcheck source=/dev/null
    source "$STS_VENV/bin/activate"
    echo "[env] venv: $STS_VENV"
    ;;
  none)
    echo "[env] using the ambient python"
    ;;
  *)
    echo "[error] STS_ENV_KIND must be conda, venv or none (got '$STS_ENV_KIND')" >&2
    exit 1
    ;;
esac

mkdir -p "$STS_LOG_DIR" "$STS_OUTPUT_DIR" "$STS_CACHE_DIR" "$HF_HUB_CACHE"

echo "[env] python : $(command -v python) ($(python -V 2>&1))"
echo "[env] repo   : $STS_REPO"
echo "[env] data   : $STS_DATA_ROOT"
echo "[env] output : $STS_OUTPUT_DIR"
echo "[env] hf cache: $HF_HUB_CACHE (offline=${HF_HUB_OFFLINE:-0})"
