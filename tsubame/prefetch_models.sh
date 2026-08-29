#!/bin/bash
# -----------------------------------------------------------------------------
# Download model weights into the shared HF cache. Run this ON THE LOGIN NODE.
#
#   bash tsubame/prefetch_models.sh
#
# Compute nodes normally have no route to the internet, so the weights have to
# be on a shared filesystem before any job starts. This is the one step that
# belongs on a login node: it is pure I/O, no GPU and no computation.
#
# Re-running is cheap — the hub client skips files it already has.
# -----------------------------------------------------------------------------
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/activate_env.sh"

# Downloading requires network access, so offline mode is off for this script only.
export HF_HUB_OFFLINE=0
if [[ -n "${STS_HTTP_PROXY// }" ]]; then
  export http_proxy="$STS_HTTP_PROXY" https_proxy="$STS_HTTP_PROXY"
  export HTTP_PROXY="$STS_HTTP_PROXY" HTTPS_PROXY="$STS_HTTP_PROXY"
  echo "[prefetch] proxy: $STS_HTTP_PROXY"
fi

cd "$STS_REPO"

echo
echo "[prefetch] reading model ids from $STS_CONFIG"
python - "$STS_CONFIG" "${STS_MODELS:-}" <<'PYTHON'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from huggingface_hub import snapshot_download
from sts.config import load_config

config_path, wanted = sys.argv[1], sys.argv[2].split()
config = load_config(config_path)
specs = [m for m in config.models if (m.key in wanted if wanted else m.enabled)]
if not specs:
    raise SystemExit(f"no models selected from {config_path} (STS_MODELS={wanted})")

for spec in specs:
    print(f"\n[prefetch] {spec.key}  ->  {spec.model_id}", flush=True)
    path = snapshot_download(spec.model_id)
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[prefetch] cached at {path} ({total / 1024**3:.2f} GiB)", flush=True)
PYTHON

echo
echo "[prefetch] verifying the cache resolves with HF_HUB_OFFLINE=1"
HF_HUB_OFFLINE=1 python - "$STS_CONFIG" "${STS_MODELS:-}" <<'PYTHON'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from huggingface_hub import snapshot_download
from sts.config import load_config
from sts.manifest import resolve_model_revision

config_path, wanted = sys.argv[1], sys.argv[2].split()
config = load_config(config_path)
specs = [m for m in config.models if (m.key in wanted if wanted else m.enabled)]
ok = True
for spec in specs:
    try:
        snapshot_download(spec.model_id, local_files_only=True)
        print(f"  OK   {spec.key:<24} revision {resolve_model_revision(spec.model_id)}")
    except Exception as exc:
        ok = False
        print(f"  FAIL {spec.key:<24} {type(exc).__name__}: {exc}")
raise SystemExit(0 if ok else 1)
PYTHON

echo
echo "[prefetch] done. Compute jobs can now run with HF_HUB_OFFLINE=1."
