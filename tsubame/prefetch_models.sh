#!/bin/bash
# -----------------------------------------------------------------------------
# Download model weights into the shared HF cache. Run this ON THE LOGIN NODE.
#
#   bash tsubame/prefetch_models.sh
#   STS_CONFIG=configs/d1_image_only.yaml bash tsubame/prefetch_models.sh   # + FLUX.2 [klein]
#
# Compute nodes normally have no route to the internet, so the weights have to
# be on a shared filesystem before any job starts. This is the one step that
# belongs on a login node: it is pure I/O, no GPU and no computation.
#
# Selected from $STS_CONFIG: the encoders (STS_MODELS, or the enabled ones), the
# enabled judges and the image generator. Ids that are local directories
# (weights downloaded to group storage yourself) are not downloaded; the check
# at the end confirms they are complete.
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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from huggingface_hub import snapshot_download
from sts.config import load_config

config_path, wanted = sys.argv[1], sys.argv[2].split()
config = load_config(config_path)
specs = [(m.key, m.model_id, "hub") for m in config.models if (m.key in wanted if wanted else m.enabled)]
specs += [(j.key, j.model_id, "hub") for j in config.judges if j.enabled]
generator = config.images.generator
if generator is not None and generator.backend == "flux2-klein":
    specs.append((generator.key, generator.model_id, "diffusers"))
if not specs:
    raise SystemExit(f"no models selected from {config_path} (STS_MODELS={wanted})")
local = [s for s in specs if os.path.isdir(s[1]) or s[1].startswith("/")]
for key, model_id, _ in local:
    print(f"\n[prefetch] {key}  ->  {model_id}  (local directory: not downloaded, checked below)", flush=True)

for key, model_id, kind in (s for s in specs if s not in local):
    print(f"\n[prefetch] {key}  ->  {model_id}", flush=True)
    if kind == "diffusers":
        # The pipeline's component folders only, not the repo's extra
        # single-file checkpoint (another 7 GB for FLUX.2-klein-4B).
        from diffusers import DiffusionPipeline

        path = DiffusionPipeline.download(model_id)
    else:
        path = snapshot_download(model_id)
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[prefetch] cached at {path} ({total / 1024**3:.2f} GiB)", flush=True)
PYTHON

echo
echo "[prefetch] verifying the cache resolves with HF_HUB_OFFLINE=1, and local directories are complete"
HF_HUB_OFFLINE=1 python - "$STS_CONFIG" "${STS_MODELS:-}" <<'PYTHON'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from huggingface_hub import snapshot_download
from sts.config import load_config
from sts.manifest import resolve_model_revision


def missing_parts(path: Path, kind: str) -> list:
    """What a local model directory lacks; empty when it looks complete."""
    if not path.is_dir():
        return ["the directory itself"]
    if kind == "diffusers":
        if not (path / "model_index.json").is_file():
            return ["model_index.json (not a diffusers pipeline folder)"]
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        names = [name for name, value in index.items() if isinstance(value, list) and value[-1] is not None]
        missing = [f"{name}/" for name in names if not (path / name).is_dir()]
        folders = [path / name for name in names if (path / name).is_dir()]
    else:
        missing = [] if (path / "config.json").is_file() else ["config.json"]
        folders = [path]
    for folder in folders:
        if not (folder / "config.json").is_file():
            continue  # tokenizer, scheduler: no weights
        shard_index = next(folder.glob("*.safetensors.index.json"), None)
        if shard_index is not None:
            shards = sorted(set(json.loads(shard_index.read_text(encoding="utf-8"))["weight_map"].values()))
            missing += [str((folder / s).relative_to(path)) for s in shards if not (folder / s).is_file()]
        elif not any(folder.glob("*.safetensors")) and not any(folder.glob("*.bin")):
            missing.append(f"{folder.relative_to(path)}/ weights" if folder != path else "weights")
    if any((path / ".cache" / "huggingface" / "download").rglob("*.incomplete")):
        missing.append("files of an interrupted download (rerun it)")
    return missing


config_path, wanted = sys.argv[1], sys.argv[2].split()
config = load_config(config_path)
specs = [(m.key, m.model_id, "hub") for m in config.models if (m.key in wanted if wanted else m.enabled)]
specs += [(j.key, j.model_id, "hub") for j in config.judges if j.enabled]
generator = config.images.generator
if generator is not None and generator.backend == "flux2-klein":
    specs.append((generator.key, generator.model_id, "diffusers"))
ok = True
for key, model_id, kind in specs:
    if os.path.isdir(model_id) or model_id.startswith("/"):
        missing = missing_parts(Path(model_id), kind)
        if missing:
            ok = False
            print(f"  FAIL {key:<24} local {model_id}: missing {', '.join(missing)}")
            if kind == "diffusers":
                print("       download the pipeline folders there on this node, e.g. for FLUX.2-klein-4B:\n"
                      "       hf download black-forest-labs/FLUX.2-klein-4B --exclude flux-2-klein-4b.safetensors "
                      f"--local-dir {model_id}")
        else:
            size = sum(f.stat().st_size for f in Path(model_id).rglob("*") if f.is_file())
            print(f"  OK   {key:<24} local {model_id} ({size / 1024**3:.1f} GiB), "
                  f"revision {resolve_model_revision(model_id)}")
        continue
    try:
        if kind == "diffusers":
            from diffusers import DiffusionPipeline

            DiffusionPipeline.download(model_id, local_files_only=True)
        else:
            snapshot_download(model_id, local_files_only=True)
        print(f"  OK   {key:<24} revision {resolve_model_revision(model_id)}")
    except Exception as exc:
        ok = False
        print(f"  FAIL {key:<24} {type(exc).__name__}: {exc}")
raise SystemExit(0 if ok else 1)
PYTHON

echo
echo "[prefetch] done. Compute jobs can now run with HF_HUB_OFFLINE=1."
