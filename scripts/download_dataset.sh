#!/usr/bin/env bash
# Download the OmniCapBench benchmark data from Hugging Face.
#
# Usage:
#   bash scripts/download_dataset.sh [TARGET_DIR]
#
# Produces:
#   TARGET_DIR/eval.jsonl            annotations in the layout the toolkit expects
#   TARGET_DIR/test/metadata.jsonl   same records with the Dataset Viewer field names
#   TARGET_DIR/test/videos/          786 video files (~12 GB)

set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ID="OmniCapBench/OmniCapBench"
TARGET_DIR="${1:-data/OmniCapBench}"

if ! "$PYTHON" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "huggingface_hub is not installed. Run: pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Downloading ${REPO_ID} into ${TARGET_DIR} (about 12 GB) ..."
"$PYTHON" - "$REPO_ID" "$TARGET_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
)
print("Downloaded to: %s" % path)
PY

echo
echo "Done. Point the evaluation at the annotations with:"
echo "  --input ${TARGET_DIR}/eval.jsonl"
echo
echo "Video paths inside eval.jsonl are relative, so set the media root with:"
echo "  export OMNICAP_EVAL_VIDEO_ROOT=${TARGET_DIR}/test"
