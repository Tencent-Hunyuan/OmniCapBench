#!/usr/bin/env python3
"""Download the local open-source captioners evaluated by OmniCapBench.

Usage:
    python -m omnicapbench_eval.download_models --list
    python -m omnicapbench_eval.download_models --all
    python -m omnicapbench_eval.download_models --model avocado

Set HF_TOKEN for gated repositories, and HF_ENDPOINT to use a mirror.
Checkpoints land in OMNICAP_EVAL_MODELS_DIR, which the model presets in
eval_config.py read as well.
"""

import os
import time
import argparse
import inspect
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

MODELS_DIR = os.environ.get(
    "OMNICAP_EVAL_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)

# Model registry: name -> (repo_id, local_dir_name)
MODEL_REGISTRY = {
    "asid_caption_3b": ("HVision-NKU/ASID-Captioner-3B", "ASID-Captioner-3B"),
    "asid_caption_7b": ("HVision-NKU/ASID-Captioner-7B", "ASID-Captioner-7B"),
    "avocado": ("AVoCaDO-Captioner/AVoCaDO", "AVoCaDO"),
    "timechat_captioner": ("yaolinli/TimeChat-Captioner-GRPO-7B", "TimeChat-Captioner-GRPO-7B"),
    "ugc_videocaptioner": ("openinterx/UGC-VideoCaptioner", "UGC-VideoCaptioner"),
    "qwen25_omni_7b": ("Qwen/Qwen2.5-Omni-7B", "Qwen2.5-Omni-7B"),
}

MAX_RETRIES = 8


def download_model(name: str, repo_id: str, local_dir: str):
    """Download a single model with retry logic."""
    os.makedirs(local_dir, exist_ok=True)

    sig = inspect.signature(snapshot_download)
    common_kwargs = dict(repo_id=repo_id, max_workers=16)

    if "token" in sig.parameters:
        common_kwargs["token"] = os.environ.get("HF_TOKEN", "")
    if "resume_download" in sig.parameters:
        common_kwargs["resume_download"] = True

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"  Repo: {repo_id}")
    print(f"  Local: {local_dir}")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(common_kwargs)
            if "local_dir" in sig.parameters:
                kwargs["local_dir"] = local_dir
                if "local_dir_use_symlinks" in sig.parameters:
                    kwargs["local_dir_use_symlinks"] = False
            else:
                kwargs["cache_dir"] = local_dir

            snapshot_path = snapshot_download(**kwargs)
            print(f"✅ {name} downloaded successfully: {local_dir}")
            print(f"   snapshot path: {snapshot_path}")
            return True

        except Exception as e:
            print(f"❌ Attempt {attempt}/{MAX_RETRIES} failed for {name}: {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                print(f"⚠️  Failed to download {name} after {MAX_RETRIES} attempts")
                return False
            sleep_sec = min(300, 15 * attempt)
            print(f"⏳ Retrying in {sleep_sec}s...")
            time.sleep(sleep_sec)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download the local open-source captioners evaluated by OmniCapBench")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()),
                        help="Download a specific model")
    parser.add_argument("--all", action="store_true",
                        help="Download all models")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, (repo_id, dir_name) in MODEL_REGISTRY.items():
            local_dir = os.path.join(MODELS_DIR, dir_name)
            exists = "✅" if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 2 else "❌"
            print(f"  {exists} {name:25s} {repo_id:45s} -> {dir_name}")
        return

    if not args.model and not args.all:
        parser.print_help()
        return

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set. Public repositories will still download, "
              "but gated ones will fail. Get a token at "
              "https://huggingface.co/settings/tokens")

    models_to_download = []
    if args.all:
        models_to_download = list(MODEL_REGISTRY.keys())
    else:
        models_to_download = [args.model]

    results = {}
    for name in models_to_download:
        repo_id, dir_name = MODEL_REGISTRY[name]
        local_dir = os.path.join(MODELS_DIR, dir_name)

        # Skip if already downloaded
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 5:
            print(f"⏭️  {name} already exists at {local_dir}, skipping.")
            results[name] = True
            continue

        results[name] = download_model(name, repo_id, local_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary:")
    for name, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  {status} {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
