#!/usr/bin/env python3
"""
MTSS Eval - Main Runner
Parallel evaluation inference pipeline with checkpoint/resume support.

Usage:
    python run_eval.py --model gemini_31
    python run_eval.py --model qwen3_omni --workers 50
    python run_eval.py --model qwen25_omni --workers 30
    python run_eval.py --model gemini_25 --resume
    python run_eval.py --model qwen35_omni_plus --start 0 --end 100
    # Local Qwen3-Omni models (use --workers 1 for GPU inference)
    python run_eval.py --model qwen3_omni_instruct --workers 1
    python run_eval.py --model qwen3_omni_thinking --workers 1
    python run_eval.py --model qwen3_omni_captioner --workers 1
    # Local Qwen2.5-Omni models
    python run_eval.py --model qwen25_omni_7b --workers 1
    python run_eval.py --model ugc_videocaptioner --workers 1
    python run_eval.py --model asid_caption_7b --workers 1
    python run_eval.py --model asid_caption_3b --workers 1
    python run_eval.py --model avocado --workers 1
    python run_eval.py --model timechat_captioner --workers 1

Supported model names:
    Closed-source API: gemini_31, gemini_25, qwen35_omni_plus
    Remote API:        qwen3_omni, qwen25_omni
    Local Qwen3-Omni:  qwen3_omni_instruct, qwen3_omni_thinking, qwen3_omni_captioner
    Local Qwen2.5:     qwen25_omni_7b, ugc_videocaptioner, asid_caption_7b, asid_caption_3b, avocado, timechat_captioner
"""

import argparse
import json
import logging
import os
import sys
import time
import fcntl
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import eval_config as config
from .inference_engine import run_inference

# ======================================================================
# Model preset mapping
# ======================================================================

MODEL_PRESETS = {
    # Closed-source API models
    "gemini_31": config.PRESET_GEMINI_31,
    "gemini_25": config.PRESET_GEMINI_25,
    "qwen35_omni_plus": config.PRESET_QWEN35_OMNI_PLUS,
    "qwen35_omni_flash": config.PRESET_QWEN35_OMNI_FLASH,
    # Remote API Qwen-Omni
    "qwen3_omni": config.PRESET_QWEN3_OMNI,
    # Local Qwen3-Omni 30B-A3B (3 variants)
    "qwen3_omni_instruct": config.PRESET_QWEN3_OMNI_INSTRUCT,
    "qwen3_omni_thinking": config.PRESET_QWEN3_OMNI_THINKING,
    "qwen3_omni_captioner": config.PRESET_QWEN3_OMNI_CAPTIONER,
    # Local Qwen2.5-Omni-7B (base)
    "qwen25_omni_7b": config.PRESET_QWEN25_OMNI_7B,
    # Local Qwen2.5-Omni fine-tuned models
    "ugc_videocaptioner": config.PRESET_UGC_VIDEOCAPTIONER,
    "asid_caption_7b": config.PRESET_ASID_CAPTION_7B,
    "asid_caption_3b": config.PRESET_ASID_CAPTION_3B,
    "avocado": config.PRESET_AVOCADO,
    "timechat_captioner": config.PRESET_TIMECHAT_CAPTIONER,
}


# ======================================================================
# Logging setup
# ======================================================================

def setup_logging(model_name: str) -> logging.Logger:
    """Setup logging to file only. Console only shows progress bar + errors."""
    # Organize logs by model: logs/<model>/eval_logs/eval_<timestamp>.log
    model_log_dir = os.path.join(config.LOG_DIR, model_name, "eval_logs")
    os.makedirs(model_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(model_log_dir, f"eval_{timestamp}.log")

    logger = logging.getLogger("mtss_eval")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent accumulation on repeated calls
    logger.handlers.clear()

    # File handler (detailed, all levels)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — only ERROR+ (errors will break through the progress bar)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("❌ %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


# ======================================================================
# Checkpoint management (file-lock based for NFS safety)
# ======================================================================

def get_output_paths(model_name: str):
    """Get output file paths for a given model. Results and cache are separated."""
    model_dir = os.path.join(config.OUTPUT_DIR, model_name)
    cache_dir = os.path.join(config.VIDEO_CACHE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    return {
        "results": os.path.join(model_dir, "results.jsonl"),
        "progress": os.path.join(model_dir, "progress.json"),
        "errors": os.path.join(model_dir, "errors.jsonl"),
        "cache_dir": cache_dir,
    }


def load_completed_ids(results_path: str) -> set:
    """Load set of successfully-completed video paths from results file.
    Only 'success' status is considered completed; failed/exception items
    will be retried on resume."""
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Only skip items that succeeded — retry failed/exception
                    if d.get("status") == "success":
                        vp = d.get("video_path", "")
                        if vp:
                            completed.add(vp)
                except json.JSONDecodeError:
                    continue
    return completed


def save_result_atomic(result_dict: dict, results_path: str):
    """Atomically append a result to the JSONL file with file locking."""
    line = json.dumps(result_dict, ensure_ascii=False) + "\n"
    lock_path = results_path + ".lock"

    # Ensure parent directory exists (NFS may not have synced yet)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def save_error_atomic(error_dict: dict, errors_path: str):
    """Atomically append an error record."""
    line = json.dumps(error_dict, ensure_ascii=False) + "\n"
    lock_path = errors_path + ".lock"

    # Ensure parent directory exists (NFS may not have synced yet)
    os.makedirs(os.path.dirname(errors_path), exist_ok=True)

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(errors_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def save_progress(progress_path: str, total: int, completed: int,
                  skipped: int, failed: int, start_time: float):
    """Save progress summary."""
    elapsed = time.time() - start_time
    progress = {
        "total": total,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_item": round(elapsed / max(completed, 1), 1),
        "last_update": datetime.now().isoformat(),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ======================================================================
# Load input data
# ======================================================================

def load_input_data(input_path: str, start: int = None, end: int = None,
                    indices_file: str = None, video_root: str = "") -> list:
    """Load input JSONL data with optional range slicing or index-based selection.

    Args:
        input_path: Path to the input JSONL file.
        start: Start index (inclusive) for range slicing.
        end: End index (exclusive) for range slicing.
        indices_file: Path to a JSON file containing a list of indices to select.
                      If provided, --start/--end are ignored.
        video_root: Directory that relative video paths are resolved against.
                    The resolved location is stored in "media_path"; "video_path"
                    stays untouched because it is the key used to join
                    predictions against the ground truth.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if video_root:
        for item in data:
            video_path = item.get("video_path", "")
            if video_path and not os.path.isabs(video_path):
                item["media_path"] = os.path.join(video_root, video_path)

    # Index-based selection takes priority over range slicing
    if indices_file:
        with open(indices_file, "r") as f:
            indices = json.load(f)
        selected = [data[i] for i in indices if 0 <= i < len(data)]
        return selected

    if start is not None or end is not None:
        s = start or 0
        e = end or len(data)
        data = data[s:e]

    return data


# ======================================================================
# Worker function
# ======================================================================

def process_one_video(video_item: dict, model_cfg: dict,
                      cache_dir: str, logger: logging.Logger) -> dict:
    """
    Process a single video item. Returns a result dict.
    """
    video_path = video_item.get("video_path", "unknown")
    t0 = time.time()

    try:
        result = run_inference(model_cfg, video_item, cache_dir=cache_dir)
        elapsed = time.time() - t0

        # Build output record
        output = {
            "video_path": video_path,
            "video_url": video_item.get("video_url", ""),
            "duration_sec": video_item.get("duration_sec", 0),
            "duration_bucket": video_item.get("duration_bucket", ""),
            "num_shots": video_item.get("num_shots", 0),
            "shot_time": video_item.get("shot_time", []),
            "status": result.get("status", "unknown"),
            "error": result.get("error"),
            "inference_time_sec": round(elapsed, 1),
            "model_prediction": {},
        }

        if result.get("status") == "success":
            output["model_prediction"] = {
                "references": result.get("references"),
                "events": result.get("events"),
                "shots": result.get("shots"),
            }
            # Capture top-level fields from single-pass
            for key in ["scene_description", "global_style", "global_audio"]:
                if key in result:
                    output["model_prediction"][key] = result[key]
            if result.get("shot_groups"):
                output["model_prediction"]["shot_groups"] = result["shot_groups"]

        # Store raw responses for debugging
        output["raw_responses"] = result.get("raw_responses", {})

        return output

    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        logger.error(f"[Worker] Exception for {video_path}: {e}\n{tb}")
        return {
            "video_path": video_path,
            "video_url": video_item.get("video_url", ""),
            "duration_sec": video_item.get("duration_sec", 0),
            "status": "exception",
            "error": str(e),
            "traceback": tb,
            "inference_time_sec": round(time.time() - t0, 1),
        }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="MTSS Eval - Model Inference Pipeline")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--input", type=str, default=config.INPUT_JSONL,
                        help="Input JSONL file path")
    parser.add_argument("--video-root", type=str, default=config.VIDEO_ROOT,
                        help="Media directory that relative video_path entries are resolved against")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index (inclusive) for data slicing")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive) for data slicing")
    parser.add_argument("--indices", type=str, default=None,
                        help="Path to JSON file with list of indices (overrides --start/--end)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (skip completed items)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (ignore existing results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be processed, don't run inference")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (e.g. 0 for greedy decoding)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top_p (e.g. 1.0 for greedy decoding)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: config.OUTPUT_DIR)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override OpenAI-compatible base_url for local vLLM servers")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    # Override output directory if specified
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    # Setup
    model_name = args.model
    model_cfg = dict(MODEL_PRESETS[model_name])  # shallow copy to allow overrides

    # Override sampling parameters if specified
    if args.temperature is not None:
        model_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        model_cfg["top_p"] = args.top_p
    if args.base_url is not None:
        model_cfg["base_url"] = args.base_url

    logger = setup_logging(model_name)

    logger.info("=" * 60)
    logger.info(f"MTSS Eval - Starting evaluation for model: {model_name}")
    logger.info(f"Model family: {model_cfg.get('model_family', 'unknown')}")
    logger.info(f"Model type: {model_cfg.get('model_type', 'unknown')}")
    logger.info(f"Temperature: {model_cfg.get('temperature', 'default')}")
    logger.info(f"Top-p: {model_cfg.get('top_p', 'default')}")
    logger.info(f"Base URL: {model_cfg.get('base_url', 'default')}")
    logger.info(f"Output dir: {config.OUTPUT_DIR}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)

    # Brief console header
    temp_str = f" | temp={model_cfg.get('temperature', '?')}" if args.temperature is not None else ""
    print(f"\n  Model: {model_name} | Workers: {args.workers}{temp_str}", file=sys.stderr)

    # Load data
    all_data = load_input_data(args.input, args.start, args.end,
                               indices_file=args.indices,
                               video_root=args.video_root)
    if args.indices:
        logger.info(f"Loaded {len(all_data)} video items from indices file: {args.indices}")
    else:
        logger.info(f"Loaded {len(all_data)} video items")

    # Get output paths
    paths = get_output_paths(model_name)

    # Filter by model family constraints
    model_family = model_cfg.get("model_family", "closed_source")
    strategy = config.SAMPLING_STRATEGY.get(model_family, {})

    if strategy.get("skip_long_video"):
        max_dur = strategy.get("max_duration", 60)
        before = len(all_data)
        all_data = [d for d in all_data if d.get("duration_sec", 0) <= max_dur]
        logger.info(f"Filtered to {len(all_data)} videos <= {max_dur}s "
                     f"(removed {before - len(all_data)} long videos) for {model_family}")

    # Resume: skip completed
    if args.resume:
        completed_ids = load_completed_ids(paths["results"])
        before = len(all_data)
        all_data = [d for d in all_data if d.get("video_path", "") not in completed_ids]
        logger.info(f"Resume: {len(completed_ids)} already completed, "
                     f"{len(all_data)} remaining (was {before})")

    if not all_data:
        print("  All done! Nothing to process.", file=sys.stderr)
        return

    # Dry run
    if args.dry_run:
        print(f"  [DRY RUN] Would process {len(all_data)} items", file=sys.stderr)
        for i, d in enumerate(all_data[:10]):
            print(f"    [{i}] {d.get('video_path', '')} "
                  f"(dur={d.get('duration_sec', 0):.1f}s, "
                  f"shots={d.get('num_shots', 0)})", file=sys.stderr)
        if len(all_data) > 10:
            print(f"    ... and {len(all_data) - 10} more", file=sys.stderr)
        return

    # Run inference
    total = len(all_data)
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()

    logger.info(f"Starting inference for {total} videos with {args.workers} workers...")
    print(f"  Processing {total} videos...", file=sys.stderr)

    def _fmt_eta(elapsed, done, total):
        """Format ETA string."""
        if done == 0:
            return "--:--"
        eta = elapsed / done * (total - done)
        m, s = divmod(int(eta), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    def _safe_stderr(text):
        """Write to stderr, ignoring BrokenPipeError (SSH disconnect)."""
        try:
            sys.stderr.write(text)
            sys.stderr.flush()
        except BrokenPipeError:
            pass

    def _print_progress(done, total, elapsed, failed):
        """Print a single-line progress bar to stderr."""
        pct = done / max(total, 1) * 100
        bar_len = 40
        filled = int(bar_len * done / max(total, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        eta = _fmt_eta(elapsed, done, total)
        avg = f"{elapsed/max(done,1):.0f}s/it" if done > 0 else "--"
        fail_str = f" | ❌{failed}" if failed else ""
        line = f"\r  {bar} {pct:5.1f}% [{done}/{total}] ETA {eta} ({avg}){fail_str}  "
        _safe_stderr(line)

    _print_progress(0, total, 0, 0)

    def _handle_result(result, video_path):
        """Process a single result: save, update counters, print progress."""
        nonlocal completed_count, skipped_count, failed_count
        status = result.get("status", "unknown")
        t_sec = result.get('inference_time_sec', 0)
        vname = os.path.basename(video_path)
        dur = result.get('duration_sec', 0)
        n_shots = result.get('num_shots', 0)

        if status == "success":
            completed_count += 1
            save_result_atomic(result, paths["results"])
            # Detailed info: refs/events/shots counts from model_prediction
            pred = result.get('model_prediction', {})
            n_refs = len(pred.get('references', []) or [])
            n_evts = len(pred.get('events', []) or [])
            n_pred_shots = len(pred.get('shots', []) or [])
            detail = f"refs={n_refs} events={n_evts} shots={n_pred_shots}"
            logger.info(
                f"✅ [{completed_count + skipped_count + failed_count}/{total}] "
                f"{vname} - SUCCESS ({t_sec:.1f}s) dur={dur:.0f}s shots={n_shots} {detail}"
            )
            _safe_stderr(
                f"\n  ✅ {vname} ({t_sec:.0f}s) dur={dur:.0f}s "
                f"{detail}\n"
            )
        elif status == "skipped":
            skipped_count += 1
            save_result_atomic(result, paths["results"])
            logger.info(
                f"⏭️ [{completed_count + skipped_count + failed_count}/{total}] "
                f"{vname} - SKIPPED: {result.get('error', '')}"
            )
            _safe_stderr(f"\n  ⏭️ {vname}: SKIPPED ({result.get('error', '')})\n")
        else:
            failed_count += 1
            save_result_atomic(result, paths["results"])
            save_error_atomic(result, paths["errors"])
            err_msg = result.get('error', 'unknown')
            logger.warning(
                f"❌ [{completed_count + skipped_count + failed_count}/{total}] "
                f"{vname} - FAILED ({t_sec:.1f}s): {err_msg}"
            )
            _safe_stderr(f"\n  ❌ {vname} ({t_sec:.0f}s): {err_msg}\n")

        done = completed_count + skipped_count + failed_count
        _print_progress(done, total, time.time() - start_time, failed_count)
        if done % 10 == 0 or done == total:
            save_progress(paths["progress"], total, completed_count,
                          skipped_count, failed_count, start_time)

    # Use simple loop for single-worker (local GPU) to avoid thread overhead
    if args.workers <= 1:
        for item in all_data:
            video_path = item.get("video_path", "unknown")
            dur = item.get("duration_sec", 0)
            n_shots = item.get("num_shots", 0)
            vname = os.path.basename(video_path)
            done = completed_count + skipped_count + failed_count
            _safe_stderr(
                f"\n  [{done+1}/{total}] {vname} "
                f"(dur={dur:.0f}s, shots={n_shots})\n"
            )
            try:
                result = process_one_video(item, model_cfg, paths["cache_dir"], logger)
                _handle_result(result, video_path)
            except Exception as e:
                failed_count += 1
                tb = traceback.format_exc()
                logger.error(f"❌ Exception for {video_path}: {e}\n{tb}")
                save_error_atomic({
                    "video_path": video_path,
                    "status": "exception",
                    "error": str(e),
                    "traceback": tb,
                }, paths["errors"])
                _safe_stderr(f"\n  ❌ {os.path.basename(video_path)}: {e}\n")
                done = completed_count + skipped_count + failed_count
                _print_progress(done, total, time.time() - start_time, failed_count)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {}
            for item in all_data:
                future = executor.submit(
                    process_one_video, item, model_cfg, paths["cache_dir"], logger
                )
                future_to_item[future] = item

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                video_path = item.get("video_path", "unknown")

                try:
                    result = future.result()
                    _handle_result(result, video_path)
                except Exception as e:
                    failed_count += 1
                    tb = traceback.format_exc()
                    logger.error(f"❌ Future exception for {video_path}: {e}\n{tb}")
                    save_error_atomic({
                        "video_path": video_path,
                        "status": "exception",
                        "error": str(e),
                        "traceback": tb,
                    }, paths["errors"])
                    _safe_stderr(f"\n  ❌ {os.path.basename(video_path)}: {e}\n")
                    done = completed_count + skipped_count + failed_count
                    _print_progress(done, total, time.time() - start_time, failed_count)

    # Final summary
    elapsed = time.time() - start_time
    save_progress(paths["progress"], total, completed_count,
                  skipped_count, failed_count, start_time)

    # Print final line to console
    sys.stderr.write("\n\n")
    m, s = divmod(int(elapsed), 60)
    print(f"  Done! ✅{completed_count} ⏭️{skipped_count} ❌{failed_count} | {m}m{s:02d}s", file=sys.stderr)
    print(f"  Results: {paths['results']}", file=sys.stderr)
    if failed_count:
        print(f"  Errors:  {paths['errors']}", file=sys.stderr)
    print(f"  Logs:    {config.LOG_DIR}/{model_name}/\n", file=sys.stderr)

    logger.info("=" * 60)
    logger.info(f"MTSS Eval - Completed for model: {model_name}")
    logger.info(f"Total: {total}")
    logger.info(f"Success: {completed_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Avg per item: {elapsed/max(completed_count, 1):.1f}s")
    logger.info(f"Results: {paths['results']}")
    logger.info(f"Errors: {paths['errors']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
