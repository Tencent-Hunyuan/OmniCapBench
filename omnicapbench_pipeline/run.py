#!/usr/bin/env python3
"""
OmniCapBench Pipeline - Entry Point
Entry point: supports command-line arguments for specifying input file, output directory, resume from checkpoint, etc.

Usage:
    python -m omnicapbench_pipeline.run --input data.json --output ./output
    python -m omnicapbench_pipeline.run --resume
    python -m omnicapbench_pipeline.run --single 0
    python -m omnicapbench_pipeline.run --model qwen_omni
    python -m omnicapbench_pipeline.run --model gemini_25 --stage2-model gemini_31
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from . import config
from .pipeline import run_pipeline, run_pipeline_for_item, _init_all_models


# ======================================================================
# Model preset mapping — CLI model name → config preset
# ======================================================================
_PRESET_MAP = {
    "qwen_omni": config.PRESET_QWEN_OMNI,
    "gemini_31": config.PRESET_GEMINI_31,
    "gemini_25": config.PRESET_GEMINI_25,
    "qwen35_omni_plus": config.PRESET_QWEN35_OMNI_PLUS,
}


def _parse_multi_model(model_str: str, min_count: int, max_count: int, stage_name: str) -> list:
    """
    Parse a model string that may contain comma-separated model names.
    If a single name is given, replicate it to min_count.
    If multiple names are given, validate the count is within [min_count, max_count].
    """
    parts = [p.strip() for p in model_str.split(",") if p.strip()]
    valid_names = list(_PRESET_MAP.keys())

    for p in parts:
        if p not in _PRESET_MAP:
            print(f"❌ Unknown model name '{p}' (available: {valid_names})")
            sys.exit(1)

    if len(parts) == 1:
        # Single model → replicate to min_count
        return [{**_PRESET_MAP[parts[0]]} for _ in range(min_count)]
    elif min_count <= len(parts) <= max_count:
        # Count within valid range → each model maps to one slot
        return [{**_PRESET_MAP[p]} for p in parts]
    else:
        print(
            f"❌ {stage_name} requires {min_count}-{max_count} models, "
            f"but {len(parts)} were provided: {parts}"
        )
        sys.exit(1)


def _apply_model_overrides(args):
    """
    Apply CLI model selection to config.
    --model sets all stages; --stage{N}-model overrides individual stages.
    Supports comma-separated model names for multi-model stages (stage2, stage3).
    """
    valid_names = list(_PRESET_MAP.keys())

    # Global override
    if args.model:
        if args.model not in _PRESET_MAP:
            print(f"❌ Unknown model name '{args.model}' (available: {valid_names})")
            sys.exit(1)
        preset = _PRESET_MAP[args.model]
        config.STAGE1_MODEL = {**preset}
        config.STAGE2_MODELS = [{**preset} for _ in range(3)]
        config.STAGE3_MODELS = [{**preset} for _ in range(2)]
        config.STAGE4_MODEL = {**preset}
        print(f"🔧 All stage models set to: {args.model}")

    # Per-stage overrides (higher priority)
    if args.stage1_model:
        if args.stage1_model not in _PRESET_MAP:
            print(f"❌ Unknown model name '{args.stage1_model}' (available: {valid_names})")
            sys.exit(1)
        config.STAGE1_MODEL = {**_PRESET_MAP[args.stage1_model]}
        print(f"🔧 Stage1 model set to: {args.stage1_model}")

    if args.stage2_model:
        config.STAGE2_MODELS = _parse_multi_model(args.stage2_model, 2, 3, "Stage2")
        print(f"🔧 Stage2 model set to: {args.stage2_model} ({len(config.STAGE2_MODELS)} models)")

    if args.stage3_model:
        config.STAGE3_MODELS = _parse_multi_model(args.stage3_model, 2, 2, "Stage3")
        print(f"🔧 Stage3 model set to: {args.stage3_model}")

    if args.stage4_model:
        if args.stage4_model not in _PRESET_MAP:
            print(f"❌ Unknown model name '{args.stage4_model}' (available: {valid_names})")
            sys.exit(1)
        config.STAGE4_MODEL = {**_PRESET_MAP[args.stage4_model]}
        print(f"🔧 Stage4 model set to: {args.stage4_model}")


def setup_logging(log_dir: str = None) -> None:
    """
    Configure logging system: output to both file and terminal.
    File log: detailed (DEBUG level)
    Terminal log: info level (with emoji progress)
    """
    log_dir = log_dir or config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    # Pipeline-specific logger
    logger = logging.getLogger("omnicapbench_pipeline")
    logger.setLevel(logging.DEBUG)

    # File handler: detailed logs
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Terminal handler: info level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # ---- Summary log: progress + errors only ----
    summary_log_file = os.path.join(log_dir, f"summary_{timestamp}.log")
    summary_logger = logging.getLogger("omnicapbench_pipeline.summary")
    summary_logger.setLevel(logging.DEBUG)
    summary_logger.propagate = False  # Don't propagate to parent (detail log)
    sfh = logging.FileHandler(summary_log_file, encoding="utf-8")
    sfh.setLevel(logging.DEBUG)
    sfh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    summary_logger.addHandler(sfh)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Summary log file: {summary_log_file}")
    print(f"📋 Log file: {log_file}")
    print(f"📋 Summary log: {summary_log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="MTSS Structured Caption Data Processing Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=config.DEFAULT_INPUT_JSON,
        help="Input JSON file path (list of video entries)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last progress (enabled by default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, ignoring previous progress",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=False,
        help="Also skip failed entries when resuming (by default only successful entries are skipped, failed ones are retried)",
    )
    parser.add_argument(
        "--single", "-s",
        type=int,
        default=None,
        help="Process only the entry at the specified index (for debugging)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel worker threads (defaults to config.NUM_WORKERS, which is 16)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only the first N entries in batch mode (defaults to all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stratified sampling of N entries by duration_bucket (e.g. --sample 50 samples ~17 from each of <1min/1-3min/3-5min)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log file directory",
    )
    # --- Model selection arguments ---
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Set model type for all stages (available: qwen_omni, gemini_31, gemini_25, qwen35_omni_plus)",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default=None,
        help="Stage1 Gen model type (single model name)",
    )
    parser.add_argument(
        "--stage2-model",
        type=str,
        default=None,
        help="Stage2 Eval model type (supports comma-separated 2-3 different models, e.g.: gemini_25,qwen35_omni_plus or gemini_25,gemini_31,qwen35_omni_plus)",
    )
    parser.add_argument(
        "--stage3-model",
        type=str,
        default=None,
        help="Stage3 Refine model type (supports comma-separated 2 different models, e.g.: qwen35_omni_plus,gemini_25)",
    )
    parser.add_argument(
        "--stage4-model",
        type=str,
        default=None,
        help="Stage4 Select model type (single model name)",
    )
    # --- Shot chunking arguments ---
    parser.add_argument(
        "--shot-chunk-size",
        type=int,
        default=None,
        help="Shot chunk group size: max shots per group (default 8, set to 0 to disable grouping)",
    )
    parser.add_argument(
        "--shot-chunk-threshold",
        type=int,
        default=None,
        help="Shot chunk threshold: grouping is enabled only when shot count exceeds this value (default 10)",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging(args.log_dir)
    logger = logging.getLogger("omnicapbench_pipeline")
    summary_logger = logging.getLogger("omnicapbench_pipeline.summary")

    # --- Apply model overrides from CLI ---
    _apply_model_overrides(args)

    # --- Apply shot chunking overrides from CLI ---
    if args.shot_chunk_size is not None:
        config.SHOT_CHUNK_SIZE = args.shot_chunk_size
        print(f"🔧 Shot chunk size set to: {config.SHOT_CHUNK_SIZE}")
    if args.shot_chunk_threshold is not None:
        config.SHOT_CHUNK_THRESHOLD = args.shot_chunk_threshold
        print(f"🔧 Shot chunk threshold set to: {config.SHOT_CHUNK_THRESHOLD}")

    # Log startup config to summary
    summary_logger.info(f"[Config] input={args.input}")
    summary_logger.info(f"[Config] stage1={args.stage1_model or args.model or 'default'}, "
                        f"stage2={args.stage2_model or args.model or 'default'}, "
                        f"stage3={args.stage3_model or args.model or 'default'}, "
                        f"stage4={args.stage4_model or args.model or 'default'}")
    summary_logger.info(f"[Config] shot_chunk_size={config.SHOT_CHUNK_SIZE}, "
                        f"shot_chunk_threshold={config.SHOT_CHUNK_THRESHOLD}, "
                        f"skip_failed={args.skip_failed}")

    resume = args.resume and not args.no_resume

    # Single-item mode (for debugging/testing)
    if args.single is not None:
        logger.info(f"Single-item mode: index={args.single}")
        print(f"\n🔬 Single-item mode: processing entry {args.single}")

        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        if args.single >= len(data):
            print(f"❌ Index {args.single} out of range (total: {len(data)})")
            sys.exit(1)

        item = data[args.single]
        models = _init_all_models()

        result = run_pipeline_for_item(
            item=item,
            item_index=args.single,
            total_items=len(data),
            models=models,
        )

        if result is not None:
            os.makedirs(args.output, exist_ok=True)
            out_file = os.path.join(
                args.output,
                f"single_{args.single}_{datetime.now().strftime('%H%M%S')}.json"
            )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({**item, "mtss_caption": result}, f,
                          ensure_ascii=False, indent=2)
            print(f"\n✅ Result saved: {out_file}")
        else:
            print(f"\n❌ Pipeline processing failed")
            sys.exit(1)
        return

    # Batch mode
    run_pipeline(
        input_json_path=args.input,
        output_dir=args.output,
        resume=resume,
        num_workers=args.workers,
        limit=args.limit,
        sample=args.sample,
        skip_failed=args.skip_failed,
    )


if __name__ == "__main__":
    main()
