#!/usr/bin/env python3
"""Offline installation check for the rule-based metrics.

Scores the bundled sample predictions against the bundled annotations. No API
key, GPU, or video file is needed.

The sample predictions are copies of the ground truth, so this is an oracle
sanity check: every metric should land at or near 1.0. Reference matching falls
back to exact identifier matching here, because no judge endpoint is configured.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omnicapbench_eval.mtss_metrics import evaluate_batch, print_summary


def main() -> None:
    gt_path = REPO_ROOT / "examples" / "eval_sample.jsonl"
    pred_path = REPO_ROOT / "examples" / "prediction_sample.jsonl"
    output_path = REPO_ROOT / "output" / "demo_metrics" / "metrics_summary.json"

    summary = evaluate_batch(str(gt_path), str(pred_path), str(output_path), workers=1)
    print_summary(summary)
    print("\nSaved summary to: %s" % output_path)
    print("Oracle sanity check: scores at or near 1.0 mean the install is healthy.")


if __name__ == "__main__":
    main()
