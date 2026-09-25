from pathlib import Path

from omnicapbench_eval.mtss_metrics import evaluate_batch


def test_eval_metrics_sample_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    gt_path = repo_root / "examples" / "eval_sample.jsonl"
    pred_path = repo_root / "examples" / "prediction_sample.jsonl"
    output_path = tmp_path / "metrics_summary.json"

    summary = evaluate_batch(str(gt_path), str(pred_path), str(output_path), workers=1)

    assert summary["num_evaluated"] == 3
    assert summary["num_errors"] == 0
    assert output_path.exists()
    assert (tmp_path / "metrics_summary_detail.jsonl").exists()
