#!/usr/bin/env python3
"""Render the two OmniCapBench result tables from computed metric files.

Table 1 (rule-based structural evaluation) is read from ``metrics_summary.json``
produced by ``omnicapbench_eval.mtss_metrics``.

Table 2 (localized semantic evaluation) is read from ``llm_judge_scores.json``
produced by ``omnicapbench_eval.mtss_llm_judge``, except for the Dialogue column
which is the rule-based dialogue line score and therefore comes from the
rule-based summary.

Usage:
    python -m omnicapbench_eval.report_tables \
        Gemini-2.5-Pro=output/eval/gemini_25 \
        Qwen3-Omni=output/eval/qwen3_omni \
        --output output/report
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("report_tables")

RULE_SUMMARY_NAME = "metrics_summary.json"
JUDGE_SUMMARY_NAME = "llm_judge_scores.json"

# (paper column header, code key in the rule-based ``aggregated`` block, domain)
RULE_TABLE_COLUMNS = [
    ("SGC", "SGC", "Gate"),
    ("RefUse", "RefUse", "Visual"),
    ("CCC", "CCC", "Visual"),
    ("Ref Subject F1", "Ref_Subject_F1", "Visual"),
    ("Ref Scene F1", "Ref_Scene_F1", "Visual"),
    ("Shot F1", "shot_f1", "Visual"),
    ("Shot tIoU", "Shot_tIoU", "Visual"),
    ("Subshot F1", "Subshot_F1", "Visual"),
    ("Subshot tIoU", "Subshot_tIoU", "Visual"),
    ("Event F1", "event_f1", "Audio"),
    ("Event tIoU", "Event_tIoU", "Audio"),
    ("Speaker F1", "Speaker_F1", "Audio-Visual"),
    ("EVSA F1", "EVSA", "Audio-Visual"),
]

# (paper column header, code key, source) where source is "judge" or "rule"
SEMANTIC_TABLE_COLUMNS = [
    ("Shot", "LLM_shot_fact", "judge"),
    ("Subject Recall", "LLM_person_ref_recall", "judge"),
    ("Subject Precision", "LLM_person_ref_precision", "judge"),
    ("Scene Recall", "LLM_scene_ref_recall", "judge"),
    ("Scene Precision", "LLM_scene_ref_precision", "judge"),
    ("Subshot Recall", "LLM_subshot_recall", "judge"),
    ("Subshot Precision", "LLM_subshot_precision", "judge"),
    ("Dialogue", "DialogueLineScore", "rule"),
    ("Non-dialogue", "LLM_event", "judge"),
]


def parse_run_spec(spec: str) -> Tuple[str, str]:
    """Split a ``NAME=DIR`` run specification, defaulting NAME to the dir name."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    path = spec.rstrip("/")
    return os.path.basename(path) or path, path


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None when it is missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def collect_run(name: str, run_dir: str) -> Dict[str, Any]:
    """Read the rule-based and judge summaries for one evaluated model."""
    rule = load_json(os.path.join(run_dir, RULE_SUMMARY_NAME)) or {}
    judge = load_json(os.path.join(run_dir, JUDGE_SUMMARY_NAME)) or {}

    if not rule:
        logger.warning("%s: no %s in %s", name, RULE_SUMMARY_NAME, run_dir)
    if not judge:
        logger.info("%s: no %s in %s, semantic table will be empty",
                    name, JUDGE_SUMMARY_NAME, run_dir)

    return {
        "name": name,
        "run_dir": run_dir,
        "num_evaluated": rule.get("num_evaluated"),
        "rule": rule.get("aggregated", {}),
        "judge": judge.get("llm_judge_table", {}),
    }


def as_percent(value: Any) -> Optional[float]:
    """Convert a 0-1 metric into a percentage, or None when unavailable.

    Rule-based aggregates are stored as ``{"count", "mean", "std", ...}`` blocks,
    whereas the judge table stores plain floats, so both shapes are accepted.
    """
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value) * 100.0, 2)
    except (TypeError, ValueError):
        return None


def build_table(runs: List[Dict[str, Any]], columns: List[Tuple[str, str, str]],
                rule_only: bool) -> Dict[str, Any]:
    """Build a table as {"columns": [...], "rows": {name: {header: percent}}}."""
    rows = {}
    for run in runs:
        row = {}
        for header, key, tag in columns:
            source = run["rule"] if rule_only or tag == "rule" else run["judge"]
            row[header] = as_percent(source.get(key))
        rows[run["name"]] = row
    return {
        "columns": [header for header, _, _ in columns],
        "rows": rows,
    }


def render_markdown(title: str, table: Dict[str, Any]) -> str:
    """Render one table as a GitHub-flavoured markdown table."""
    headers = ["Model"] + table["columns"]
    lines = [
        "### %s" % title,
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for name, row in table["rows"].items():
        cells = [name]
        for header in table["columns"]:
            value = row.get(header)
            cells.append("-" if value is None else "%.2f" % value)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render OmniCapBench result tables from computed metric files")
    parser.add_argument("runs", nargs="+", metavar="NAME=DIR",
                        help="Evaluation output directories, optionally prefixed with a display name")
    parser.add_argument("--output", default=None,
                        help="Directory for report_tables.md and report_tables.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    runs = [collect_run(*parse_run_spec(spec)) for spec in args.runs]

    rule_table = build_table(runs, RULE_TABLE_COLUMNS, rule_only=True)
    semantic_table = build_table(runs, SEMANTIC_TABLE_COLUMNS, rule_only=False)

    markdown = "\n".join([
        "# OmniCapBench Results",
        "",
        "All scores are macro-averaged across videos and reported as percentages.",
        "",
        render_markdown("Rule-based structural evaluation", rule_table),
        render_markdown("Localized semantic evaluation", semantic_table),
    ])
    print(markdown)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        md_path = os.path.join(args.output, "report_tables.md")
        json_path = os.path.join(args.output, "report_tables.json")
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump({
                "runs": [
                    {"name": r["name"], "run_dir": r["run_dir"],
                     "num_evaluated": r["num_evaluated"]}
                    for r in runs
                ],
                "rule_based_structural": rule_table,
                "localized_semantic": semantic_table,
            }, handle, indent=2, ensure_ascii=False)
        logger.info("Wrote %s and %s", md_path, json_path)


if __name__ == "__main__":
    main()
