#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    "tables.md",
    "predictions_vanilla.jsonl",
    "predictions_text-rag.jsonl",
    "predictions_kg-rag.jsonl",
    "predictions_policy-kg-guardrails.jsonl",
]

REQUIRED_TABLE_HEADERS = [
    "## Decision reliability",
    "## Grounding and faithfulness",
    "## Citation quality",
    "## Adversarial robustness",
    "## Counterfactual quality",
]


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-lines", type=int, default=1, help="Minimum non-empty lines per predictions file")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        raise SystemExit(f"Output directory does not exist: {out_dir}")

    missing = []
    for name in REQUIRED_FILES:
        p = out_dir / name
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    table_path = out_dir / "tables.md"
    text = table_path.read_text(encoding="utf-8")
    missing_headers = [h for h in REQUIRED_TABLE_HEADERS if h not in text]
    if missing_headers:
        raise SystemExit("tables.md missing sections: " + ", ".join(missing_headers))

    for name in REQUIRED_FILES:
        if not name.endswith(".jsonl"):
            continue
        p = out_dir / name
        count = _line_count(p)
        if count < args.min_lines:
            raise SystemExit(f"{p} has too few lines: {count}")

    print(f"Result check passed for {out_dir}")


if __name__ == "__main__":
    main()
