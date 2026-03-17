#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("[LOCAL-PIPELINE] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def _default_output_dir() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/local_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local smoke test + experiments + checks without SLURM"
    )
    parser.add_argument("--policy", default="examples/healthcare.abac", help="Path to .abac policy file")
    parser.add_argument("--readme", default="examples/healthcare_README.md", help="Path to policy README")
    parser.add_argument("--output-dir", default=_default_output_dir(), help="Output directory")
    parser.add_argument(
        "--llm-backend",
        choices=["heuristic", "hf-local"],
        default="heuristic",
        help="LLM backend for experiments",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="HF model id for hf-local backend")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization for hf-local")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-phase1", type=int, default=400)
    parser.add_argument("--max-phase2", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=900)

    parser.add_argument(
        "--run-smoke",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run qwen smoke test before experiments (default: true for hf-local, false for heuristic)",
    )
    parser.add_argument("--smoke-output", default=None, help="Optional smoke output path")
    parser.add_argument("--smoke-max-tokens", type=int, default=96)
    parser.add_argument(
        "--check-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run scripts/check_results.py after experiments",
    )
    parser.add_argument("--min-lines", type=int, default=1, help="Minimum lines for check_results validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_smoke = args.run_smoke
    if run_smoke is None:
        run_smoke = args.llm_backend == "hf-local"

    if run_smoke:
        smoke_output = args.smoke_output or str(output_dir / "qwen_smoke_local.jsonl")
        smoke_cmd = [
            sys.executable,
            "scripts/qwen_smoke_test.py",
            "--model-id",
            args.model_id,
            "--output",
            smoke_output,
            "--max-tokens",
            str(args.smoke_max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
        ]
        if args.no_4bit:
            smoke_cmd.append("--no-4bit")
        if args.trust_remote_code:
            smoke_cmd.append("--trust-remote-code")
        _run(smoke_cmd)

    exp_cmd = [
        sys.executable,
        "scripts/run_experiments.py",
        "--policy",
        args.policy,
        "--readme",
        args.readme,
        "--output-dir",
        str(output_dir),
        "--llm-backend",
        args.llm_backend,
        "--seed",
        str(args.seed),
        "--max-phase1",
        str(args.max_phase1),
        "--max-phase2",
        str(args.max_phase2),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
    ]

    if args.llm_backend == "hf-local":
        exp_cmd.extend(["--model-id", args.model_id])
        if args.no_4bit:
            exp_cmd.append("--no-4bit")
        if args.trust_remote_code:
            exp_cmd.append("--trust-remote-code")

    _run(exp_cmd)

    if args.check_results:
        _run(
            [
                sys.executable,
                "scripts/check_results.py",
                "--output-dir",
                str(output_dir),
                "--min-lines",
                str(args.min_lines),
            ]
        )

    print(f"[LOCAL-PIPELINE] Completed successfully. Outputs: {output_dir}")


if __name__ == "__main__":
    main()