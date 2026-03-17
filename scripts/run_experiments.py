#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policykg.experiments import ExperimentConfig, run_experiments
from policykg.llm import HFLocalCausalLMClient, HeuristicLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Policy-KG guardrails experiments")
    parser.add_argument("--policy", required=True, help="Path to .abac policy file")
    parser.add_argument("--readme", default=None, help="Path to policy README")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-phase1", type=int, default=400)
    parser.add_argument("--max-phase2", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--llm-backend",
        choices=["heuristic", "hf-local"],
        default="heuristic",
        help="LLM backend for generation",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-4B",
        help="Hugging Face model ID for --llm-backend=hf-local",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization for hf-local backend",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for hf-local backend",
    )
    args = parser.parse_args()

    if args.llm_backend == "heuristic":
        llm_client = HeuristicLLMClient()
    else:
        llm_client = HFLocalCausalLMClient(
            model_id=args.model_id,
            load_in_4bit=not args.no_4bit,
            trust_remote_code=args.trust_remote_code,
        )

    result = run_experiments(
        policy_path=args.policy,
        readme_path=args.readme,
        output_dir=args.output_dir,
        llm_client=llm_client,
        config=ExperimentConfig(
            seed=args.seed,
            max_phase1_samples=args.max_phase1,
            max_phase2_samples=args.max_phase2,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
    )
    pprint(result)


if __name__ == "__main__":
    main()
