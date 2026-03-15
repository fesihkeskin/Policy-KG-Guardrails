#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policykg.llm import HFLocalCausalLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen smoke test via local HF backend")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B", help="HF model id")
    parser.add_argument("--output", default="outputs/qwen_smoke.jsonl", help="Output jsonl path")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    client = HFLocalCausalLMClient(
        model_id=args.model_id,
        load_in_4bit=not args.no_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    prompts = [
        "Reply with exactly one word: READY",
        "What is 2+2? Answer with one number only.",
        (
            "You are a policy assistant. Return exactly:\n"
            "Decision\n"
            "decision = Permit\n"
            "confidence = High"
        ),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts, start=1):
            response = client.generate(
                system_prompt="You are a concise assistant.",
                user_prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            row = {"idx": idx, "prompt": prompt, "response": response}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"[QWEN-SMOKE] prompt#{idx} response: {response[:220]}")
            if not response.strip():
                failures += 1

    if failures > 0:
        raise SystemExit(f"Smoke test failed: {failures} empty responses")

    print(f"Smoke test passed. Outputs written to: {output_path}")


if __name__ == "__main__":
    main()
