from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Protocol


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
    ) -> str: ...


@dataclass
class ScriptedLLMClient:
    """Deterministic client used in tests and local dry runs."""

    responses: list[str]
    fallback_decision: str = "Deny"
    _cursor: int = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
    ) -> str:
        if self._cursor < len(self.responses):
            text = self.responses[self._cursor]
            self._cursor += 1
            return text
        decision = self.fallback_decision
        return (
            "Decision\n"
            f"decision = {decision}\n"
            "confidence = Medium\n\n"
            "PolicyContext\n"
            "policy_id = scripted\n"
            "request_subject = {}\n"
            "request_resource = {}\n"
            "request_action = read\n"
            "request_environment = {}\n\n"
            "Evidence\n"
            "evidence_subgraphs = []\n"
            "decisive_rules = []\n"
            "decisive_predicates = []\n\n"
            "Explanation\n"
            "1 Placeholder supports =\n\n"
            "Citations\n"
            "citations = []\n"
        )


@dataclass
class HeuristicLLMClient:
    """Simple provider-agnostic fallback for offline experiments."""

    prefer_permit_tokens: set[str] = field(default_factory=lambda: {"author", "same", "treating"})

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
    ) -> str:
        prompt = user_prompt.lower()
        decision = "Permit" if any(tok in prompt for tok in self.prefer_permit_tokens) else "Deny"
        return (
            "Decision\n"
            f"decision = {decision}\n"
            "confidence = Medium\n\n"
            "PolicyContext\n"
            "policy_id = heuristic\n"
            "request_subject = {}\n"
            "request_resource = {}\n"
            "request_action = read\n"
            "request_environment = {}\n\n"
            "Evidence\n"
            "evidence_subgraphs = [SG1]\n"
            "decisive_rules = [R1]\n"
            "decisive_predicates = [{ rule = R1, predicate_ids = [P1] }]\n\n"
            "Explanation\n"
            "1 Heuristic prediction supports = SG1 R1 P1\n\n"
            "Citations\n"
            "citations = [SG1]\n"
        )


@dataclass
class HFLocalCausalLMClient:
    """
    Local Hugging Face causal LM backend (Qwen-compatible).

    Recommended for 16GB VRAM:
    - `model_id=\"Qwen/Qwen3.5-9B\"`
    - `load_in_4bit=True`
    """

    model_id: str
    load_in_4bit: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = False
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _processor: Any = field(init=False, repr=False, default=None)
    _backend: str = field(init=False, repr=False, default="causal")
    _torch: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "HFLocalCausalLMClient requires `transformers`, `torch`, and `accelerate`."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - depends on optional deps
                raise RuntimeError(
                    "4-bit loading requested but bitsandbytes support is unavailable. "
                    "Install `bitsandbytes` or run with load_in_4bit=False."
                ) from exc

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self._backend = "causal"
        except Exception:
            # Some Qwen 3.5 checkpoints are exposed as image-text models.
            # Fallback keeps text-only prompting operational for smoke tests.
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError(
                    "Could not load model via AutoModelForCausalLM and fallback "
                    "AutoModelForImageTextToText is unavailable."
                ) from exc

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            if hasattr(self._processor, "tokenizer"):
                self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            self._backend = "image_text"

        self._model.eval()

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> Any:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self._backend == "image_text" and self._processor is not None:
            if hasattr(self._processor, "apply_chat_template"):
                chat_text = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            elif hasattr(self._tokenizer, "apply_chat_template"):
                chat_text = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                chat_text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            return self._processor(text=[chat_text], return_tensors="pt")

        if hasattr(self._tokenizer, "apply_chat_template"):
            return {"input_ids": self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )}

        # Fallback for tokenizers without chat template support.
        text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        return self._tokenizer(text, return_tensors="pt")

    def _model_device(self) -> Any:
        try:
            return next(self._model.parameters()).device
        except Exception:
            if self._torch.cuda.is_available():
                return self._torch.device("cuda")
            return self._torch.device("cpu")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
    ) -> str:
        model_inputs = self._build_prompt(system_prompt, user_prompt)
        model_device = self._model_device()

        for key, value in list(model_inputs.items()):
            if hasattr(value, "to"):
                model_inputs[key] = value.to(model_device)

        do_sample = temperature > 0.0
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        pad_token_id = eos_token_id

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        input_len = model_inputs["input_ids"].shape[-1]
        generated = output_ids[0, input_len:]

        if self._backend == "image_text" and self._processor is not None and hasattr(self._processor, "batch_decode"):
            return self._processor.batch_decode(generated.unsqueeze(0), skip_special_tokens=True)[0].strip()
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
