from __future__ import annotations

from dataclasses import dataclass, field

from .evaluator import evaluate
from .guardrails import verify_and_revise
from .kg import PolicyKG
from .llm import LLMClient
from .response_contract import format_contract_request, parse_response_contract
from .retrieval import retrieve_graph, retrieve_text
from .types import Decision, GuardrailedResponse, ModelDraft, RequestContext


@dataclass
class DecodingConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 900


@dataclass
class VariantRunner:
    policy_kg: PolicyKG
    llm_client: LLMClient
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    def _compose_prompt(
        self,
        *,
        variant_name: str,
        query: str,
        request: RequestContext,
        text_evidence: str,
        graph_evidence: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are an ABAC policy assistant. Do not invent rules or attributes. "
            "Answer strictly with the required contract."
        )

        contract = format_contract_request(
            policy_id=self.policy_kg.policy.meta.policy_id,
            subject=request.subject,
            resource=request.resource,
            action=request.action,
            environment=request.environment,
            evidence_ids=["SG1", "SG2", "SG3", "SG4", "SG5"],
            rule_ids=[rule.rule_id for rule in self.policy_kg.policy.rules],
        )

        user_prompt = (
            f"Variant: {variant_name}\n"
            f"Question: {query}\n"
            f"RequestContext: subject={request.subject} resource={request.resource} "
            f"action={request.action} environment={request.environment}\n"
            f"TextEvidence:\n{text_evidence}\n\n"
            f"GraphEvidence:\n{graph_evidence}\n\n"
            f"{contract}"
        )
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        return self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.decoding.temperature,
            top_p=self.decoding.top_p,
            max_tokens=self.decoding.max_tokens,
        )

    def run_variant(
        self,
        variant_name: str,
        query: str,
        request_context: RequestContext,
        evidence: list[str] | None = None,
    ) -> ModelDraft:
        v = variant_name.lower()

        text_evidence = ""
        graph_evidence = ""

        if v == "vanilla":
            pass
        elif v == "text-rag":
            text_items = retrieve_text(query, k=8, policy_kg=self.policy_kg)
            text_evidence = "\n".join(f"- {item.evidence_id}: {item.text}" for item in text_items)
        elif v in {"kg-rag", "policy-kg-guardrails", "guardrails"}:
            graph_items = retrieve_graph(self.policy_kg, query, k_rules=5)
            graph_evidence = "\n".join(
                f"- {item.evidence_id} rules={list(item.rule_ids)} edges={len(item.edges)}"
                for item in graph_items
            )
        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        system_prompt, user_prompt = self._compose_prompt(
            variant_name=variant_name,
            query=query,
            request=request_context,
            text_evidence=text_evidence,
            graph_evidence=graph_evidence,
        )

        raw = self._call_llm(system_prompt, user_prompt)
        return parse_response_contract(
            raw,
            default_policy_id=self.policy_kg.policy.meta.policy_id,
            default_subject=request_context.subject,
            default_resource=request_context.resource,
            default_action=request_context.action,
            default_environment=request_context.environment,
        )

    def run_guardrails(self, query: str, request_context: RequestContext) -> GuardrailedResponse:
        graph_items = retrieve_graph(self.policy_kg, query, k_rules=5)
        draft = self.run_variant("policy-kg-guardrails", query, request_context)
        trace = evaluate(self.policy_kg, request_context)

        # Expand retrieval once before abstaining, as specified.
        if trace.decision == Decision.INSUFFICIENT and trace.missing_attributes:
            graph_items = retrieve_graph(self.policy_kg, query, k_rules=10)

        return verify_and_revise(
            draft,
            trace,
            graph_items,
            llm_client=self.llm_client,
            allow_revise=True,
        )


def run_variant(
    variant_name: str,
    query: str,
    request_context: RequestContext,
    evidence: list[str] | None,
    *,
    runner: VariantRunner,
) -> ModelDraft:
    _ = evidence
    return runner.run_variant(variant_name, query, request_context)
