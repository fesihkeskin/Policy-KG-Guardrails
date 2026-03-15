from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PolicyError(Exception):
    """Base class for policy processing errors."""


class PolicyParseError(PolicyError):
    """Raised when parsing an ABAC artifact fails."""


class PolicyValidationError(PolicyError):
    """Raised when parsed policy violates canonical integrity constraints."""


class TriValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    PERMIT = "Permit"
    DENY = "Deny"
    INSUFFICIENT = "InsufficientPolicyEvidence"


@dataclass(frozen=True)
class PolicyMeta:
    policy_id: str
    domain: str = "unknown"
    version: str = "unknown"
    dataset: str = "unknown"
    combining: str = "deny-overrides"


@dataclass(frozen=True)
class AttrDef:
    scope: str
    name: str
    datatype: str
    multivalue: bool = False


@dataclass(frozen=True)
class ActionDef:
    name: str


@dataclass(frozen=True)
class Predicate:
    predicate_id: str
    op: str
    left_scope: str
    left_attr: str
    right_value: Any = None
    right_scope: Optional[str] = None
    right_attr: Optional[str] = None


@dataclass(frozen=True)
class ExprNode:
    node_id: str
    op: str
    children: tuple[str, ...] = ()
    predicate_id: Optional[str] = None


@dataclass(frozen=True)
class Rule:
    rule_id: str
    effect: str
    actions: tuple[str, ...]
    root_expr_id: str
    expr_nodes: dict[str, ExprNode]
    predicates: dict[str, Predicate]
    priority: Optional[int] = None
    source: Optional[str] = None


@dataclass
class CanonicalPolicyIR:
    meta: PolicyMeta
    subject_attrs: dict[str, AttrDef]
    resource_attrs: dict[str, AttrDef]
    environment_attrs: dict[str, AttrDef]
    actions: dict[str, ActionDef]
    rules: list[Rule]
    users: dict[str, dict[str, Any]] = field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RequestContext:
    subject: dict[str, Any]
    resource: dict[str, Any]
    action: Optional[str]
    environment: dict[str, Any] = field(default_factory=dict)
    unknown_token: str = "UNKNOWN"


@dataclass
class DecisionTrace:
    decision: Decision
    decisive_rules: list[str]
    decisive_predicates: dict[str, dict[str, TriValue]]
    rule_results: dict[str, TriValue]
    missing_attributes: list[str] = field(default_factory=list)
    combining: str = "deny-overrides"
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextEvidence:
    evidence_id: str
    text: str
    rule_ids: tuple[str, ...] = ()
    score: float = 0.0


@dataclass(frozen=True)
class EvidenceSubgraph:
    evidence_id: str
    rule_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    score: float = 0.0


@dataclass
class ExplanationClaim:
    index: int
    text: str
    supports: list[str]
    aligned: bool = False
    alignment_reason: str = ""


@dataclass
class ModelDraft:
    decision: Decision
    confidence: str
    policy_id: str
    request_subject: dict[str, Any]
    request_resource: dict[str, Any]
    request_action: Optional[str]
    request_environment: dict[str, Any]
    evidence_subgraphs: list[str]
    decisive_rules: list[str]
    decisive_predicates: dict[str, list[str]]
    claims: list[ExplanationClaim]
    citations: list[str]
    raw_text: str = ""


@dataclass
class GuardrailedResponse:
    decision: Decision
    confidence: str
    policy_id: str
    request_subject: dict[str, Any]
    request_resource: dict[str, Any]
    request_action: Optional[str]
    request_environment: dict[str, Any]
    evidence_subgraphs: list[str]
    decisive_rules: list[str]
    decisive_predicates: dict[str, list[str]]
    claims: list[ExplanationClaim]
    citations: list[str]
    missing_attributes: list[str] = field(default_factory=list)
    was_overridden: bool = False
    revised: bool = False
    unsupported_claim_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class MetricsBundle:
    accuracy: float
    false_permit_rate: float
    false_deny_rate: float
    abstain_rate: float
    graph_faithfulness: float
    rule_precision: float
    attr_precision: float
    kg_hallucination_rate: float
    citation_correctness: float
    citation_faithfulness: float
    unsupported_claim_rate: float
    counterfactual_validity: float = 0.0
    counterfactual_minimality: float = 0.0
    avg_attr_changes: float = 0.0


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    task_type: str
    query: str
    request: RequestContext
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantPrediction:
    task_id: str
    variant: str
    response: GuardrailedResponse
    oracle_decision: Decision
    metadata: dict[str, Any] = field(default_factory=dict)
