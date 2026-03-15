from __future__ import annotations

import pytest

from policykg.parser import parse_abac_text
from policykg.types import PolicyValidationError


def test_duplicate_rule_ids_raise_validation_error() -> None:
    payload = {
        "meta": {"policy_id": "dup"},
        "users": {"u1": {"position": "doctor"}},
        "resources": {"r1": {"type": "HR"}},
        "rules": [
            {
                "id": "R1",
                "effect": "Permit",
                "actions": ["read"],
                "predicates": [
                    {
                        "id": "P1",
                        "op": "EQ",
                        "left_scope": "resource",
                        "left_attr": "type",
                        "right_value": "HR",
                    }
                ],
            },
            {
                "id": "R1",
                "effect": "Permit",
                "actions": ["read"],
                "predicates": [
                    {
                        "id": "P2",
                        "op": "EQ",
                        "left_scope": "resource",
                        "left_attr": "type",
                        "right_value": "HR",
                    }
                ],
            },
        ],
    }

    import json

    with pytest.raises(PolicyValidationError):
        parse_abac_text(json.dumps(payload), policy_id="dup")


def test_invalid_attribute_reference_raises() -> None:
    payload = {
        "meta": {"policy_id": "badattr"},
        "users": {"u1": {"position": "doctor"}},
        "resources": {"r1": {"type": "HR"}},
        "rules": [
            {
                "id": "R1",
                "effect": "Permit",
                "actions": ["read"],
                "predicates": [
                    {
                        "id": "P1",
                        "op": "EQ",
                        "left_scope": "subject",
                        "left_attr": "unknownAttr",
                        "right_value": "x",
                    }
                ],
            }
        ],
    }

    import json

    with pytest.raises(PolicyValidationError):
        parse_abac_text(json.dumps(payload), policy_id="badattr")


def test_type_mismatch_raises() -> None:
    payload = {
        "meta": {"policy_id": "typemismatch"},
        "users": {"u1": {"age": 30}},
        "resources": {"r1": {"type": "HR"}},
        "rules": [
            {
                "id": "R1",
                "effect": "Permit",
                "actions": ["read"],
                "predicates": [
                    {
                        "id": "P1",
                        "op": "EQ",
                        "left_scope": "subject",
                        "left_attr": "age",
                        "right_value": "thirty",
                    }
                ],
            }
        ],
    }

    import json

    with pytest.raises(PolicyValidationError):
        parse_abac_text(json.dumps(payload), policy_id="typemismatch")
