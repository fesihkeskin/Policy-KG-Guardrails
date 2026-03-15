from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth_brace = 0
    depth_paren = 0
    for ch in text:
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        if ch == sep and depth_brace == 0 and depth_paren == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def parse_scalar(value: str) -> Any:
    v = value.strip()
    if v == "":
        return ""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v


def parse_value(value: str) -> Any:
    v = value.strip()
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if not inner:
            return set()
        # ABAC files use whitespace-separated set values.
        tokens = [tok for tok in re.split(r"[\s,]+", inner) if tok]
        return set(parse_scalar(tok) for tok in tokens)
    return parse_scalar(v)


def infer_datatype(values: Iterable[Any]) -> tuple[str, bool]:
    vals = list(values)
    if not vals:
        return "string", False
    multivalue = any(isinstance(v, (set, list, tuple)) for v in vals)
    scalar_vals: list[Any] = []
    for value in vals:
        if isinstance(value, (set, list, tuple)):
            scalar_vals.extend(list(value))
        else:
            scalar_vals.append(value)
    if not scalar_vals:
        base = "string"
    elif all(isinstance(v, bool) for v in scalar_vals):
        base = "bool"
    elif all(isinstance(v, int) and not isinstance(v, bool) for v in scalar_vals):
        base = "int"
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in scalar_vals):
        base = "float"
    else:
        base = "string"
    if multivalue:
        return f"set[{base}]", True
    return base, False


def jdump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def nan_to_zero(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return value
