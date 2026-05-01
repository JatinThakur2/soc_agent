"""Robust JSON parser for small-LLM outputs (handles nesting, markdown fences, etc.)."""
import json
import re


def safe_parse(text: str, fallback: dict) -> dict:
    """
    Parse JSON from LLM output robustly.

    Handles three common phi3.5/small-LLM quirks:
    1. Valid JSON returned directly
    2. JSON buried inside prose ("Here is the output: {...}")
    3. Real JSON nested as a string value inside an outer JSON envelope
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Attempt 1: direct parse
    parsed = _try_parse(text)
    if parsed is not None:
        return _unwrap_nested(parsed, fallback)

    # Attempt 2: extract first complete {...} block
    start = text.find("{")
    end = _find_matching_brace(text, start) if start >= 0 else -1
    if start >= 0 and end > start:
        parsed = _try_parse(text[start:end + 1])
        if parsed is not None:
            return _unwrap_nested(parsed, fallback)

    # Attempt 3: last resort — take widest {...} span
    start2, end2 = text.find("{"), text.rfind("}")
    if start2 >= 0 and end2 > start2:
        parsed = _try_parse(text[start2:end2 + 1])
        if parsed is not None:
            return _unwrap_nested(parsed, fallback)

    return fallback


def _try_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _find_matching_brace(text: str, start: int) -> int:
    """Find the index of the closing brace that matches text[start]."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _unwrap_nested(parsed: dict, fallback: dict) -> dict:
    """
    phi3.5 sometimes outputs the real answer as a JSON string nested inside
    one of the outer fields (e.g. executive_summary = '{"final_severity": ...}').
    Detect and unwrap it.
    """
    for v in parsed.values():
        if not isinstance(v, str):
            continue
        candidate = v.strip()
        if not candidate.startswith("{"):
            continue
        inner = _try_parse(candidate)
        if inner and isinstance(inner, dict) and len(inner) >= 2:
            # Only use the inner dict if it has more meaningful keys
            if len(inner) >= len(parsed):
                return inner
    return parsed
