# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def schemas_dir() -> Path:
    return repo_root() / "schemas"


def load_schema(filename: str) -> dict[str, Any]:
    path = schemas_dir() / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected == "null":
        return value is None
    return False


def _validate(instance: Any, schema: dict[str, Any], path: str, issues: list[ValidationIssue]) -> None:
    expected_type = schema.get("type")
    if expected_type:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_type_matches(instance, t) for t in expected_types):
            issues.append(ValidationIssue(path, f"type mismatch: expected {expected_types}, got {type(instance).__name__}"))
            return

    if "enum" in schema and instance not in schema["enum"]:
        issues.append(ValidationIssue(path, f"value {instance!r} not in enum {schema['enum']!r}"))

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and instance < minimum:
            issues.append(ValidationIssue(path, f"value {instance} < minimum {minimum}"))
        if maximum is not None and instance > maximum:
            issues.append(ValidationIssue(path, f"value {instance} > maximum {maximum}"))

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                issues.append(ValidationIssue(path, f"missing required property: {key}"))
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", issues)
            else:
                if schema.get("additionalProperties", True) is False:
                    issues.append(ValidationIssue(path, f"unexpected property: {key}"))

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                _validate(item, item_schema, f"{path}[{index}]", issues)


def validate_or_raise(instance: Any, schema: dict[str, Any], *, label: str = "instance") -> None:
    issues: list[ValidationIssue] = []
    _validate(instance, schema, "$", issues)
    if issues:
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise SchemaValidationError(f"{label} failed schema validation: {detail}")

