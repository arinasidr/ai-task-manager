"""Схема структурированного ответа модели и упрощённая проверка JSON по этой схеме.

Схема одна для всех моделей: её же передаём движку (Ollama `format`, OpenAI-совместимый
`response_format`), чтобы модели сравнивались в одинаковых условиях.
Проверка написана вручную, чтобы бенчмарк не тянул дополнительную зависимость jsonschema.
Поддерживается только используемое здесь подмножество JSON Schema:
type, enum, anyOf, properties, required, items, additionalProperties.
"""

from typing import Any

INTENTS = [
    "create_task",
    "get_tasks",
    "update_task",
    "delete_task",
    "delete_tasks",
    "plan_tasks",
    "update_preference",
    "clarification_required",
    "unsupported",
]

WARNINGS = [
    "task_not_found",
    "dependency_cycle",
    "incomplete_dependencies",
    "dependent_tasks_will_be_detached",
    "dependency_deadline_conflict",
    "invalid_date",
    "ambiguous_local_time",
    "past_deadline",
    "ambiguous_request",
]

MISSING_FIELDS = ["task_id", "title", "deadline", "timezone", "time_disambiguation"]

PRIORITIES = ["low", "medium", "high"]
STATUSES = ["todo", "in_progress", "done"]
WORK_TIMES = ["morning", "afternoon", "evening", "night"]


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_INT_LIST = {"type": "array", "items": {"type": "integer"}}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": _nullable({"type": "string", "enum": PRIORITIES}),
                    "deadline": _nullable({"type": "string"}),
                },
                "required": ["title", "priority", "deadline"],
                "additionalProperties": False,
            },
        },
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "depends_on": {"type": "string"},
                },
                "required": ["task", "depends_on"],
                "additionalProperties": False,
            },
        },
        "task_id": _nullable({"type": "integer"}),
        "task_ids": _INT_LIST,
        "updates": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "deadline": _nullable({"type": "string"}),
                "priority": _nullable({"type": "string", "enum": PRIORITIES}),
                "status": {"type": "string", "enum": STATUSES},
                "add_dependency_ids": _INT_LIST,
                "remove_dependency_ids": _INT_LIST,
                "preferred_work_time": {"type": "string", "enum": WORK_TIMES},
                "working_days": _STR_LIST,
                "preferred_task_types": _STR_LIST,
            },
            "additionalProperties": False,
        },
        "ordered_task_ids": _INT_LIST,
        "matching_tasks": _INT_LIST,
        "candidate_task_ids": _INT_LIST,
        "answer_status": {"type": "string", "enum": ["found", "not_found"]},
        "missing_fields": {"type": "array", "items": {"type": "string", "enum": MISSING_FIELDS}},
        "should_change_deadlines": {"type": "boolean"},
        "replaces_existing": {"type": "boolean"},
        "warnings": {"type": "array", "items": {"type": "string", "enum": WARNINGS}},
        "requires_confirmation": {"type": "boolean"},
        "should_execute": {"type": "boolean"},
        "message": {"type": "string"},
    },
    "required": ["intent", "warnings", "requires_confirmation", "should_execute"],
    "additionalProperties": False,
}

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool в Python является подклассом int, поэтому исключаем его явно
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value: Any, schema: dict[str, Any] = OUTPUT_SCHEMA, path: str = "$") -> list[str]:
    """Возвращает список нарушений схемы (пустой список, если ответ корректен)."""
    if "anyOf" in schema:
        if any(not validate(value, option, path) for option in schema["anyOf"]):
            return []
        return [f"{path}: значение {value!r} не подходит ни под один вариант"]

    expected_type = schema.get("type")
    if expected_type and not _TYPE_CHECKS[expected_type](value):
        return [f"{path}: ожидался тип {expected_type}, получено {type(value).__name__}"]

    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: недопустимое значение {value!r}")

    if expected_type == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: нет обязательного поля {key!r}")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: лишнее поле {key!r}")

    if expected_type == "array" and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{index}]"))

    return errors
