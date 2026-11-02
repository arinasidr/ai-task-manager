import json
from pathlib import Path

from ai_task_manager.agents.llm import OllamaJsonLLM

JUDGE_MODEL = "phi4-mini:3.8b-q4_K_M"
CRITERIA = ("coverage", "order", "constraints", "explanation", "restraint")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            name: {"type": "integer", "minimum": 0, "maximum": 2}
            for name in CRITERIA
        },
        "reason": {"type": "string"},
    },
    "required": [*CRITERIA, "reason"],
    "additionalProperties": False,
}

RUBRIC = Path(__file__).with_name("judge_rubric.md").read_text(encoding="utf-8")


def judge_record(record: dict, model: str = JUDGE_MODEL) -> dict:
    """Оценить сохранённый план другой моделью, без повторного запуска агентов."""
    case = record["case"]
    if not case.get("judge"):
        raise ValueError("Этот сценарий не предназначен для оценки судьёй")

    run = record["run"]
    before = run["before"]
    result = run["result"]
    titles_by_id = {task["id"]: task["title"] for task in before["tasks"]}
    plan = result.get("plan") or {}

    context = {
        "request": case["message"],
        "now": case["now"],
        "timezone": case["timezone"],
        "tasks": [
            {
                key: task[key]
                for key in (
                    "id", "title", "status", "priority",
                    "due_date", "due_time", "due_timezone",
                )
            }
            for task in before["tasks"]
        ],
        "dependencies": case["initial_db"]["dependencies"],
        "preferences": case["initial_db"].get("preferences", []),
        "plan": plan,
        "ordered_titles": [
            titles_by_id.get(task_id, f"Неизвестный ID {task_id}")
            for task_id in plan.get("ordered_task_ids", [])
        ],
        "reply": result["reply"],
    }

    instruction = (
        RUBRIC
        + "\nОцени план и ответ только по переданным фактам. "
        + "Верни JSON с пятью целыми оценками 0–2: "
        + ", ".join(CRITERIA)
        + ". В reason объясни каждую оценку на конкретных задачах: "
        + "назови срок, зависимость или приоритет только если он есть во входных данных. "
        + "Если приоритеты не заданы, прямо скажи об этом. "
        + "Не пиши общую похвалу и не утверждай, что учтены отсутствующие ограничения."
    )
    raw = OllamaJsonLLM(model).complete_json(
        instruction,
        json.dumps(context, ensure_ascii=False),
        JUDGE_SCHEMA,
    )

    scores = {name: raw[name] for name in CRITERIA}
    if any(type(value) is not int or not 0 <= value <= 2 for value in scores.values()):
        raise ValueError(f"Некорректные баллы судьи: {scores}")
    if not isinstance(raw["reason"], str) or not raw["reason"].strip():
        raise ValueError("Судья не объяснил оценку")

    return {
        "case_id": case["id"],
        "model": model,
        "scores": scores,
        "total": sum(scores.values()),
        "reason": raw["reason"].strip(),
    }