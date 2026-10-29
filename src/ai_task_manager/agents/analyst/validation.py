import re

from .schema import AnalystInput, AnalystResult

_MARKED_ID_RE = re.compile(r"(?:[#№]|\bID\s*[:#№]?)\s*(\d+)\b", re.IGNORECASE)
_TASK_IDS_RE = re.compile(
    r"\bзадач(?:а|у|и|е|ей)\s+"
    r"(?:(?:с\s+номером|номер)\s+)?"
    r"(\d+(?:\s*(?:,|и)\s*\d+)*)\b",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE_RE = re.compile(
    r"\b(?:её|ее|ей|ею|его|нему|ней|эту|этой|этого)\b",
    re.IGNORECASE,
)
_ID_QUESTION_RE = re.compile(
    r"\b(?:укажи|назови|напиши|какой|нужен)\b.{0,60}\b(?:номер|id|идентификатор)\b",
    re.IGNORECASE,
)


def _explicit_task_ids(request: str) -> set[int]:
    ids = {int(value) for value in _MARKED_ID_RE.findall(request)}
    for group in _TASK_IDS_RE.findall(request):
        ids.update(int(value) for value in re.findall(r"\d+", group))
    return ids


def _contextual_task_ids(data: AnalystInput) -> set[int]:
    """Разрешает только однозначную ссылку на последний ответ или номер после вопроса."""
    if not data.history or data.history[-1].role != "assistant":
        return set()
    last_reply = data.history[-1].content
    if _ID_QUESTION_RE.search(last_reply) and re.fullmatch(r"\s*\d+\s*", data.request):
        return {int(data.request)}
    if _CONTEXT_REFERENCE_RE.search(data.request):
        ids = _explicit_task_ids(last_reply)
        return ids if len(ids) == 1 else set()
    return set()


def validate_task_references(data: AnalystInput, result: AnalystResult) -> AnalystResult:
    """Отклоняет ссылки модели на задачи, которых нет во входном контексте"""
    known_ids = {task.id for task in data.tasks}

    referenced_ids = set(result.task_ids)
    if result.task_id is not None:
        referenced_ids.add(result.task_id)

    if result.updates is not None:
        referenced_ids.update(result.updates.add_dependency_ids)
        referenced_ids.update(result.updates.remove_dependency_ids)

    unknown_ids = referenced_ids - known_ids
    if unknown_ids:
        raise ValueError(f"Неизвестные ID задач: {sorted(unknown_ids)}")

    if result.intent in {"update_task", "delete_task", "delete_tasks"}:
        explicit_ids = _explicit_task_ids(data.request) | _contextual_task_ids(data)
        missing_ids = referenced_ids - explicit_ids
        if missing_ids:
            return AnalystResult(
                intent="clarification_required",
                missing_fields=["task_id"],
                clarification_question="Укажи номер задачи, которую нужно изменить или удалить.",
            )
    return result
