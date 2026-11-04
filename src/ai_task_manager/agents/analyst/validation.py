import re

from ai_task_manager.memory.inference import from_text, normalize_value

from .schema import AnalystInput, AnalystResult, ExtractedPreference

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
_BARE_ANSWER_RE = re.compile(
    r"\s*(?:да|нет|ага|угу|ок|окей|yes|no)\s*[.!?]?\s*",
    re.IGNORECASE,
)
_MEMORY_COMMAND_RE = re.compile(
    r"\b(?:запомни|запомнить|запиши|сохрани|учитывай)\b",
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


def _ground_preference(data: AnalystInput, result: AnalystResult) -> AnalystResult:
    """Сверяет предпочтение модели с текущим запросом пользователя."""
    question = AnalystResult(
        intent="clarification_required",
        missing_fields=["preference"],
        clarification_question="Уточните, какое предпочтение нужно запомнить.",
    )

    if not _MEMORY_COMMAND_RE.search(data.request):
        return question

    candidates = from_text(data.request)
    if len(candidates) == 1:
        candidate = candidates[0]
        return result.model_copy(
            update={
                "preference": ExtractedPreference(
                    kind=candidate.kind,
                    value=candidate.value,
                )
            }
        )

    preference = result.preference
    if preference is None or normalize_value(preference.kind, preference.value) is None:
        return question
    return result


class UnknownTaskIdsError(ValueError):
    def __init__(self, unknown_ids: set[int], user_supplied_ids: set[int]) -> None:
        self.unknown_ids = unknown_ids
        self.user_supplied_ids = user_supplied_ids
        super().__init__(f"Неизвестные ID задач: {sorted(unknown_ids)}")


def validate_task_references(data: AnalystInput, result: AnalystResult) -> AnalystResult:
    """Проверяет ID и не даёт модели повторить действие из истории по одному «да»."""
    if _BARE_ANSWER_RE.fullmatch(data.request) and result.intent not in {
        "clarification_required",
        "unsupported",
    }:
        return AnalystResult(
            intent="clarification_required",
            missing_fields=["request"],
            clarification_question=(
                "По одному ответу «да» действие не выполняется. Уточните запрос."
            ),
        )

    known_ids = {task.id for task in data.tasks}

    referenced_ids = set(result.task_ids)
    if result.task_id is not None:
        referenced_ids.add(result.task_id)

    if result.updates is not None:
        referenced_ids.update(result.updates.add_dependency_ids)
        referenced_ids.update(result.updates.remove_dependency_ids)

    unknown_ids = referenced_ids - known_ids
    if unknown_ids:
        user_supplied_ids = _explicit_task_ids(data.request) | _contextual_task_ids(data)
        raise UnknownTaskIdsError(unknown_ids, user_supplied_ids)

    if result.intent in {"update_task", "delete_task", "delete_tasks"}:
        explicit_ids = _explicit_task_ids(data.request) | _contextual_task_ids(data)
        missing_ids = referenced_ids - explicit_ids
        if missing_ids:
            return AnalystResult(
                intent="clarification_required",
                missing_fields=["task_id"],
                clarification_question="Укажи номер задачи, которую нужно изменить или удалить.",
            )

    if result.intent == "update_preference" and not result.tasks:
        return _ground_preference(data, result)

    return result
