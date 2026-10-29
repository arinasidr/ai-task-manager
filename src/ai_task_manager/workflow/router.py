from math import isfinite

from ai_task_manager.config import Settings

from .state import RouteKind, WorkflowState

_SIMPLE_READ_REQUESTS = frozenset(
    {
        "покажи задачи",
        "покажи мои задачи",
        "покажи все задачи",
        "список задач",
        "покажи список задач",
        "какие у меня задачи",
    }
)


def classify_request(request: str) -> RouteKind:
    normalized = " ".join(request.casefold().strip().rstrip("?.!").split())
    if not normalized:
        raise ValueError("Пустой запрос")
    if normalized in _SIMPLE_READ_REQUESTS:
        return "simple_read"
    return "full"


def _model_requires_full_route(decision: object) -> bool:
    if not isinstance(decision, dict):
        return False

    for question in ("request_type", "route"):
        answer = decision.get(question)
        if not isinstance(answer, dict):
            continue
        choice = answer.get("choice")
        if question == "request_type" and choice == "read_tasks":
            continue
        if question == "route" and choice != "full":
            continue
        probabilities = answer.get("probabilities")
        probability = probabilities.get(choice) if isinstance(probabilities, dict) else None
        if type(probability) in (int, float) and isfinite(probability) and 0.75 <= probability <= 1:
            return True

    for question in ("needs_clarification", "bypass_confirmation"):
        answer = decision.get(question)
        if not isinstance(answer, dict):
            continue

        probability = answer.get("noul")
        if type(probability) in (int, float) and isfinite(probability) and 0.5 <= probability <= 1:
            return True

    return False


def route_request(state: WorkflowState, settings: Settings) -> dict[str, str]:
    """Ускоряет только известные запросы чтения; Tev1 может потребовать полный разбор."""
    exact_read = classify_request(state["request"]) == "simple_read"
    model_alert = _model_requires_full_route(state.get("decision"))

    # Ошибка классификатора не должна переводить изменение данных в ветку только чтения.
    simple = exact_read and not model_alert
    route: RouteKind = "simple_read" if simple else "full"

    if model_alert:
        reason = "Tev1: рекомендуется полный разбор запроса"
    elif exact_read:
        reason = "Точный запрос списка задач"
    else:
        reason = "Запрос требует полноценного разбора или меняет данные"

    return {
        "route": route,
        "analyst_model": settings.fast_model if simple else settings.analyst_model,
        "planner_model": settings.fast_model if simple else settings.planner_model,
        "validator_model": settings.validator_model,
        "routing_reason": reason,
    }
