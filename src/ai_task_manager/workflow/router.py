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


def route_request(state: WorkflowState, settings: Settings) -> dict[str, str]:
    """Возвращает обновление State, не вызывая LLM и не меняя входной State."""
    route = classify_request(state["request"])
    simple = route == "simple_read"

    return {
        "route": route,
        "analyst_model": settings.fast_model if simple else settings.analyst_model,
        "planner_model": settings.fast_model if simple else settings.planner_model,
        "validator_model": settings.validator_model,
        "routing_reason": (
            "Точный запрос списка задач"
            if simple
            else "Запрос требует полноценного разбора или меняет данные"
        ),
    }
