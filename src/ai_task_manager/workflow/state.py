from typing import Any, Literal, NotRequired, TypedDict

RouteKind = Literal["simple_read", "full"]


class WorkflowState(TypedDict):
    request: str
    conversation_id: str | None
    current_datetime: str
    timezone: str

    route: NotRequired[RouteKind]
    analyst_model: NotRequired[str]
    planner_model: NotRequired[str]
    validator_model: NotRequired[str]

    analysis: NotRequired[dict[str, Any] | None]
    plan: NotRequired[dict[str, Any] | None]
    verdict: NotRequired[dict[str, Any] | None]
    validator_feedback: NotRequired[list[str]]
    validation_attempts: NotRequired[int]
    fast_retry_done: NotRequired[bool]

    reply: NotRequired[str | None]
    error: NotRequired[str | None]
    pending_action_id: NotRequired[str | None]

    routing_reason: NotRequired[str]
