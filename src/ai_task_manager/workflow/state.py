from typing import Any, Literal, NotRequired, TypedDict


class DialogueMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


RouteKind = Literal["simple_read", "full"]


class WorkflowState(TypedDict):
    request: str
    conversation_id: str | None
    current_datetime: str
    timezone: str
    history: NotRequired[list[DialogueMessage]]

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

    decision: NotRequired[dict[str, Any] | None]
    decision_error: NotRequired[str | None]

    # P12: предложенное предпочтение, на которое ждём «да/нет» (живёт между ходами диалога)
    pending_preference: NotRequired[dict[str, Any] | None]
    # P12: id предложения, заданного в этом ходе (для ответа API)
    preference_id: NotRequired[int | None]
