"""HTTP-маршрут общения с агентами."""

from fastapi import APIRouter, HTTPException

from ai_task_manager.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"description": "Агентный workflow пока не подключён"}},
)
def chat(payload: ChatRequest) -> ChatResponse:
    """Принимает сообщение; обработку подключим вместе с workflow."""
    raise HTTPException(status_code=503, detail="Агентный workflow пока не подключён")
