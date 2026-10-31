"""HTTP-маршруты долгосрочной памяти (P12): просмотр, подтверждение догадок, изменение, удаление.

Хранилище — `db/preferences.py` (A13). Планировщик видит только подтверждённые значения,
поэтому предложение (`proposed`) до `/confirm` ни на что не влияет (FR-17, AC-19).
"""

import json
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai_task_manager.api.dependencies import get_db
from ai_task_manager.db.preferences import (
    confirm_preference,
    delete_preference,
    reject_preference,
    update_preference,
)
from ai_task_manager.memory.inference import describe, normalize_value
from ai_task_manager.memory.models import Preference, PreferenceKind

router = APIRouter(prefix="/api/v1/preferences", tags=["preferences"])
Db = Annotated[sqlite3.Connection, Depends(get_db)]


class PreferenceView(Preference):
    text: str


class PreferencesResponse(BaseModel):
    confirmed: list[PreferenceView]
    proposed: list[PreferenceView]


class PreferenceValue(BaseModel):
    value: str | list[str]


def _view(row: sqlite3.Row | Preference) -> PreferenceView:
    if isinstance(row, Preference):
        data: dict[str, Any] = row.model_dump()
    else:
        data = dict(row)
        data["value"] = json.loads(data.pop("value_json"))
    return PreferenceView(**data, text=describe(data["kind"], data["value"]))


@router.get("", response_model=PreferencesResponse)
def list_preferences(db: Db) -> PreferencesResponse:
    """Действующие предпочтения и догадки, ожидающие подтверждения."""
    rows = db.execute(
        "SELECT * FROM preferences WHERE status IN ('confirmed', 'proposed') ORDER BY id"
    ).fetchall()
    views = [_view(row) for row in rows]
    return PreferencesResponse(
        confirmed=[v for v in views if v.status == "confirmed"],
        proposed=[v for v in views if v.status == "proposed"],
    )


@router.post("/{preference_id}/confirm", response_model=PreferenceView)
def confirm(preference_id: int, db: Db) -> PreferenceView:
    """Подтвердить догадку: с этого момента её учитывает планировщик."""
    try:
        preference = confirm_preference(db, preference_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    return _view(preference)


@router.post("/{preference_id}/reject")
def reject(preference_id: int, db: Db) -> dict[str, Any]:
    """Отклонить догадку: она не применяется и больше не предлагается."""
    if not reject_preference(db, preference_id):
        raise HTTPException(status_code=409, detail="Предложение не найдено или уже обработано")
    db.commit()
    return {"id": preference_id, "status": "rejected"}


@router.put("/{kind}", response_model=PreferenceView)
def change(kind: PreferenceKind, payload: PreferenceValue, db: Db) -> PreferenceView:
    """Явно задать или изменить предпочтение; прежнее значение перестаёт действовать."""
    value = normalize_value(kind, payload.value)
    if value is None:
        raise HTTPException(status_code=422, detail="Значение не распознано")
    preference = update_preference(db, kind, value)
    db.commit()
    return _view(preference)


@router.delete("/{kind}")
def remove(kind: PreferenceKind, db: Db) -> dict[str, Any]:
    """Удалить предпочтение вместе с историей: дальше оно не применяется (FR-18)."""
    if not delete_preference(db, kind):
        raise HTTPException(status_code=404, detail="Такого предпочтения нет")
    db.commit()
    return {"kind": kind, "deleted": True}
