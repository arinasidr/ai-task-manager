from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

PreferenceKind = Literal["preferred_work_time", "working_days", "preferred_task_types"]
PreferenceStatus = Literal["proposed", "confirmed", "superseded", "rejected"]
PreferenceSource = Literal["explicit", "inferred"]


class PreferenceInput(BaseModel):
    kind: PreferenceKind
    value: str | list[str]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Предпочтение не может быть пустым")
            return value.strip()

        cleaned = [item.strip() for item in value]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("Список предпочтений не может быть пустым")
        return cleaned


class Preference(PreferenceInput):
    id: int
    status: PreferenceStatus
    source: PreferenceSource
    created_at: datetime
    updated_at: datetime
