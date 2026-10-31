"""Агент-планировщик.

Получает разбор запроса от аналитика (`Analysis`), читает актуальные задачи из БД и строит план
(`Plan`): какие Tools вызвать, нужно ли подтверждение, в каком порядке выполнять задачи.

Как устроен:
- **Действия с задачами** (создать, изменить, удалить, связать) план строит код. Аналитик уже
  извлёк всё нужное, а перевод в вызовы Tools однозначен — LLM здесь только добавила бы ошибок
  («не делаем LLM-агента для CRUD только ради количества», ТЗ, раздел 5). Код же проверяет, что
  задачи существуют, связи не создают циклов и что нужно подтверждение.
- **Планирование порядка** (`plan_tasks`) делает LLM: она взвешивает сроки, приоритеты,
  зависимости и предпочтения и объясняет выбор. После неё код гарантирует жёсткие правила:
  только существующие задачи, каждая один раз, зависимости соблюдены. Если модель не справилась
  дважды, порядок строится по правилам (fallback), чтобы пользователь всё равно получил ответ.

Планировщик только читает БД. Записывают Tools после валидатора и подтверждения.
"""

import difflib
import heapq
import json
import re
import sqlite3
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from ai_task_manager.agents.contracts import (
    Analysis,
    ExtractedDependency,
    Operation,
    Plan,
    Recommendation,
)
from ai_task_manager.agents.llm import JsonLLM, LLMError
from ai_task_manager.db.crud_func import (
    get_dependencies,
    get_task,
    get_unfinished_predecessors,
    list_tasks,
)
from ai_task_manager.memory.inference import (
    WEEKDAYS,
    describe,
    normalize_days,
    normalize_value,
    normalize_work_time,
)
from ai_task_manager.prompting import build_prompt

PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.md"
# как выполнять работу: процедура, типичные случаи, чек-лист (skills/task-planning/SKILL.md)
SKILL = "task-planning"

PLANNER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ordered_task_ids": {"type": "array", "items": {"type": "integer"}},
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "recommended_priority": {
                        "anyOf": [
                            {"type": "string", "enum": ["low", "medium", "high"]},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["task_id", "reason", "recommended_priority"],
            },
        },
        "message": {"type": "string"},
    },
    "required": ["ordered_task_ids", "recommendations", "message"],
}

_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2, None: 3}
_UPDATE_FIELDS = {"title", "description", "priority"}
_PREFERENCE_FIELDS = {"preferred_work_time", "working_days", "preferred_task_types"}
_ANALYST_DECIDES = {"clarification_required", "unsupported"}


def load_system_prompt() -> str:
    return build_prompt(PROMPT_PATH.read_text(encoding="utf-8"), [SKILL])


# ---------------------------------------------------------------- сроки


def deadline_fields(value: str | None, tz: ZoneInfo) -> dict[str, str | None]:
    """Переводит срок из ответа аналитика в поля задачи БД.

    «2026-10-20» — только дата; «2026-10-20T18:30:00+03:00» — момент: локальные дата и время,
    часовой пояс (если смещение совпадает с поясом пользователя) и момент в UTC.
    Некорректная дата (например, 31 февраля) — ValueError.
    """
    empty: dict[str, str | None] = {
        "due_date": None,
        "due_time": None,
        "due_timezone": None,
        "due_at_utc": None,
    }
    if value is None or not str(value).strip():
        return empty
    text = str(value).strip()
    if len(text) == 10:
        return {**empty, "due_date": date.fromisoformat(text).isoformat()}

    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    same_zone = moment.utcoffset() == tz.utcoffset(moment.replace(tzinfo=None))
    return {
        "due_date": moment.date().isoformat(),
        "due_time": moment.time().replace(tzinfo=None).isoformat(),
        "due_timezone": tz.key if same_zone else None,
        "due_at_utc": moment.astimezone(UTC).isoformat(),
    }


def deadline_moment(fields: dict[str, Any], tz: ZoneInfo) -> datetime | None:
    """Момент срока для сравнения: точное время, а для срока «на дату» — конец этого дня."""
    if fields.get("due_at_utc"):
        moment = datetime.fromisoformat(fields["due_at_utc"])
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    if fields.get("due_date"):
        return datetime.combine(date.fromisoformat(fields["due_date"]), time.max, tzinfo=tz)
    return None


def _normalize_title(title: str) -> str:
    """Название для сравнения: регистр, «ё», кавычки, знаки препинания и пробелы не важны."""
    text = title.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _match_title(title: str, refs: dict[str, str]) -> str | None:
    """Находит новую задачу по названию из связи: точно, по вхождению или по похожести.

    Модели часто пишут в связях название чуть иначе, чем в списке задач («сбор данных» и
    «Собрать данные для отчёта»).
    """
    key = _normalize_title(title)
    if key in refs:
        return refs[key]
    contained = [ref for name, ref in refs.items() if key and (key in name or name in key)]
    if len(set(contained)) == 1:
        return contained[0]
    scored = sorted(
        ((difflib.SequenceMatcher(None, key, name).ratio(), ref) for name, ref in refs.items()),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.75 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        return scored[0][1]
    return None


_ORDER_MARKERS = re.compile(
    r"\b(сначала|потом|затем|после этого|после чего|а после|в конце)\b", re.IGNORECASE
)


def _order_from_request(
    request: str, tasks: list[Any], *, strict: bool = False
) -> tuple[int, int] | None:
    """Порядок двух новых задач по тексту запроса: «сначала A, потом B» → (A, B).

    Только для явных слов порядка и ровно двух задач; задачи ищутся в тексте по первому
    слову названия. Если что-то не сходится — None, и пользователь получит вопрос.

    `strict` — аналитик связи не дал вовсе: тогда слово порядка должно стоять перед первой
    задачей («сначала») или между задачами («A, потом B»), а не где угодно в запросе
    («добавь A и B, потом покажи список» — это не связь).
    """
    if len(tasks) != 2 or not _ORDER_MARKERS.search(request):
        return None
    text = _normalize_title(request)
    positions = []
    for task in tasks:
        words = _normalize_title(task.title).split()
        if not words:
            return None
        found = re.search(rf"\b{re.escape(words[0][:5])}", text)
        if found is None:
            return None
        positions.append(found.start())
    if positions[0] == positions[1]:
        return None
    first_by_text = (0, 1) if positions[0] < positions[1] else (1, 0)
    if strict:
        markers = [m for m in _ORDER_MARKERS.finditer(text) if m.start() < max(positions)]
        if not markers:
            return None
    # «написать отчёт, но сначала собрать данные»: первой идёт задача сразу после «сначала»
    first_marker = re.search(r"\bсначала\b", text)
    if first_marker is not None:
        after = [i for i, pos in enumerate(positions) if pos > first_marker.start()]
        if len(after) == 1:
            return (after[0], 1 - after[0])
    return first_by_text


def _deadline_text(task: dict[str, Any]) -> str | None:
    if not task.get("due_date"):
        return None
    if task.get("due_time"):
        zone = f" ({task['due_timezone']})" if task.get("due_timezone") else ""
        return f"{task['due_date']} {task['due_time'][:5]}{zone}"
    return task["due_date"]


# ---------------------------------------------------------------- граф зависимостей


def _creates_cycle(connection: sqlite3.Connection, predecessor_id: int, successor_id: int) -> bool:
    """Создаст ли связь predecessor → successor цикл (только чтение, без вставки)."""
    if predecessor_id == successor_id:
        return True
    row = connection.execute(
        """
        WITH RECURSIVE reachable(task_id) AS (
            SELECT successor_task_id FROM task_dependencies WHERE predecessor_task_id = ?
            UNION
            SELECT d.successor_task_id
            FROM task_dependencies AS d JOIN reachable AS r ON d.predecessor_task_id = r.task_id
        )
        SELECT 1 FROM reachable WHERE task_id = ? LIMIT 1
        """,
        (successor_id, predecessor_id),
    ).fetchone()
    return row is not None


def _ordered_respecting_dependencies(
    task_ids: list[int], edges: set[tuple[int, int]], preferred: list[int]
) -> list[int]:
    """Топологическая сортировка: зависимости соблюдены, остальное — как в `preferred`.

    Kahn с очередью по позиции в `preferred`: из задач, которые уже можно начинать, первой идёт
    та, что раньше в желаемом порядке.
    """
    rank = {task_id: index for index, task_id in enumerate(preferred)}
    allowed = set(task_ids)
    incoming = {task_id: 0 for task_id in task_ids}
    outgoing: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    for predecessor, successor in edges:
        if predecessor in allowed and successor in allowed:
            incoming[successor] += 1
            outgoing[predecessor].append(successor)

    queue = [(rank.get(t, len(rank)), t) for t in task_ids if incoming[t] == 0]
    heapq.heapify(queue)
    result: list[int] = []
    while queue:
        _, task_id = heapq.heappop(queue)
        result.append(task_id)
        for successor in outgoing[task_id]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                heapq.heappush(queue, (rank.get(successor, len(rank)), successor))
    # В БД циклов быть не должно; если всё же есть, оставшиеся задачи добавляем как есть.
    result += [t for t in preferred if t in allowed and t not in result]
    return result


# ---------------------------------------------------------------- планировщик


class Planner:
    def __init__(
        self,
        llm: JsonLLM | None = None,
        *,
        timezone: str = "Europe/Moscow",
        max_llm_attempts: int = 2,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tz = ZoneInfo(timezone)
        self.max_llm_attempts = max_llm_attempts
        self.system_prompt = system_prompt or load_system_prompt()
        self._request = ""
        # Связи, исправленные кодом (см. _order_from_request); узел графа кладёт их в разбор,
        # чтобы валидатор сверял план с исправленными, а не с ошибочными связями аналитика.
        self.repaired_dependencies: list[ExtractedDependency] | None = None

    def plan(
        self,
        analysis: Analysis | dict[str, Any],
        connection: sqlite3.Connection,
        *,
        request: str = "",
        now: datetime | None = None,
        feedback: list[str] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> Plan:
        """Строит план. `feedback` — замечания валидатора при повторной попытке."""
        if not isinstance(analysis, Analysis):
            analysis = Analysis.model_validate(analysis)
        now = (now or datetime.now(UTC)).astimezone(self.tz)
        self._request = request
        self.repaired_dependencies = None

        handlers = {
            "create_task": self._plan_create,
            "update_task": self._plan_update,
            "delete_task": self._plan_delete,
            "delete_tasks": self._plan_delete,
            "get_tasks": self._plan_read,
            "update_preference": self._plan_preference,
        }
        if analysis.intent == "plan_tasks":
            plan = self._plan_order(analysis, connection, request, now, feedback, preferences)
        elif analysis.intent in handlers:
            plan = handlers[analysis.intent](analysis, connection, now)
        else:
            plan = self._clarify(analysis, analysis.missing_fields)

        if feedback and plan.source == "rules":
            # План по правилам от замечаний не меняется: если валидатор его отклоняет, решать
            # должен пользователь, а не повторный цикл.
            plan.reasoning.append("Замечания валидатора: " + "; ".join(feedback))
        return plan

    # ---------------------------------------------------------- уточнение

    def _clarify(
        self, analysis: Analysis, missing: list[str], warnings: list[str] | tuple[str, ...] = ()
    ) -> Plan:
        # Предупреждения аналитика берём, только если уточнения требует он сам. Если отказ из-за
        # проверок кода, причины — только найденные кодом.
        from_analyst = analysis.warnings if analysis.intent in _ANALYST_DECIDES else []
        all_warnings = list(dict.fromkeys([*from_analyst, *warnings]))
        if analysis.message:
            message = analysis.message
        elif "task_id" in missing:
            message = "Уточните, пожалуйста, ID задачи: могу показать список задач."
        elif "invalid_date" in all_warnings:
            message = "Такой даты нет в календаре. Уточните, пожалуйста, срок."
        elif "ambiguous_local_time" in all_warnings:
            message = "Это время в указанный день неоднозначно из-за перевода часов. Уточните."
        elif analysis.intent == "unsupported":
            message = "Я умею работать только с задачами: создавать, менять, удалять и планировать."
        else:
            message = "Уточните, пожалуйста, запрос."
        return Plan(
            intent=analysis.intent,
            warnings=all_warnings,
            needs_clarification=True,
            message_to_user=message,
            reasoning=["Данных недостаточно для действия, нужен ответ пользователя"],
        )

    # ---------------------------------------------------------- чтение

    def _plan_read(self, analysis: Analysis, connection: sqlite3.Connection, now: datetime) -> Plan:
        return Plan(
            intent="get_tasks",
            operations=[Operation(tool="list_tasks")],
            message_to_user="Показываю ваши задачи.",
            reasoning=["Запрос только на чтение: подтверждение не нужно"],
        )

    # ---------------------------------------------------------- создание

    def _plan_create(
        self, analysis: Analysis, connection: sqlite3.Connection, now: datetime
    ) -> Plan:
        if not analysis.tasks:
            return self._clarify(analysis, ["title"])

        operations: list[Operation] = []
        # Предупреждения аналитика — догадки модели; в плане действий их ставит только код.
        warnings: list[str] = []
        reasoning: list[str] = []
        refs: dict[str, str] = {}
        moments: dict[str, datetime | None] = {}

        for index, task in enumerate(analysis.tasks, start=1):
            try:
                fields = deadline_fields(task.deadline, self.tz)
            except ValueError:
                return self._clarify(analysis, ["deadline"], ["invalid_date"])
            ref = f"new{index}"
            args = {"title": task.title.strip()}
            if task.priority:
                args["priority"] = task.priority
            args.update({key: value for key, value in fields.items() if value is not None})
            operations.append(Operation(tool="create_task", ref=ref, args=args))
            refs[_normalize_title(task.title)] = ref
            moments[ref] = deadline_moment(fields, self.tz)
            if moments[ref] is not None and moments[ref] < now:
                warnings.append("past_deadline")
                reasoning.append(
                    f"Срок задачи «{task.title}» уже прошёл — создаём, но предупреждаем"
                )

        dependencies = list(analysis.dependencies)
        if not dependencies:
            # Аналитик пропустил связь, хотя порядок явно назван в запросе («сначала A, потом B»):
            # без неё валидатор справедливо отклоняет план. Восстанавливаем связь кодом.
            recovered = _order_from_request(self._request, analysis.tasks, strict=True)
            if recovered is not None:
                first, second = recovered
                dependencies = [
                    ExtractedDependency(
                        task=analysis.tasks[second].title,
                        depends_on=analysis.tasks[first].title,
                    )
                ]
                self.repaired_dependencies = dependencies
                reasoning.append(
                    "Аналитик не указал связь; порядок восстановлен кодом по словам "
                    "«сначала … потом» в запросе"
                )

        for dependency in dependencies:
            successor = _match_title(dependency.task, refs)
            predecessor = _match_title(dependency.depends_on, refs)
            broken = successor is None or predecessor is None or successor == predecessor
            if broken and len(dependencies) == 1:
                # Аналитик ошибся в связи (например, «A ← A»), но порядок явно назван в запросе
                # («сначала A, потом B») — восстанавливаем его кодом по тексту запроса.
                recovered = _order_from_request(self._request, analysis.tasks)
                if recovered is not None:
                    first, second = recovered
                    self.repaired_dependencies = [
                        ExtractedDependency(
                            task=analysis.tasks[second].title,
                            depends_on=analysis.tasks[first].title,
                        )
                    ]
                    predecessor, successor = f"new{first + 1}", f"new{second + 1}"
                    broken = False
                    reasoning.append(
                        "Связь от аналитика некорректна "
                        f"({dependency.task!r} ← {dependency.depends_on!r}); "
                        "порядок восстановлен кодом по словам «сначала … потом» в запросе"
                    )
            if broken:
                plan = self._clarify(analysis, [], ["ambiguous_request"])
                plan.message_to_user = (
                    "Не понял, какая задача должна идти после какой "
                    f"(«{dependency.task}» после «{dependency.depends_on}»). "
                    "Напишите, пожалуйста, порядок ещё раз: «сначала …, потом …»."
                )
                plan.reasoning.append(
                    "Связь не сопоставилась с задачами из запроса: "
                    f"{dependency.task!r} ← {dependency.depends_on!r}, задачи: {list(refs)}"
                )
                return plan
            operations.append(
                Operation(
                    tool="add_dependency",
                    args={
                        "predecessor_task_id": f"${predecessor}",
                        "successor_task_id": f"${successor}",
                    },
                )
            )
            before, after = moments[predecessor], moments[successor]
            if before is not None and after is not None and after < before:
                warnings.append("dependency_deadline_conflict")
                reasoning.append(
                    f"Срок «{dependency.task}» раньше срока задачи, от которой она зависит"
                )

        titles = ", ".join(f"«{op.args['title']}»" for op in operations if op.tool == "create_task")
        count = len(analysis.tasks)
        return Plan(
            intent="create_task",
            operations=operations,
            warnings=list(dict.fromkeys(warnings)),
            message_to_user=f"Создам {'задачу' if count == 1 else f'задачи ({count})'}: {titles}.",
            reasoning=reasoning or ["Создание задач не требует подтверждения"],
        )

    # ---------------------------------------------------------- изменение

    def _plan_update(
        self, analysis: Analysis, connection: sqlite3.Connection, now: datetime
    ) -> Plan:
        task_id = analysis.task_id
        if task_id is None:
            return self._clarify(analysis, ["task_id"])
        task = get_task(connection, task_id)
        if task is None:
            plan = self._clarify(analysis, [], ["task_not_found"])
            plan.message_to_user = f"Задачи #{task_id} нет. Проверьте ID: могу показать список."
            return plan
        if not analysis.updates:
            plan = self._clarify(analysis, [])
            plan.message_to_user = f"Что изменить в задаче #{task_id}?"
            return plan

        updates = dict(analysis.updates)
        operations: list[Operation] = []
        # Предупреждения аналитика — догадки модели; в плане действий их ставит только код.
        warnings: list[str] = []
        reasoning: list[str] = []
        requires_confirmation = False
        blocked: list[str] = []

        field_updates = {key: updates.pop(key) for key in list(updates) if key in _UPDATE_FIELDS}
        if "deadline" in updates:
            try:
                field_updates.update(deadline_fields(updates.pop("deadline"), self.tz))
            except ValueError:
                return self._clarify(analysis, ["deadline"], ["invalid_date"])
        if field_updates:
            operations.append(
                Operation(tool="update_task", args={"task_id": task_id, "updates": field_updates})
            )

        if "status" in updates:
            status = updates.pop("status")
            if status == "done" and task["status"] != "done":
                blockers = get_unfinished_predecessors(connection, task_id)
                if blockers:
                    requires_confirmation = True
                    warnings.append("incomplete_dependencies")
                    reasoning.append(
                        f"Задача #{task_id} зависит от незавершённых задач {blockers}: "
                        "завершение только после подтверждения (FR-06)"
                    )
            operations.append(
                Operation(tool="change_task_status", args={"task_id": task_id, "status": status})
            )

        for predecessor_id in updates.pop("add_dependency_ids", []) or []:
            if get_task(connection, predecessor_id) is None:
                warnings.append("task_not_found")
                blocked.append(f"задачи #{predecessor_id} нет")
            elif _creates_cycle(connection, predecessor_id, task_id):
                warnings.append("dependency_cycle")
                blocked.append(f"связь с #{predecessor_id} создаст цикл")
            else:
                operations.append(
                    Operation(
                        tool="add_dependency",
                        args={"predecessor_task_id": predecessor_id, "successor_task_id": task_id},
                    )
                )

        for predecessor_id in updates.pop("remove_dependency_ids", []) or []:
            operations.append(
                Operation(
                    tool="remove_dependency",
                    args={"predecessor_task_id": predecessor_id, "successor_task_id": task_id},
                )
            )

        if updates:
            reasoning.append(f"Неизвестные поля проигнорированы: {sorted(updates)}")

        if blocked:
            # Частично выполнять запрос нельзя: пользователь ждёт всё изменение целиком.
            plan = self._clarify(analysis, [], warnings)
            plan.message_to_user = (
                f"Не могу изменить задачу #{task_id}: " + "; ".join(blocked) + "."
            )
            plan.reasoning = reasoning + ["Изменение отклонено проверками кода"]
            return plan

        message = f"Изменю задачу #{task_id} «{task['title']}»."
        if requires_confirmation:
            message = (
                f"Задача #{task_id} зависит от незавершённых задач. Всё равно отметить её "
                "выполненной?"
            )
        return Plan(
            intent="update_task",
            operations=operations,
            warnings=list(dict.fromkeys(warnings)),
            requires_confirmation=requires_confirmation,
            message_to_user=message,
            reasoning=reasoning or ["Изменение одной существующей задачи"],
        )

    # ---------------------------------------------------------- удаление

    def _plan_delete(
        self, analysis: Analysis, connection: sqlite3.Connection, now: datetime
    ) -> Plan:
        ids = list(analysis.task_ids)
        if analysis.task_id is not None and analysis.task_id not in ids:
            ids.insert(0, analysis.task_id)
        if not ids:
            return self._clarify(analysis, ["task_id"])

        missing = [task_id for task_id in ids if get_task(connection, task_id) is None]
        if missing:
            plan = self._clarify(analysis, [], ["task_not_found"])
            plan.message_to_user = f"Задач {', '.join(f'#{i}' for i in missing)} нет. Проверьте ID."
            return plan

        # Предупреждения аналитика — догадки модели; в плане действий их ставит только код.
        warnings: list[str] = []
        details: list[str] = []
        for task_id in ids:
            links = get_dependencies(connection, task_id)
            linked = sorted({*links["predecessors"], *links["successors"]} - set(ids))
            if linked:
                warnings.append("dependent_tasks_will_be_detached")
                details.append(
                    f"связи #{task_id} с {', '.join(f'#{i}' for i in linked)} будут удалены"
                )

        names = ", ".join(f"#{task_id}" for task_id in ids)
        message = f"Удалить {'задачу' if len(ids) == 1 else 'задачи'} {names}?"
        if details:
            message += " " + "; ".join(details).capitalize() + "."
        return Plan(
            intent=analysis.intent,
            operations=[Operation(tool="delete_task", args={"task_id": i}) for i in ids],
            warnings=list(dict.fromkeys(warnings)),
            requires_confirmation=True,
            message_to_user=message,
            reasoning=[
                "Удаление необратимо: всегда только после подтверждения (FR-07), "
                "даже если пользователь просит без него"
            ],
        )

    # ---------------------------------------------------------- предпочтения

    def _plan_preference(
        self, analysis: Analysis, connection: sqlite3.Connection, now: datetime
    ) -> Plan:
        values = {k: v for k, v in analysis.updates.items() if k in _PREFERENCE_FIELDS}
        if not values:
            return self._clarify(analysis, [])
        return Plan(
            intent="update_preference",
            operations=[Operation(tool="update_preference", args=values)],
            message_to_user="Запомню это.",
            reasoning=["Явная просьба запомнить: новое значение заменит прежнее (P04, раздел 5)"],
        )

    # ---------------------------------------------------------- планирование порядка

    def _plan_order(
        self,
        analysis: Analysis,
        connection: sqlite3.Connection,
        request: str,
        now: datetime,
        feedback: list[str] | None,
        preferences: dict[str, Any] | None,
    ) -> Plan:
        tasks = [task for task in list_tasks(connection) if task["status"] != "done"]
        if not tasks:
            return Plan(
                intent="plan_tasks",
                message_to_user="Незавершённых задач нет — планировать нечего.",
                reasoning=["Список незавершённых задач пуст, модель не вызывалась"],
            )

        by_id = {task["id"]: task for task in tasks}
        edges: set[tuple[int, int]] = set()
        for task_id in by_id:
            links = get_dependencies(connection, task_id)
            edges.update((p, task_id) for p in links["predecessors"] if p in by_id)

        moments = {task_id: deadline_moment(task, self.tz) for task_id, task in by_id.items()}
        warnings: list[str] = []
        conflicts: list[dict[str, Any]] = []
        for predecessor, successor in sorted(edges):
            before, after = moments[predecessor], moments[successor]
            if before is not None and after is not None and after < before:
                conflicts.append({"task_id": successor, "depends_on_id": predecessor})
        if conflicts:
            warnings.append("dependency_deadline_conflict")
        if any(m is not None and m < now for m in moments.values()):
            warnings.append("past_deadline")

        prefs = _PreferenceFacts(preferences or {}, by_id, moments, now)
        baseline = sorted(
            by_id,
            key=lambda t: (
                moments[t] is None,
                moments[t] or now,
                _PRIORITY_RANK[by_id[t]["priority"]],
                # предпочтения не спорят со сроками и приоритетами: только при прочих равных
                t not in prefs.preferred_type_ids,
                t,
            ),
        )
        rule_order = _ordered_respecting_dependencies(list(by_id), edges, baseline)

        conflict_note = [
            f"Срок задачи #{c['task_id']} раньше срока задачи #{c['depends_on_id']}, "
            "от которой она зависит"
            for c in conflicts
        ]

        if self.llm is not None:
            context = {
                "request": request,
                "now": now.isoformat(timespec="minutes"),
                # Факты считает код: маленькие модели плохо считают даты и путают связи.
                "tasks": [
                    {
                        "id": t["id"],
                        "title": t["title"],
                        "status": t["status"],
                        "priority": t["priority"],
                        "deadline": _deadline_text(t),
                        "days_left": _days_left(moments[t["id"]], now),
                        "blocked_by": sorted(p for p, s in edges if s == t["id"]),
                        "blocks": sorted(s for p, s in edges if p == t["id"]),
                        **prefs.task_facts(t["id"]),
                    }
                    for t in tasks
                ],
                "dependencies": [
                    {"task_id": succ, "depends_on_id": pred} for pred, succ in sorted(edges)
                ],
                "conflicts": conflict_note,
                "preferences": prefs.context(),
                "validator_feedback": feedback or [],
            }
            user = json.dumps(context, ensure_ascii=False, indent=1)
            errors: list[str] = []
            for attempt in range(1, self.max_llm_attempts + 1):
                try:
                    raw = self.llm.complete_json(self.system_prompt, user, PLANNER_OUTPUT_SCHEMA)
                    return prefs.apply(
                        self._accept_llm_order(raw, by_id, edges, warnings, conflict_note, baseline)
                    )
                except (LLMError, ValidationError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"попытка {attempt}: {exc}")

            fallback_reason = "Модель не дала корректный ответ (" + "; ".join(errors) + ")"
        else:
            fallback_reason = "Модель не подключена"

        return prefs.apply(
            Plan(
                intent="plan_tasks",
                ordered_task_ids=rule_order,
                recommendations=[
                    Recommendation(
                        task_id=t,
                        reason=_rule_reason(by_id[t], edges, moments[t])
                        + (
                            "; предпочтительный тип задачи" if t in prefs.preferred_type_ids else ""
                        ),
                    )
                    for t in rule_order
                ],
                warnings=warnings,
                message_to_user=_order_message(rule_order, by_id, conflict_note),
                reasoning=[
                    fallback_reason,
                    "Порядок по правилам: зависимости, затем срок, затем приоритет",
                    *conflict_note,
                ],
                source="fallback",
            )
        )

    def _accept_llm_order(
        self,
        raw: dict[str, Any],
        by_id: dict[int, dict[str, Any]],
        edges: set[tuple[int, int]],
        warnings: list[str],
        conflict_note: list[str],
        baseline: list[int],
    ) -> Plan:
        """Принимает порядок от модели и приводит его к жёстким правилам."""
        proposed = raw["ordered_task_ids"]
        if not isinstance(proposed, list) or not proposed:
            raise ValueError("пустой ordered_task_ids")

        notes: list[str] = []
        invented = [t for t in proposed if t not in by_id]
        if invented:
            notes.append(f"Код убрал несуществующие ID из ответа модели: {invented}")
        seen: list[int] = []
        for task_id in proposed:
            if task_id in by_id and task_id not in seen:
                seen.append(task_id)
        if len(seen) < len([t for t in proposed if t in by_id]):
            notes.append("Код убрал повторы ID")
        # забытые моделью задачи встают в конец в порядке правил: срок, затем приоритет
        forgotten = [t for t in baseline if t not in seen]
        if forgotten:
            notes.append(f"Модель пропустила задачи {forgotten}, код добавил их в конец")
        preferred = seen + forgotten
        order = _ordered_respecting_dependencies(list(by_id), edges, preferred)
        if order != preferred:
            notes.append("Код переставил задачи, чтобы зависимости шли раньше зависящих от них")
        if not seen:
            raise ValueError("в ответе модели нет ни одного существующего ID")

        recommendations: list[Recommendation] = []
        for item in raw.get("recommendations") or []:
            rec = Recommendation.model_validate(item)
            if rec.task_id not in by_id or any(r.task_id == rec.task_id for r in recommendations):
                continue
            if by_id[rec.task_id]["priority"] is not None and rec.recommended_priority is not None:
                # Явно заданный приоритет пользователя не переопределяем (FR-13).
                rec.recommended_priority = None
            recommendations.append(rec)
        position = {task_id: index for index, task_id in enumerate(order)}
        recommendations.sort(key=lambda r: position[r.task_id])

        message = str(raw.get("message") or "").strip() or _order_message(
            order, by_id, conflict_note
        )
        return Plan(
            intent="plan_tasks",
            ordered_task_ids=order,
            recommendations=recommendations,
            warnings=warnings,
            message_to_user=message,
            reasoning=[*notes, *conflict_note] or ["Порядок модели принят без изменений"],
            source="llm",
        )


class _PreferenceFacts:
    """Подтверждённые предпочтения → факты для плана (P12).

    Как и даты, факты считает код: маленькая модель не сопоставит «созвоны» с «Созвон с
    командой» и не посчитает рабочие дни до срока. Предпочтения не спорят с зависимостями,
    сроками и явными приоритетами — они влияют на порядок только при прочих равных и
    объясняются пользователю отдельной строкой.
    """

    def __init__(
        self,
        preferences: dict[str, Any],
        by_id: dict[int, dict[str, Any]],
        moments: dict[int, datetime | None],
        now: datetime,
    ) -> None:
        self.preferences = preferences
        self.now = now
        self.moments = moments
        work_time = preferences.get("preferred_work_time")
        self.work_time = normalize_work_time(work_time) or work_time
        days = preferences.get("working_days")
        self.days = normalize_days(days) if days else None
        types = normalize_value("preferred_task_types", preferences.get("preferred_task_types"))
        self.type_stems = [_stem(word) for item in (types or []) for word in item.split()]
        self.type_stems = [stem for stem in self.type_stems if len(stem) >= 3]
        self.preferred_type_ids = {
            task_id for task_id, task in by_id.items() if self._matches_type(task["title"])
        }
        self.today = WEEKDAYS[now.weekday()]
        self.today_is_working = self.days is None or self.today in self.days

    def _matches_type(self, title: str) -> bool:
        words = re.findall(r"\w+", title.casefold().replace("ё", "е"))
        return any(word.startswith(stem) for stem in self.type_stems for word in words)

    def _working_days_left(self, moment: datetime | None) -> int | None:
        if self.days is None or moment is None:
            return None
        start, end = self.now.date(), moment.astimezone(self.now.tzinfo).date()
        count, day = 0, start
        while day <= end:
            if WEEKDAYS[day.weekday()] in self.days:
                count += 1
            day = date.fromordinal(day.toordinal() + 1)
        return count

    def task_facts(self, task_id: int) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if self.type_stems:
            facts["preferred_type"] = task_id in self.preferred_type_ids
        left = self._working_days_left(self.moments[task_id])
        if left is not None:
            facts["working_days_left"] = left
        return facts

    def context(self) -> dict[str, Any]:
        context = dict(self.preferences)
        if self.days is not None:
            context["today"] = self.today
            context["today_is_working_day"] = self.today_is_working
        return context

    def apply(self, plan: Plan) -> Plan:
        """Дописывает в план, какие предпочтения учтены и что из них следует."""
        if not self.preferences:
            return plan
        lines = [describe(kind, value) for kind, value in self.preferences.items()]
        notes = ["Учтены ваши предпочтения: " + "; ".join(lines) + "."]
        if self.preferred_type_ids:
            ids = ", ".join(f"#{task_id}" for task_id in sorted(self.preferred_type_ids))
            notes.append(f"Задачи предпочтительного типа: {ids} — при прочих равных они раньше.")
        if not self.today_is_working:
            plan.warnings.append("non_working_day")
            notes.append(
                f"Сегодня {self.today} — по вашим предпочтениям нерабочий день; "
                "порядок рассчитан на ближайший рабочий день."
            )
        if self.work_time:
            notes.append(f"Начать лучше {self.work_time}, как вам удобнее.")
        plan.message_to_user = "\n\n".join([plan.message_to_user, " ".join(notes)]).strip()
        plan.reasoning.append("Применены подтверждённые предпочтения (P12): " + "; ".join(lines))
        return plan


def _stem(word: str) -> str:
    """Грубая основа слова для сопоставления типов задач: «созвоны» ↔ «созвон»."""
    word = word.casefold().replace("ё", "е")
    return word[:5] if len(word) > 5 else word.rstrip("аяыиоеуюь")


def _days_left(moment: datetime | None, now: datetime) -> int | None:
    """Сколько календарных дней до срока: 0 — сегодня, отрицательное — просрочено."""
    if moment is None:
        return None
    return (moment.astimezone(now.tzinfo).date() - now.date()).days


def _rule_reason(task: dict[str, Any], edges: set[tuple[int, int]], moment: datetime | None) -> str:
    parts = []
    blocked_by = sorted(p for p, s in edges if s == task["id"])
    blocks = sorted(s for p, s in edges if p == task["id"])
    if blocked_by:
        parts.append("после задач " + ", ".join(f"#{i}" for i in blocked_by))
    if moment is not None:
        parts.append(f"срок {_deadline_text(task)}")
    if task["priority"]:
        parts.append(f"приоритет {task['priority']}")
    if blocks:
        parts.append("от неё зависят " + ", ".join(f"#{i}" for i in blocks))
    return "; ".join(parts) or "без срока и приоритета"


def _order_message(
    order: list[int], by_id: dict[int, dict[str, Any]], conflict_note: list[str]
) -> str:
    first = by_id[order[0]]
    message = f"Предлагаю начать с задачи #{first['id']} «{first['title']}»."
    if conflict_note:
        message += " Есть противоречия сроков и зависимостей — сроки я не менял."
    return message
