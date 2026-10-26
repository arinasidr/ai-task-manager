"""Агент-валидатор: независимая проверка плана до того, как он изменит данные.

Проверка в два слоя:
1. **Код** — формальные правила, которые нельзя доверять LLM: план не выходит за рамки
   намерения, касается только существующих и названных пользователем задач, не создаёт циклов,
   опасные операции идут через подтверждение, порядок выполнения соблюдает зависимости. Если код
   нашёл нарушение, модель не вызывается.
2. **LLM другого семейства** (по умолчанию `ministral-3:3b`, Mistral, а аналитик и планировщик —
   Gemma) — смысл: делает ли план то, что просил пользователь, не выдуманы ли данные, правильно ли
   посчитаны сроки, правдиво ли сообщение пользователю.

Если модель не ответила, план, который меняет данные, отклоняется: лучше переспросить, чем
записать непроверенное (fail-closed). Планы только на чтение принимаются по результатам кода.
"""

import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

from ai_task_manager.agents.contracts import Analysis, Plan
from ai_task_manager.agents.llm import JsonLLM, LLMError
from ai_task_manager.agents.planner import (
    _creates_cycle,
    _days_left,
    _deadline_text,
    deadline_fields,
    deadline_moment,
)
from ai_task_manager.db.crud_func import get_dependencies, get_task, list_tasks
from ai_task_manager.prompting import with_soul

PROMPT_PATH = Path(__file__).parent / "prompts" / "validator.md"

IssueCode = Literal[
    "intent_mismatch",
    "wrong_task",
    "invented_data",
    "missing_data",
    "wrong_deadline",
    "misleading_message",
    "bad_order",
    "missing_confirmation",
    "unknown_task",
    "dependency_cycle",
    "other",
]

ISSUE_CODES: tuple[str, ...] = IssueCode.__args__  # type: ignore[attr-defined]
_LLM_ISSUE_CODES = [
    "intent_mismatch",
    "wrong_task",
    "invented_data",
    "missing_data",
    "wrong_deadline",
    "misleading_message",
    "bad_order",
    "other",
]

VALIDATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": _LLM_ISSUE_CODES},
                    "detail": {"type": "string"},
                },
                "required": ["code", "detail"],
            },
        },
    },
    "required": ["decision", "issues"],
}

# Какими инструментами может пользоваться план для каждого намерения.
ALLOWED_TOOLS: dict[str, frozenset[str]] = {
    "create_task": frozenset({"create_task", "add_dependency"}),
    "update_task": frozenset(
        {"update_task", "change_task_status", "add_dependency", "remove_dependency"}
    ),
    "delete_task": frozenset({"delete_task"}),
    "delete_tasks": frozenset({"delete_task"}),
    "get_tasks": frozenset({"list_tasks"}),
    "plan_tasks": frozenset(),
    "update_preference": frozenset({"update_preference"}),
    "clarification_required": frozenset(),
    "unsupported": frozenset(),
}

_ID_ARGS = ("task_id", "predecessor_task_id", "successor_task_id")

# Семейство модели по началу имени. Валидатор обязан быть из другого семейства (ТЗ, FR-19).
_FAMILIES = (
    ("gemma", "Google Gemma"),
    ("qwen", "Alibaba Qwen"),
    ("ministral", "Mistral"),
    ("mistral", "Mistral"),
    ("phi", "Microsoft Phi"),
    ("llama", "Meta Llama"),
    ("lfm", "Liquid LFM"),
    ("granite", "IBM Granite"),
    ("deepseek", "DeepSeek"),
)


def model_family(model: str) -> str:
    name = model.casefold().rsplit("/", 1)[-1]
    for prefix, family in _FAMILIES:
        if name.startswith(prefix):
            return family
    return name.split(":", 1)[0]


def check_validator_family(validator_model: str, *checked_models: str | None) -> None:
    """Падает, если валидатор из того же семейства, что и проверяемые им модели."""
    family = model_family(validator_model)
    same = [m for m in checked_models if m and model_family(m) == family]
    if same:
        raise ValueError(
            f"Валидатор {validator_model} из того же семейства ({family}), что и {same}: "
            "нужна модель другого семейства"
        )


def load_system_prompt() -> str:
    return with_soul(PROMPT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- вердикт


class Issue(BaseModel):
    code: IssueCode
    detail: str


class Verdict(BaseModel):
    decision: Literal["accept", "reject"]
    issues: list[Issue] = Field(default_factory=list)
    # code — решение принял код (нарушение правил или нечего проверять моделью);
    # llm — код не нашёл нарушений, решение модели; unavailable — модель не ответила.
    source: Literal["code", "llm", "unavailable"] = "code"
    model: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    @property
    def reasons(self) -> list[str]:
        """Замечания для планировщика при повторной попытке."""
        return [issue.detail for issue in self.issues]


def _reject(*issues: tuple[str, str], source: str = "code", **kw: Any) -> Verdict:
    return Verdict(
        decision="reject",
        issues=[Issue(code=code, detail=detail) for code, detail in issues],
        source=source,
        **kw,
    )


_WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")


def calendar_hint(now: datetime, days: int = 8) -> list[str]:
    """«сегодня: 2026-10-26, понедельник», «завтра: …» и следующие дни — для сверки сроков."""
    names = {0: "сегодня", 1: "завтра", 2: "послезавтра"}
    lines = []
    for offset in range(days):
        day = (now + timedelta(days=offset)).date()
        label = names.get(offset, f"через {offset} дн.")
        lines.append(f"{label}: {day.isoformat()}, {_WEEKDAYS[day.weekday()]}")
    return lines


_DUE_FIELDS = ("due_date", "due_time", "due_timezone", "due_at_utc")

_RELATIVE_WORDS = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
_NUMBER_WORDS = {
    "один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}  # fmt: skip
_IN_DAYS = re.compile(r"\bчерез\s+(\d+|[а-яё]+)\s+(?:день|дня|дней)\b", re.IGNORECASE)
_WORD = re.compile(r"\b(послезавтра|завтра|сегодня)\b", re.IGNORECASE)


def mentioned_dates(request: str, now: datetime) -> dict[str, date]:
    """Даты, которые код может точно вычислить из запроса: «сегодня», «завтра», «через 3 дня».

    Дни недели («в пятницу») и прочие формулировки не распознаются — их проверяет модель.
    """
    found: dict[str, date] = {}
    today = now.date()
    for match in _WORD.finditer(request):
        word = match.group(1).casefold()
        found[word] = today + timedelta(days=_RELATIVE_WORDS[word])
    for match in _IN_DAYS.finditer(request):
        raw = match.group(1).casefold()
        days = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
        if days is not None:
            found[match.group(0).casefold()] = today + timedelta(days=days)
    return found


def _plan_dates(plan: Plan) -> list[date]:
    dates = []
    for op in plan.operations:
        sources = [op.args]
        if isinstance(op.args.get("updates"), dict):
            sources.append(op.args["updates"])
        for source in sources:
            if source.get("due_date"):
                dates.append(date.fromisoformat(source["due_date"]))
    return dates


def check_relative_dates(
    request: str, plan: Plan, now: datetime
) -> tuple[list[tuple[str, str]], list[str]]:
    """Сверяет сроки плана с «сегодня/завтра/через N дней» из запроса.

    Возвращает (нарушения, подтверждения). Если в запросе таких слов нет или в плане нет сроков,
    оба списка пустые — даты проверяет модель.
    """
    mentioned = mentioned_dates(request, now)
    planned = _plan_dates(plan)
    if not mentioned or not planned:
        return [], []
    allowed = set(mentioned.values())
    described = ", ".join(f"«{word}» = {day.isoformat()}" for word, day in mentioned.items())
    wrong = [d for d in planned if d not in allowed]
    if wrong:
        return [
            (
                "wrong_deadline",
                f"В запросе {described}, а в плане срок " + ", ".join(d.isoformat() for d in wrong),
            )
        ], []
    return [], [f"Даты проверены кодом: {described} — совпадает с планом. Даты не проверяй."]


def _collapse_deadline(args: dict[str, Any]) -> dict[str, Any]:
    """Четыре технических поля срока → одно понятное `deadline` (для модели)."""
    if not set(_DUE_FIELDS) & set(args):
        return dict(args)
    view = {key: value for key, value in args.items() if key not in _DUE_FIELDS}
    view["deadline"] = _deadline_text(args)  # «2026-10-27» или «2026-10-27 18:30 (Europe/Moscow)»
    return view


def plan_view(plan: Plan) -> dict[str, Any]:
    """План в виде, удобном для проверки моделью: без пустых и технических полей."""
    operations = []
    for op in plan.operations:
        args = _collapse_deadline(op.args)
        if isinstance(args.get("updates"), dict):
            args["updates"] = _collapse_deadline(args["updates"])
        item: dict[str, Any] = {"tool": op.tool, "args": args}
        if op.ref:
            item["ref"] = op.ref
        operations.append(item)
    view: dict[str, Any] = {
        "intent": plan.intent,
        "operations": operations,
        "requires_confirmation": plan.requires_confirmation,
        "message_to_user": plan.message_to_user,
    }
    if plan.intent == "plan_tasks":
        view["ordered_task_ids"] = plan.ordered_task_ids
        view["recommendations"] = [r.model_dump() for r in plan.recommendations]
    if plan.warnings:
        view["warnings"] = plan.warnings
    return view


# ---------------------------------------------------------------- валидатор


class Validator:
    def __init__(
        self,
        llm: JsonLLM | None = None,
        *,
        model: str | None = None,
        timezone: str = "Europe/Moscow",
        max_llm_attempts: int = 2,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.tz = ZoneInfo(timezone)
        self.max_llm_attempts = max_llm_attempts
        self.system_prompt = system_prompt or load_system_prompt()

    def validate(
        self,
        request: str,
        analysis: Analysis | dict[str, Any] | None,
        plan: Plan | dict[str, Any] | None,
        connection: sqlite3.Connection,
        *,
        now: datetime | None = None,
    ) -> Verdict:
        if plan is None:
            return _reject(("other", "Планировщик не вернул план"))
        plan = plan if isinstance(plan, Plan) else Plan.model_validate(plan)
        analysis_obj = None
        if analysis is not None:
            analysis_obj = (
                analysis if isinstance(analysis, Analysis) else Analysis.model_validate(analysis)
            )
        now = (now or datetime.now(UTC)).astimezone(self.tz)

        issues = self.hard_checks(plan, analysis_obj, connection)
        date_issues, dates_verified = check_relative_dates(request, plan, now)
        issues += date_issues
        if issues:
            return _reject(*issues)

        if not self._needs_llm(plan):
            return Verdict(
                decision="accept",
                notes=["План ничего не меняет, проверки кода достаточно"],
            )

        if self.llm is None:
            return self._unavailable(plan, "Модель валидатора не подключена")

        context = self._llm_context(request, analysis_obj, plan, connection, now)
        if dates_verified:
            context["verified_by_code"] = dates_verified
        user = json.dumps(context, ensure_ascii=False, indent=1)
        errors: list[str] = []
        for attempt in range(1, self.max_llm_attempts + 1):
            try:
                raw = self.llm.complete_json(self.system_prompt, user, VALIDATOR_OUTPUT_SCHEMA)
                return self._accept_llm_answer(raw, dates_verified=bool(dates_verified))
            except (LLMError, ValidationError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"попытка {attempt}: {exc}")
        return self._unavailable(
            plan, "Модель не дала корректный ответ (" + "; ".join(errors) + ")"
        )

    # ---------------------------------------------------------- слой 1: код

    def hard_checks(
        self, plan: Plan, analysis: Analysis | None, connection: sqlite3.Connection
    ) -> list[tuple[str, str]]:
        issues: list[tuple[str, str]] = []
        tools = [op.tool for op in plan.operations]

        if analysis is not None and plan.intent != analysis.intent:
            planned_clarification = plan.needs_clarification and not plan.writes
            if not planned_clarification:
                issues.append(
                    (
                        "intent_mismatch",
                        f"Аналитик понял запрос как {analysis.intent}, а план делает {plan.intent}",
                    )
                )

        allowed = ALLOWED_TOOLS.get(plan.intent, frozenset())
        extra = sorted(set(tools) - allowed)
        if extra:
            issues.append(
                ("intent_mismatch", f"Для {plan.intent} недопустимы операции: {', '.join(extra)}")
            )

        if plan.needs_clarification and plan.writes:
            issues.append(("other", "План одновременно просит уточнение и меняет данные"))

        # ссылки на задачи: существующие ID или $ссылки на задачи, созданные раньше в этом плане
        defined_refs: set[str] = set()
        touched: set[int] = set()
        for op in plan.operations:
            for key in _ID_ARGS:
                if key not in op.args:
                    continue
                value = op.args[key]
                if isinstance(value, str) and value.startswith("$"):
                    if value[1:] not in defined_refs:
                        issues.append(("unknown_task", f"Ссылка {value} на несозданную задачу"))
                elif isinstance(value, int) and not isinstance(value, bool):
                    touched.add(value)
                    if get_task(connection, value) is None:
                        issues.append(("unknown_task", f"Задачи #{value} нет"))
                else:
                    issues.append(("other", f"Некорректный ID в {op.tool}: {value!r}"))
            if op.tool == "create_task":
                if not str(op.args.get("title", "")).strip():
                    issues.append(("missing_data", "У создаваемой задачи нет названия"))
                if op.ref:
                    defined_refs.add(op.ref)

        # план трогает только задачи, которые назвал пользователь (через аналитика)
        if analysis is not None and plan.intent in {"update_task", "delete_task", "delete_tasks"}:
            named = set(analysis.task_ids)
            if analysis.task_id is not None:
                named.add(analysis.task_id)
            for key in ("add_dependency_ids", "remove_dependency_ids"):
                named.update(analysis.updates.get(key) or [])
            foreign = sorted(touched - named)
            if foreign:
                issues.append(
                    ("wrong_task", f"План затрагивает задачи, которых нет в запросе: {foreign}")
                )

        if analysis is not None and plan.intent == "create_task" and not plan.needs_clarification:
            created = tools.count("create_task")
            if created != len(analysis.tasks):
                issues.append(
                    (
                        "missing_data" if created < len(analysis.tasks) else "invented_data",
                        f"В запросе задач: {len(analysis.tasks)}, в плане создаётся: {created}",
                    )
                )

        for op in plan.operations:
            if op.tool == "add_dependency":
                pred, succ = op.args.get("predecessor_task_id"), op.args.get("successor_task_id")
                if isinstance(pred, int) and isinstance(succ, int):
                    if _creates_cycle(connection, pred, succ):
                        issues.append(("dependency_cycle", f"Связь #{pred} → #{succ} даст цикл"))

        if "delete_task" in tools and not plan.requires_confirmation:
            issues.append(("missing_confirmation", "Удаление без подтверждения пользователя"))

        for op in plan.operations:
            if op.tool == "change_task_status" and op.args.get("status") == "done":
                task_id = op.args.get("task_id")
                if isinstance(task_id, int) and get_task(connection, task_id) is not None:
                    unfinished = [
                        p
                        for p in get_dependencies(connection, task_id)["predecessors"]
                        if get_task(connection, p)["status"] != "done"
                    ]
                    if unfinished and not plan.requires_confirmation:
                        issues.append(
                            (
                                "missing_confirmation",
                                f"Задача #{task_id} зависит от незавершённых {unfinished}: "
                                "нужно подтверждение",
                            )
                        )

        if plan.intent == "plan_tasks":
            issues.extend(self._check_order(plan, connection))

        if analysis is not None:
            issues.extend(self._check_deadlines(plan, analysis))

        # одинаковые замечания не дублируем
        return list(dict.fromkeys(issues))

    def _check_deadlines(self, plan: Plan, analysis: Analysis) -> list[tuple[str, str]]:
        """Срок в плане должен совпадать со сроком из разбора аналитика.

        Правильно ли аналитик понял «завтра» относительно запроса, проверяет модель; код
        проверяет, что по дороге к плану срок не изменился.
        """
        expected: list[tuple[str, str | None]] = []
        actual: list[dict[str, Any]] = []
        if plan.intent == "create_task":
            creates = [op for op in plan.operations if op.tool == "create_task"]
            for task, op in zip(analysis.tasks, creates, strict=False):
                expected.append((task.title, task.deadline))
                actual.append(op.args)
        elif plan.intent == "update_task" and "deadline" in analysis.updates:
            updates = [op for op in plan.operations if op.tool == "update_task"]
            if not updates or not set(_DUE_FIELDS) & set(updates[0].args.get("updates", {})):
                return [("missing_data", "В запросе есть новый срок, а план его не меняет")]
            expected.append((f"#{analysis.task_id}", analysis.updates["deadline"]))
            actual.append(updates[0].args["updates"])

        issues: list[tuple[str, str]] = []
        for (name, deadline), args in zip(expected, actual, strict=True):
            try:
                want = deadline_fields(deadline, self.tz)
            except ValueError:
                continue  # некорректную дату обрабатывает планировщик (уточнение)
            got = {key: args.get(key) for key in _DUE_FIELDS}
            if got != want:
                issues.append(
                    (
                        "wrong_deadline",
                        f"Срок {name} в плане ({_deadline_text(got) or 'нет'}) не совпадает "
                        f"с разбором запроса ({deadline or 'нет'})",
                    )
                )
        return issues

    def _check_order(self, plan: Plan, connection: sqlite3.Connection) -> list[tuple[str, str]]:
        open_ids = [t["id"] for t in list_tasks(connection) if t["status"] != "done"]
        order = plan.ordered_task_ids
        issues: list[tuple[str, str]] = []
        if sorted(order) != sorted(open_ids) or len(set(order)) != len(order):
            issues.append(
                (
                    "bad_order",
                    f"В порядке должны быть все незавершённые задачи {sorted(open_ids)} "
                    f"по одному разу, а получено {order}",
                )
            )
            return issues
        position = {task_id: index for index, task_id in enumerate(order)}
        for task_id in order:
            for predecessor in get_dependencies(connection, task_id)["predecessors"]:
                if predecessor in position and position[predecessor] > position[task_id]:
                    issues.append(
                        ("bad_order", f"Задача #{predecessor} должна идти раньше #{task_id}")
                    )
        return issues

    # ---------------------------------------------------------- слой 2: LLM

    @staticmethod
    def _needs_llm(plan: Plan) -> bool:
        """Модель нужна, если план меняет данные или даёт рекомендации по порядку."""
        if plan.needs_clarification:
            return False
        return plan.writes or plan.intent == "plan_tasks"

    def _llm_context(
        self,
        request: str,
        analysis: Analysis | None,
        plan: Plan,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> dict[str, Any]:
        if plan.intent == "plan_tasks":
            rows = [t for t in list_tasks(connection) if t["status"] != "done"]
        else:
            ids = {
                v
                for op in plan.operations
                for k, v in op.args.items()
                if k in _ID_ARGS and isinstance(v, int)
            }
            rows = [task for task_id in sorted(ids) if (task := get_task(connection, task_id))]

        tasks = []
        for row in rows:
            links = get_dependencies(connection, row["id"])
            tasks.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "status": row["status"],
                    "priority": row["priority"],
                    "deadline": _deadline_text(row),
                    "days_left": _days_left(deadline_moment(row, self.tz), now),
                    "blocked_by": links["predecessors"],
                    "blocks": links["successors"],
                }
            )

        return {
            "request": request,
            "now": now.isoformat(timespec="minutes"),
            # Готовый календарь: маленькие модели плохо считают даты и дни недели сами.
            "calendar": calendar_hint(now),
            "analysis": analysis.model_dump(exclude_defaults=True) if analysis else None,
            "plan": plan_view(plan),
            "tasks": tasks,
        }

    def _accept_llm_answer(self, raw: dict[str, Any], *, dates_verified: bool = False) -> Verdict:
        decision = raw["decision"]
        if decision not in ("accept", "reject"):
            raise ValueError(f"неизвестное решение {decision!r}")
        issues = [Issue.model_validate(item) for item in raw.get("issues") or []]
        notes: list[str] = []
        if decision == "reject" and not issues:
            # Отказ без причины нельзя передать планировщику — считаем ответ некорректным.
            raise ValueError("отказ без указания проблем")
        if dates_verified:
            # Даты уже точно проверил код; замечания модели о сроках — её ошибка счёта.
            overruled = [i for i in issues if i.code == "wrong_deadline"]
            if overruled:
                issues = [i for i in issues if i.code != "wrong_deadline"]
                notes.append(
                    "Замечание модели о сроке отклонено: даты проверены кодом — "
                    + "; ".join(i.detail for i in overruled)
                )
                if decision == "reject" and not issues:
                    decision = "accept"
        if decision == "accept" and issues:
            notes.append("Модель приняла план, но перечислила замечания — они не блокируют")
        return Verdict(
            decision=decision,
            issues=issues if decision == "reject" else [],
            source="llm",
            model=self.model,
            notes=notes + [f"{i.code}: {i.detail}" for i in issues if decision == "accept"],
        )

    def _unavailable(self, plan: Plan, why: str) -> Verdict:
        if plan.writes:
            return _reject(
                ("other", "Не удалось проверить план, данные не изменены. Повторите запрос."),
                source="unavailable",
                model=self.model,
                notes=[why],
            )
        return Verdict(
            decision="accept",
            source="unavailable",
            model=self.model,
            notes=[why, "План ничего не меняет: принят по проверкам кода"],
        )


# ---------------------------------------------------------------- узел графа


def make_validator_node(
    *,
    llm_for: Callable[[str], JsonLLM | None],
    connect: Callable[[], sqlite3.Connection],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Узел LangGraph: читает request, analysis и plan из State, пишет verdict.

    `llm_for(model)` возвращает клиента нужной модели (её выбрал роутер, `validator_model`),
    `connect()` открывает БД только для чтения задач.
    """

    def validator_node(state: dict[str, Any]) -> dict[str, Any]:
        model = state["validator_model"]
        check_validator_family(model, state.get("analyst_model"), state.get("planner_model"))
        validator = Validator(llm_for(model), model=model, timezone=state["timezone"])
        with closing(connect()) as connection:
            verdict = validator.validate(
                state["request"],
                state.get("analysis"),
                state.get("plan"),
                connection,
                now=datetime.fromisoformat(state["current_datetime"]),
            )
        return {
            "verdict": verdict.model_dump(),
            "validator_feedback": [] if verdict.accepted else verdict.reasons,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    return validator_node
