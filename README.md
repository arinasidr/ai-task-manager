# ai-task-manager


```bash
git clone https://github.com/arinasidr/ai-task-manager.git
cd ai-task-manager
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
pytest                         # тесты
ruff check .                   # линтер
ruff format .                  # автоформатирование
uvicorn ai_task_manager.api.app:app --reload   # API: http://127.0.0.1:8000/docs
atm health                     # CLI: проверить, что API запущен
```

## Структура

```
src/ai_task_manager/
├── api/             FastAPI: маршруты /api/v1/*, OpenAPI
├── cli/             терминальный клиент (ходит только в API)
├── agents/          Аналитик, Планировщик, Валидатор
├── workflow/        граф LangGraph, общее состояние, подтверждения
├── tools/           Tools с строгими схемами (обычный Python)
├── memory/          рабочая, краткосрочная, долгосрочная память
├── db/              SQLite: задачи, зависимости
├── observability/   логи, метрики, трейсы, алерты
└── config.py        настройки из переменных окружения ATM_*
tests/               pytest
benchmarks/          эталонные запросы (cases.jsonl) и скрипт бенчмарка LLM
data/                SQLite база данных
requirements.md      ТЗ
```

## База данных

Задачи, зависимости и операции подтверждения хранятся в SQLite.
Путь по умолчанию — `data/tasks.db`; его можно изменить через
`ATM_DATABASE_PATH`. Файл БД не добавляется в Git.

Инициализация таблиц:

```bash
python -c "from ai_task_manager.db.database import init_db; init_db()"
```

Проверка:

```bash
pytest tests/test_db.py
```

## Память диалога

`WorkflowState` передаёт между агентами разбор запроса, план и вердикт.
Перед новым запросом эти промежуточные результаты сбрасываются. В текущем
состоянии хранятся последние шесть реплик. Ответ добавляется в историю,
когда граф заполнил `reply` или когда `run_turn()` получил фактический
ответ через `build_reply`.

`SqliteSaver` сохраняет состояние графа в SQLite по ключу
`thread_id = conversation_id`. `run_turn()` создаёт ID нового диалога
или продолжает существующий. Одновременные ходы одного диалога
выполняются по очереди в пределах одного процесса. Соединение
`open_checkpointer()` должно оставаться открытым во время работы графа.

При открытии Checkpointer удаляются диалоги, чей последний чекпоинт
старше семи дней. При вызове `run_turn()` срок проверяется только для
запрошенного диалога. Лимит шести реплик относится к текущему состоянию:
старые чекпоинты активного диалога остаются в SQLite до удаления диалога.
Задачи и подтверждённые предпочтения читаются из актуальных таблиц SQLite.

## HTTP API

После запуска приложения документация маршрутов доступна по адресу
`http://127.0.0.1:8000/docs`. Таблицы SQLite создаются при запуске API.

`GET` и `POST /api/v1/tasks` показывают и создают задачи.
`GET` и `PATCH /api/v1/tasks/{id}` читают и изменяют задачу;
`PATCH /api/v1/tasks/{id}/status` меняет её статус.

`POST /api/v1/tasks/{id}/delete-request` возвращает `action_id`, не удаляя задачу.
Действие подтверждается через `POST /api/v1/actions/{action_id}/confirm`
или отменяется через `POST /api/v1/actions/{action_id}/cancel`.
Подтверждение также требуется для завершения задачи с незавершёнными зависимостями.

`POST /api/v1/chat` пока возвращает `503`:

Проверка API: `pytest tests/test_api.py`.

## Агенты

`src/ai_task_manager/agents/`:
- `analyst/` — агент-аналитик (A09): разбор запроса в `AnalystResult`;
- `contracts.py` — контракты между агентами: `Analysis` (вход планировщика, принимает `AnalystResult`) и `Plan` (выход планировщика);
- `planner.py` — агент-планировщик: действия с задачами планирует код, порядок выполнения (`plan_tasks`) — LLM с проверкой кодом;
- `prompts/planner.md` — системный промпт планировщика;
- `validator.py` — агент-валидатор (P08): проверки кодом + смысловая проверка LLM другого семейства (`ministral-3:3b`); узел графа — `make_validator_node`;
- `prompts/validator.md` — системный промпт валидатора. Промпты всех агентов начинаются с общего `prompts/SOUL.md` (`prompting.with_soul`);
- `llm.py` — вызов модели Ollama со структурированным ответом (JSON Schema).

Проверить планировщик на настоящей модели (нужна запущенная Ollama):

```bash
python -m ai_task_manager.agents.planner_demo                  # модель из ATM_PLANNER_MODEL
python -m ai_task_manager.agents.planner_demo --model qwen3:1.7b
python -m ai_task_manager.agents.validator_demo                 # модель из ATM_VALIDATOR_MODEL
```
