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
atm chat                       # интерактивный диалог
```

## Локальные модели

Для полного сценария установите модели, указанные в настройках проекта:

```bash
ollama pull tev1:0.8b-q8_0
ollama pull qwen3:1.7b
ollama pull gemma4:e2b
ollama pull ministral-3:3b
ollama list
```

Tev1 предварительно оценивает запрос, Qwen3 обрабатывает простое чтение задач,
Gemma4 работает аналитиком и планировщиком, Ministral3 проверяет планы.
Теги установленных моделей должны совпадать со значениями `ATM_*_MODEL` в `.env`.
Для Tev1 нужна Ollama версии 0.35 или новее.

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

Задачи, зависимости, операции подтверждения и ключи повторов `/chat` хранятся в SQLite.
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
Аналитик получает до четырёх предыдущих реплик. Ссылку «её/его» на задачу
разрешаем только при одном ID в последнем ответе; при нескольких ID система
просит уточнить номер.

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

`POST /api/v1/chat` — сообщение проходит всю цепочку агентов (см. «Workflow» ниже);
`conversation_id` из ответа продолжает тот же диалог. Для безопасного повтора
после сетевого сбоя передайте `request_id` (уникальную строку для сообщения).
Повтор с тем же ID и теми же данными вернёт сохранённый ответ; пока первый
запрос выполняется или его исход неизвестен, API вернёт 409 и не запустит
агентов второй раз. CLI создаёт `request_id` автоматически.

Проверка API: `pytest tests/test_api.py`.

## CLI

`atm chat` запускает диалог через FastAPI. Команда показывает ID нового диалога
после первого ответа; для продолжения после перезапуска используйте:

```bash
atm chat --conversation-id ID_ПРЕДЫДУЩЕГО_ДИАЛОГА
```

Введите `/exit`, чтобы выйти. Если агент подготовил действие, требующее
подтверждения, CLI показывает его описание и спрашивает согласие.
Ответ `y` отправляет запрос на подтверждение; `n` или Enter отменяет действие.
Если ответ API потерялся, введите `/retry`: CLI повторит тот же запрос с тем же
`request_id` или повторит подтверждение того же `action_id`. До разрешения
ошибки новый запрос не принимается, чтобы случайно не создать дубль.

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

## Подтверждение опасных операций (P09)

`src/ai_task_manager/tools/`:
- `executor.py` — исполнитель плана: операции плана одной транзакцией; удаление и завершение с незавершёнными зависимостями без подтверждения не выполняет, даже если план говорит иначе;
- `confirmation.py` — предварительный план («что изменится»), ожидающее действие `agent_plan` в таблице `operations`, подтверждение с повторной проверкой, узел графа `make_action_node`.

Подтверждение и отмена — те же маршруты `POST /api/v1/actions/{action_id}/confirm` и `/cancel`.

```bash
python -m ai_task_manager.tools.confirmation_demo   # сценарии подтверждения, без LLM
```

## Workflow (P10, A14)

`src/ai_task_manager/workflow/`:
- `graph.py` — граф LangGraph: `prepare → decision (Tev1) → route → analyst → planner → validator → act → finish → remember_reply`; при отказе валидатора план, построенный моделью, возвращается планировщику с замечаниями (не больше `MAX_VALIDATION_ATTEMPTS = 2` проверок);
- `nodes.py` — узлы аналитика, планировщика и итогового ответа;
- `service.py` — сборка графа из настоящих агентов (`create_workflow`) и обработка сообщения (`run_chat`); если граф собран с `SqliteSaver`, ход идёт через `run_turn()` и история диалога сохраняется (см. «Память диалога»).
- `decision.py` и `decision_client.py` — Tev1 через `/v1/systemone`: предварительная оценка типа запроса, маршрута, необходимости уточнения и попытки обойти подтверждение. Быстрый путь разрешён только для известных формулировок чтения; Tev1 может отправить их на полный разбор, но не может перевести неизвестный запрос с возможным изменением данных на путь чтения. При недоступной модели роутер использует прежние правила; уточнение решает аналитик, подтверждение проверяет код.

После `finish` идёт узел памяти `remember_reply`: в историю попадает итоговый ответ любой ветки.

```bash
python -m ai_task_manager.workflow.chat_demo   # полный диалог на настоящих моделях (нужна Ollama)
uvicorn ai_task_manager.api.app:app --reload   # затем Swagger: POST /api/v1/chat
```

## Промпты и скиллы (P11)

Системный промпт агента = `prompts/SOUL.md` (общие принципы системы) + файл роли `agents/prompts/<роль>.md` (кто агент: принципы, запреты, вход, выход) + скилл `skills/<имя>/SKILL.md` (как делать работу: процедура, типичные случаи, чек-лист). Сборка — `prompting.build_prompt`.

| Агент | Роль | Скилл |
|---|---|---|
| Аналитик | — | `skills/task-extraction` |
| Планировщик | `agents/prompts/planner.md` | `skills/task-planning` |
| Валидатор | `agents/prompts/validator.md` | `skills/task-validation` |
