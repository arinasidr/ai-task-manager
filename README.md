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

`POST /api/v1/chat` пока возвращает `503`: обработка сообщений будет подключена
в задачах A09–A10.

Проверка API: `pytest tests/test_api.py`.