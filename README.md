# ai-task-manager


```bash
git clone https://github.com/arinasidr/ai-task-manager.git
cd ai-task-manager
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
pytest                         # тесты
ruff check .                   # линтер
ruff format .                  # автоформатирование
uvicorn ai_task_manager.api.app:app --reload   # API: http://127.0.0.1:8000/docs
atm health                     # CLI: проверить, что API запущен
```

## Зависимости

Список лежит в `requirements.txt` (основные) и `requirements-dev.txt` (плюс pytest и ruff); `pyproject.toml` дублирует его для сборки пакета. При добавлении библиотеки правьте оба места.

Версии в файлах заданы нижними границами, поэтому у разных людей могут оказаться разные версии. Чтобы зафиксировать одинаковые, после рабочей установки: `pip freeze > requirements.lock`, закоммитить и ставить через `pip install -r requirements.lock`.

## Структура

```
src/ai_task_manager/
├── api/             FastAPI: маршруты /api/v1/*, OpenAPI
├── cli/             терминальный клиент (ходит только в API)
├── agents/          Аналитик, Планировщик, Валидатор
├── workflow/        граф LangGraph, общее состояние, подтверждения
├── tools/           Tools с строгими схемами (обычный Python)
├── memory/          рабочая, краткосрочная, долгосрочная память
├── db/              SQLite: задачи, зависимости, предпочтения
├── observability/   логи, метрики, трейсы, алерты
└── config.py        настройки из переменных окружения ATM_*
tests/               pytest
benchmarks/          эталонные запросы (cases.jsonl) и скрипт бенчмарка LLM
requirements.md      ТЗ
```

## Работа в команде

- `main` — только рабочий код; изменения идут через ветки (`feat/...`, `docs/...`) и Pull Request.
- Перед коммитом: `ruff format . && ruff check . && pytest`. Те же проверки запускает CI на каждый push и PR.
