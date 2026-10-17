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
├── db/              SQLite: задачи, зависимости, предпочтения
├── observability/   логи, метрики, трейсы, алерты
└── config.py        настройки из переменных окружения ATM_*
tests/               pytest
benchmarks/          эталонные запросы (cases.jsonl) и скрипт бенчмарка LLM
requirements.md      ТЗ
```
