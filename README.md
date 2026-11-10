# AI Task Manager

Мультиагентная система для личных задач на русском языке. Пишешь обычным текстом
(«добавь задачу подготовить презентацию к пятнице», «что мне делать сначала?»), а система
сама создаёт задачи, меняет их, строит порядок выполнения и учитывает зависимости, сроки
и твои предпочтения. Все модели работают локально через Ollama.

## Как это устроено

Запрос проходит граф агентов на LangGraph:

1. **Tev1** быстро оценивает запрос, и роутер выбирает модель.
2. **Аналитик** разбирает, что хочет пользователь: задачи, сроки, приоритеты, ID.
3. **Планировщик** строит план действий и порядок выполнения задач.
4. **Валидатор** проверяет план кодом (ID, зависимости, даты), а изменения задач
   дополнительно проверяет модель другого семейства.
5. **Исполнитель** применяет план к SQLite одной транзакцией.

Удаление и завершение задачи с незавершёнными зависимостями без подтверждения
не выполняются: система показывает, что изменится, и ждёт ответа.

| Роль | Модель | Тег в Ollama |
|---|---|---|
| Предварительная оценка запроса | Tev1 | `tev1:0.8b-q8_0` |
| Простые запросы на чтение | Qwen3 1.7B | `qwen3:1.7b` |
| Аналитик и планировщик | Gemma4 E2B | `gemma4:e2b` |
| Валидатор | Ministral-3 3B | `ministral-3:3b` |

## Что нужно

- Python 3.11 или новее (мы работаем на 3.12)
- [Ollama](https://ollama.com) версии 0.35 или новее
- Git

## Установка

```bash
git clone https://github.com/arinasidr/ai-task-manager.git
cd ai-task-manager
python -m venv .venv
```

Активировать окружение:

```bash
.venv\Scripts\Activate.ps1      # Windows (PowerShell)
source .venv/bin/activate       # macOS / Linux
```

Если PowerShell не даёт запустить скрипт, один раз выполни
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Поставить зависимости и создать файл настроек:

```bash
pip install -r requirements.txt
copy .env.example .env          # macOS / Linux: cp .env.example .env
```

Скачать модели:

```bash
ollama pull tev1:0.8b-q8_0
ollama pull qwen3:1.7b
ollama pull gemma4:e2b
ollama pull ministral-3:3b
```

## Запуск

Ollama должна быть запущена (приложение Ollama или `ollama serve`).

Запустить API:

```bash
uvicorn ai_task_manager.api.app:app
```

Swagger откроется на http://127.0.0.1:8000/docs. База `data/tasks.db` создаётся сама при
первом запуске.

Во втором терминале (с тем же активированным `.venv`) открыть чат:

```bash
atm health      # проверить, что API отвечает
atm chat        # диалог с системой
```

```
Вы: добавь две задачи: собрать данные, потом написать отчёт к среде
Вы: что мне делать сначала?
Вы: удали задачу 2
```

Если система просит подтвердить действие, ответь `y` (выполнить) или `n` (отменить).
`/exit` закрывает чат. Продолжить старый диалог: `atm chat --conversation-id ID`.

## Настройки

Все настройки лежат в `.env` и начинаются с `ATM_`. Основные:

| Переменная | Что задаёт | Значение в `.env.example` |
|---|---|---|
| `ATM_DATABASE_PATH` | путь к базе SQLite | `data/tasks.db` |
| `ATM_TIMEZONE` | часовой пояс для сроков | `Europe/Moscow` |
| `ATM_OLLAMA_BASE_URL` | адрес Ollama | `http://localhost:11434` |
| `ATM_ANALYST_MODEL`, `ATM_PLANNER_MODEL` | аналитик и планировщик | `gemma4:e2b` |
| `ATM_FAST_MODEL` | быстрые запросы на чтение | `qwen3:1.7b` |
| `ATM_DECISION_MODEL` | Tev1 | `tev1:0.8b-q8_0` |
| `ATM_VALIDATOR_MODEL` | валидатор | `ministral-3:3b` |
| `ATM_LOG_DIR` | папка для логов и траекторий | `data/logs` |
| `ATM_PHOENIX_ENDPOINT` | куда отправлять трейсы | `http://localhost:6006/v1/traces` |

Теги моделей в `.env` должны совпадать с тем, что показывает `ollama list`. Если убрать
`ATM_LOG_DIR` или `ATM_PHOENIX_ENDPOINT`, логи или трейсы просто выключатся.

## Трейсы в Phoenix (по желанию)

Phoenix ставится в отдельное окружение, чтобы его зависимости не попали в проект:

```bash
python -m venv .venv-phoenix
.venv-phoenix\Scripts\pip install arize-phoenix     # macOS / Linux: .venv-phoenix/bin/pip
.venv-phoenix\Scripts\phoenix serve                 # macOS / Linux: .venv-phoenix/bin/phoenix
```

Трейсы смотреть на http://localhost:6006 (проект `ai-task-manager`). Если Phoenix не
запущен, приложение работает как обычно, трейсы просто не сохраняются.

## Запуск в песочнице

На Windows приложение можно запустить внутри microVM Docker Sandboxes: без доступа
к файлам хоста и с закрытой сетью (открыта только Ollama). Установка и проверка
описаны в [deploy/sandbox/README.md](deploy/sandbox/README.md), коротко:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\sandbox\setup.ps1
powershell -ExecutionPolicy Bypass -File deploy\sandbox\start.ps1
powershell -ExecutionPolicy Bypass -File deploy\sandbox\check.ps1
```

## Тесты

То же самое запускается в GitHub Actions на каждый pull request:

```bash
pytest
ruff check .
ruff format --check .
python -m perf.run --mode mock --check
```

## Эксперименты

| Папка | Что там |
|---|---|
| [benchmarks/](benchmarks/README.md) | сравнение локальных моделей и движков на 30 сценариях |
| [evals/](evals/README.md) | оценка всей системы, негативные сценарии, LLM-as-a-judge |
| [perf/](perf/run.py) | задержки, скорость моделей, CPU и память, роутинг против одной модели |

Ручные проверки на настоящих моделях лежат рядом с кодом, например
`python -m ai_task_manager.workflow.security_demo` (атаки на агента) или
`python -m ai_task_manager.workflow.chat_demo` (полный диалог на временной базе).

## Структура

```
src/ai_task_manager/
├── api/             HTTP API на FastAPI
├── cli/             команда atm (работает через API)
├── agents/          аналитик, планировщик, валидатор
├── workflow/        граф LangGraph, роутер, Tev1
├── tools/           исполнение плана и подтверждения
├── memory/          память диалога и поиск предпочтений
├── db/              SQLite: задачи, зависимости, операции, предпочтения
├── observability/   логи, метрики, трейсы
├── prompts/, skills/  общие правила агентов и их Skills
└── config.py        настройки ATM_*
tests/               тесты pytest
benchmarks/, evals/, perf/   эксперименты и их результаты
deploy/sandbox/      скрипты песочницы
requirements.md      техническое задание
```

## Авторы

Павел Чеботарев и Арина Сидорина, группа 11-411.
