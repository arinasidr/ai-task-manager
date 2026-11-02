# Запуск API AI Task Manager внутри песочницы (P13). Окно остаётся занятым, пока работает
# сервер; остановка — Ctrl+C. API доступно с хоста: http://localhost:8000/docs
#
#   powershell -ExecutionPolicy Bypass -File deploy\sandbox\start.ps1

param([string]$Name = "atm")

# Пользователь atm без sudo, рабочая папка /opt/atm (оттуда читается .env с настройками
# песочницы). Сервер слушает 0.0.0.0, иначе проброшенный порт не дойдёт до него.
& sbx exec -u atm -w /opt/atm $Name /opt/atm/.venv/bin/uvicorn ai_task_manager.api.app:app --host 0.0.0.0 --port 8000
