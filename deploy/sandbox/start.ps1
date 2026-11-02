# Запуск API AI Task Manager внутри песочницы (P13). Окно остаётся занятым, пока работает
# сервер; остановка — Ctrl+C. API доступно с хоста: http://localhost:8000/docs
#
#   powershell -ExecutionPolicy Bypass -File deploy\sandbox\start.ps1

param([string]$Name = "atm")

# Docker Sandboxes при старте пересоздаёт список пользователей (файлы при этом сохраняются),
# поэтому atm восстанавливаем с тем же UID/GID, что у каталога данных. Без sudo, как в setup.
$ensure = 'id atm >/dev/null 2>&1 || { getent group atm >/dev/null || groupadd -g $(stat -c %g /var/lib/atm) atm; useradd -u $(stat -c %u /var/lib/atm) -g atm -d /home/atm -m -s /bin/bash atm; }'
& sbx exec -u root $Name bash -c $ensure
if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось восстановить пользователя atm — запустите setup.ps1 -Recreate" -ForegroundColor Red; exit 1 }

# sbx exec -u понимает только пользователей шаблона (root, agent), поэтому заходим root и сразу
# сбрасываем права до atm через runuser. Переменные с путями к сертификатам root убираем:
# файлы по ним atm прочитать не может.
$asAtm = 'cd /opt/atm && exec runuser -u atm -- env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE HOME=/home/atm'

# Пользователь atm без sudo, рабочая папка /opt/atm (оттуда читается .env с настройками
# песочницы). Сервер слушает 0.0.0.0, иначе проброшенный порт не дойдёт до него.
& sbx exec -u root $Name bash -c "$asAtm /opt/atm/.venv/bin/uvicorn ai_task_manager.api.app:app --host 0.0.0.0 --port 8000"
