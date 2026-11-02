# Проверка изоляции AI Task Manager в Docker Sandboxes (P13).
#
#   powershell -ExecutionPolicy Bypass -File deploy\sandbox\check.ps1
#
# 1) изнутри песочницы — deploy/sandbox/check_isolation.py от пользователя atm;
# 2) с хоста — API отвечает на проброшенном порту;
# 3) политика сети и журнал заблокированных запросов.
# Для полной проверки API должен быть запущен (start.ps1 в другом окне).

param([string]$Name = "atm", [int]$Port = 8000)

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }

Step "1. Проверки изнутри песочницы"
# Docker Sandboxes при старте пересоздаёт список пользователей (файлы при этом сохраняются),
# поэтому atm восстанавливаем с тем же UID/GID, что у каталога данных. Без sudo, как в setup.
$ensure = 'id atm >/dev/null 2>&1 || { getent group atm >/dev/null || groupadd -g $(stat -c %g /var/lib/atm) atm; useradd -u $(stat -c %u /var/lib/atm) -g atm -d /home/atm -m -s /bin/bash atm; }'
& sbx exec -u root $Name bash -c $ensure
if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось восстановить пользователя atm — запустите setup.ps1 -Recreate" -ForegroundColor Red; exit 1 }
# sbx exec -u понимает только пользователей шаблона (root, agent), поэтому заходим root и сразу
# сбрасываем права до atm через runuser. Переменные с путями к сертификатам root убираем:
# файлы по ним atm прочитать не может.
$asAtm = 'cd /opt/atm && exec runuser -u atm -- env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE HOME=/home/atm'
& sbx exec -u root $Name bash -c "$asAtm /opt/atm/.venv/bin/python deploy/sandbox/check_isolation.py"
$inside = $LASTEXITCODE

Step "2. API с хоста"
try {
    $health = Invoke-RestMethod -Uri "http://localhost:$Port/health" -TimeoutSec 5
    Write-Host "[OK  ] http://localhost:$Port/health -> $($health.status), версия $($health.version)"
    $api = 0
} catch {
    Write-Host "[FAIL] API не отвечает на http://localhost:$Port — запущен ли start.ps1?" -ForegroundColor Red
    $api = 1
}

Step "3. Политика сети песочницы"
& sbx policy ls

Step "4. Последние решения прокси (видно, что внешние адреса заблокированы)"
# фильтра по песочнице у log нет: смотрим строки с SANDBOX = atm
& sbx policy log --limit 20

if ($inside -eq 0 -and $api -eq 0) {
    Write-Host "`nИзоляция в порядке" -ForegroundColor Green
    exit 0
}
Write-Host "`nЕсть непройденные проверки — см. FAIL выше" -ForegroundColor Red
exit 1
