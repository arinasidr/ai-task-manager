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
& sbx exec -u atm -w /opt/atm $Name /opt/atm/.venv/bin/python deploy/sandbox/check_isolation.py
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
& sbx policy ls $Name

Step "4. Последние решения прокси (видно, что внешние адреса заблокированы)"
& sbx policy log $Name --limit 20

if ($inside -eq 0 -and $api -eq 0) {
    Write-Host "`nИзоляция в порядке" -ForegroundColor Green
    exit 0
}
Write-Host "`nЕсть непройденные проверки — см. FAIL выше" -ForegroundColor Red
exit 1
