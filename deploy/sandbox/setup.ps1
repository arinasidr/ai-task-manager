# Развёртывание AI Task Manager в Docker Sandboxes на Windows (P13).
#
#   powershell -ExecutionPolicy Bypass -File deploy\sandbox\setup.ps1            # первый раз
#   powershell -ExecutionPolicy Bypass -File deploy\sandbox\setup.ps1 -Recreate  # пересоздать
#
# Перед запуском (один раз, см. deploy/sandbox/README.md): Windows Hypervisor Platform,
# winget install -h Docker.sbx, sbx login. Ollama запущена на хосте.
#
# Файл сохранён в UTF-8 с BOM: иначе Windows PowerShell 5.1 прочитает русский текст неверно.
#
# Код попадает в песочницу из ПОСЛЕДНЕГО КОММИТА (git archive): незакоммиченные правки,
# .env, .venv и data/ внутрь не копируются.

param(
    [string]$Name = "atm",
    [int]$Cpus = 2,
    [string]$Memory = "2g",
    [int]$Port = 8000,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
# Сайты, нужные только на время установки зависимостей. После установки правила удаляются.
$TemporaryHosts = @("pypi.org", "files.pythonhosted.org", "archive.ubuntu.com", "security.ubuntu.com")

function Invoke-Sbx {
    Write-Host "> sbx $($args -join ' ')" -ForegroundColor DarkGray
    & sbx @args
    if ($LASTEXITCODE -ne 0) { throw "Команда 'sbx $($args -join ' ')' завершилась с кодом $LASTEXITCODE" }
}

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }

Step "0. Проверки на хосте"
if (-not (Get-Command sbx -ErrorAction SilentlyContinue)) { throw "sbx не найден: winget install -h Docker.sbx" }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git не найден" }
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "Ollama на хосте работает, моделей: $($tags.models.Count)"
} catch {
    throw "Ollama на хосте не отвечает на http://localhost:11434 — запустите её и повторите"
}
$dirty = git -C $Root status --porcelain
if ($dirty) { Write-Warning "Есть незакоммиченные изменения: в песочницу попадёт только последний коммит" }

Step "1. Сеть по умолчанию: всё запрещено"
# Если домен подходит и под разрешение, и под запрет, побеждает запрет. Поэтому не запрещаем
# лишнее точечно, а ставим «запрещено всё» и открываем песочнице только нужное.
# Задаётся один раз; при повторном запуске sbx отвечает «already initialized» — это нормально.
# Если когда-то был выбран другой режим (Balanced/Open): sbx policy reset и запустить заново.
# В Windows PowerShell 5.1 перенаправление stderr при ErrorActionPreference=Stop прерывает
# скрипт на первой строке ошибки, поэтому на время этой команды — Continue.
$ErrorActionPreference = "Continue"
$out = (& sbx policy set-default deny-all 2>&1 | Out-String)
$ErrorActionPreference = "Stop"
if ($out -match "already initialized") { Write-Host "Общая политика уже задана раньше" }
else { Write-Host $out }
& sbx policy ls

Step "2. Песочница без папок хоста, с лимитами и пробросом порта API"
$exists = (& sbx ls) -match "\b$Name\b"
if ($exists -and $Recreate) {
    Invoke-Sbx rm --force $Name
    $exists = $false
}
if ($exists) {
    Write-Host "Песочница $Name уже есть — используем её (для чистой установки: -Recreate)"
} else {
    # Без пути к папке: на хосте песочнице не видно ничего (mountless sandbox).
    Invoke-Sbx create shell --name $Name --cpus $Cpus --memory $Memory --publish "${Port}:8000"
}

Step "3. Разрешения сети только для этой песочницы"
# Ollama на хосте: из песочницы адрес host.docker.internal, прокси переводит его в localhost.
# Песочница в sbx policy задаётся флагом --sandbox (позиционное имя в новых версиях убрано).
Invoke-Sbx policy allow network --sandbox $Name "localhost:11434"
foreach ($h in $TemporaryHosts) { Invoke-Sbx policy allow network --sandbox $Name $h }

try {
    Step "4. Код из последнего коммита"
    $tmp = Join-Path $env:TEMP "atm-setup"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    git -C $Root archive --format=tar.gz -o (Join-Path $tmp "atm-src.tar.gz") HEAD
    if ($LASTEXITCODE -ne 0) { throw "git archive не удался" }
    # Скрипт для bash: только LF-переводы строк и UTF-8 без BOM, иначе bash его не прочитает.
    $script = (Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "inside_setup.sh")) -replace "`r`n", "`n"
    [IO.File]::WriteAllText((Join-Path $tmp "inside_setup.sh"), $script)
    Invoke-Sbx cp $tmp "${Name}:/tmp/"

    Step "5. Установка внутри песочницы"
    Invoke-Sbx exec -u root $Name bash /tmp/atm-setup/inside_setup.sh
}
finally {
    Step "6. Закрываем временный доступ к PyPI и Ubuntu"
    foreach ($h in $TemporaryHosts) {
        & sbx policy rm network --sandbox $Name --resource $h
        if ($LASTEXITCODE -ne 0) { Write-Warning "Не удалось удалить правило для $h — удалите вручную" }
    }
}

Step "7. Итоговая политика сети"
& sbx policy ls --sandbox $Name
if ($LASTEXITCODE -ne 0) { & sbx policy ls }

Write-Host "`nГотово. Дальше:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File deploy\sandbox\start.ps1   # API в песочнице"
Write-Host "  powershell -ExecutionPolicy Bypass -File deploy\sandbox\check.ps1   # проверка изоляции"
