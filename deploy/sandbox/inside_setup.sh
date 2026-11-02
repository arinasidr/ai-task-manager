#!/usr/bin/env bash
# Установка AI Task Manager ВНУТРИ песочницы Docker Sandboxes (P13). Запускает setup.ps1:
#
#   sbx exec -u root atm bash /tmp/atm-setup/inside_setup.sh
#
# Что делает:
#   1. проверяет Python >= 3.11 и venv (при необходимости ставит python3-venv);
#   2. создаёт пользователя приложения atm БЕЗ sudo — приложение не может менять систему;
#   3. распаковывает код в /opt/atm (владелец root, для atm только чтение);
#   4. ставит зависимости в /opt/atm/.venv (сеть до PyPI открыта временно, setup.ps1 её закроет);
#   5. создаёт /var/lib/atm — единственный каталог, куда приложение может писать (SQLite);
#   6. пишет /opt/atm/.env: Ollama на хосте через host.docker.internal, API на 0.0.0.0;
#   7. прогоняет тесты от имени atm.
set -euo pipefail

ARCHIVE="${1:-/tmp/atm-setup/atm-src.tar.gz}"
APP_DIR=/opt/atm
DATA_DIR=/var/lib/atm
APP_USER=atm

step() { printf '\n=== %s\n' "$*"; }

# Прокси песочницы может подменять TLS-сертификаты: pip должен доверять системному хранилищу.
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  export PIP_CERT=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

step "1. Python"
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' \
  || { echo "Нужен Python 3.11+"; exit 1; }
python3 --version
if ! python3 -m venv /tmp/atm-venv-check >/dev/null 2>&1; then
  echo "venv недоступен — ставлю python3-venv (нужен временный доступ к archive.ubuntu.com)"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
fi
rm -rf /tmp/atm-venv-check

step "2. Пользователь $APP_USER без sudo"
id "$APP_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$APP_USER"
for group in sudo admin wheel docker; do
  gpasswd -d "$APP_USER" "$group" >/dev/null 2>&1 || true
done
rm -f "/etc/sudoers.d/$APP_USER"
id "$APP_USER"

step "3. Код в $APP_DIR (только чтение для $APP_USER)"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"
tar -xzf "$ARCHIVE" -C "$APP_DIR"

step "4. Зависимости"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --no-cache-dir -q -r "$APP_DIR/requirements.txt" \
  --no-warn-script-location
"$APP_DIR/.venv/bin/python" -c 'import ai_task_manager, langgraph; print("пакеты на месте")'

step "5. Каталог данных $DATA_DIR"
mkdir -p "$DATA_DIR"
chown "$APP_USER:$APP_USER" "$DATA_DIR"
chmod 700 "$DATA_DIR"

step "6. Настройки"
cat > "$APP_DIR/.env" <<'EOF'
# Настройки внутри песочницы (P13). Остальные значения — по умолчанию из config.py.
ATM_API_HOST=0.0.0.0
ATM_DATABASE_PATH=/var/lib/atm/tasks.db
# 127.0.0.1 хоста из песочницы недоступен; прокси переводит этот адрес в localhost хоста,
# а политика сети пропускает только порт Ollama.
ATM_OLLAMA_BASE_URL=http://host.docker.internal:11434
EOF
chown -R root:root "$APP_DIR"
chmod -R u=rwX,go=rX "$APP_DIR"

step "7. Тесты от имени $APP_USER"
cd "$APP_DIR"
# Чистое окружение: переменные root (например, SSL_CERT_FILE на недоступный atm файл) не нужны,
# тестам сеть не нужна.
runuser -u "$APP_USER" -- env -i PATH=/usr/bin:/bin HOME="/home/$APP_USER" LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 "$APP_DIR/.venv/bin/python" -m pytest -q -p no:cacheprovider

echo
echo "Готово. Запуск API: deploy/sandbox/start.ps1, проверка изоляции: deploy/sandbox/check.ps1"
