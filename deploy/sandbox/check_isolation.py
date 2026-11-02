"""Проверка изоляции AI Task Manager внутри Docker Sandboxes (P13).

Запускается ВНУТРИ песочницы от пользователя приложения:

    sbx exec -u atm -w /opt/atm atm /opt/atm/.venv/bin/python deploy/sandbox/check_isolation.py

Что проверяется (то же самое ограничено настройками в setup.ps1 и inside_setup.sh):
- пользователь: не root и без sudo;
- файлы: код приложения только для чтения, запись — только в каталог данных; папок хоста нет;
- сеть: Ollama на хосте доступна, интернет и другие порты хоста — нет;
- Docker: нет доступа к Docker Engine;
- ресурсы: лимиты CPU и памяти действуют.

Только стандартная библиотека Python. Код возврата 0 — все обязательные проверки пройдены.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Файловые системы, через которые VM видит папки хоста.
HOST_SHARE_TYPES = {
    "virtiofs",
    "9p",
    "drvfs",
    "vboxsf",
    "cifs",
    "smb3",
    "nfs",
    "nfs4",
    "fuse.sshfs",
}
# Пути, где обычно появляются диски Windows/macOS при монтировании.
HOST_PATHS = ["/mnt/c", "/mnt/host", "/c", "/Users", "/host", "/run/desktop/mnt/host"]
# Код ответа прокси песочницы на запрещённый адрес.
BLOCKED_STATUSES = {403, 407, 451, 502, 503}


@dataclass
class Check:
    name: str
    status: str  # ok | fail | warn
    detail: str


# ---------------------------------------------------------------- чистые функции (тестируются)


def host_mounts(mounts_text: str) -> list[tuple[str, str, bool]]:
    """Монтирования папок хоста из /proc/mounts: (точка, тип ФС, только чтение)."""
    found = []
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        _, mountpoint, fstype, options = parts[:4]
        if fstype in HOST_SHARE_TYPES or (fstype.startswith("fuse.") and fstype != "fuse.lxcfs"):
            found.append((mountpoint, fstype, "ro" in options.split(",")))
    return found


def classify_response(status: int | None, error: str | None) -> str:
    """'open' — адрес доступен; 'blocked' — запрос не прошёл (ошибка или отказ прокси)."""
    if status is None:
        return "blocked"
    if status in BLOCKED_STATUSES:
        return "blocked"
    return "open"


def mem_total_gib(meminfo_text: str) -> float | None:
    for line in meminfo_text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024 / 1024
    return None


# ---------------------------------------------------------------- проверки


def probe(url: str, timeout: float = 8.0) -> tuple[int | None, str | None, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.status, None, response.read(200_000)
    except urllib.error.HTTPError as exc:
        return exc.code, f"HTTP {exc.code}", b""
    except Exception as exc:  # noqa: BLE001 — любая ошибка соединения = недоступно
        return None, f"{type(exc).__name__}: {exc}"[:160], b""


def check_user() -> list[Check]:
    checks = []
    uid = os.geteuid()
    name = subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
    checks.append(
        Check("пользователь не root", "ok" if uid != 0 else "fail", f"{name} (uid {uid})")
    )
    try:
        sudo = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10)
        has_sudo = sudo.returncode == 0
    except FileNotFoundError:
        has_sudo = False
    checks.append(
        Check(
            "нет sudo",
            "fail" if has_sudo else "ok",
            "sudo без пароля работает" if has_sudo else "sudo запрещён",
        )
    )
    return checks


def check_files(app_dir: Path, data_dir: Path) -> list[Check]:
    checks = []
    target = app_dir / "src" / "ai_task_manager" / "__init__.py"
    try:
        with target.open("a"):
            pass
        checks.append(Check("код только для чтения", "fail", f"удалось открыть {target} на запись"))
    except OSError as exc:
        checks.append(Check("код только для чтения", "ok", type(exc).__name__))

    probe_file = data_dir / ".write-check"
    try:
        probe_file.write_text("ok")
        probe_file.unlink()
        checks.append(Check("данные доступны на запись", "ok", str(data_dir)))
    except OSError as exc:
        checks.append(Check("данные доступны на запись", "fail", f"{data_dir}: {exc}"))

    mounts = host_mounts(Path("/proc/mounts").read_text())
    writable = [m for m in mounts if not m[2]]
    if writable:
        shown = ", ".join(f"{point} ({fs})" for point, fs, _ in writable)
        checks.append(Check("нет папок хоста на запись", "fail", shown))
    else:
        shown = ", ".join(f"{point} ({fs}, ro)" for point, fs, _ in mounts) or "монтирований нет"
        checks.append(Check("нет папок хоста на запись", "ok", shown))

    present = [path for path in HOST_PATHS if Path(path).exists() and any(Path(path).iterdir())]
    checks.append(
        Check(
            "диски хоста не видны",
            "fail" if present else "ok",
            ", ".join(present) if present else "ни одного из " + ", ".join(HOST_PATHS),
        )
    )
    return checks


def check_network(
    ollama_url: str, required_models: list[str], extra_ports: list[int]
) -> list[Check]:
    checks = []
    status, error, body = probe(f"{ollama_url.rstrip('/')}/api/tags")
    if status == 200:
        try:
            names = {model["name"] for model in json.loads(body)["models"]}
        except (ValueError, KeyError, TypeError):
            names = set()
        missing = [m for m in required_models if m not in names and f"{m}:latest" not in names]
        checks.append(
            Check("Ollama на хосте доступна", "ok", f"{ollama_url}, моделей: {len(names)}")
        )
        checks.append(
            Check(
                "нужные модели установлены",
                "warn" if missing else "ok",
                ("нет: " + ", ".join(missing)) if missing else ", ".join(required_models),
            )
        )
    else:
        checks.append(Check("Ollama на хосте доступна", "fail", error or f"HTTP {status}"))

    outside = [
        "https://example.com",
        "http://example.com",
        "https://pypi.org/simple/",
        "http://1.1.1.1",
    ]
    outside += [f"http://host.docker.internal:{port}" for port in extra_ports]
    for url in outside:
        status, error, _ = probe(url, timeout=6)
        verdict = classify_response(status, error)
        checks.append(
            Check(
                f"закрыт {url}",
                "ok" if verdict == "blocked" else "fail",
                error or f"HTTP {status}",
            )
        )
    return checks


def check_docker() -> list[Check]:
    socket = Path("/var/run/docker.sock")
    if not socket.exists():
        return [Check("нет доступа к Docker", "ok", "сокета нет")]
    writable = os.access(socket, os.W_OK)
    return [
        Check(
            "нет доступа к Docker",
            "fail" if writable else "ok",
            "сокет доступен на запись" if writable else "сокет есть, но доступа нет",
        )
    ]


def check_resources(max_cpus: int, max_mem_gib: float) -> list[Check]:
    cpus = os.cpu_count() or 0
    mem = mem_total_gib(Path("/proc/meminfo").read_text()) or 0.0
    return [
        Check(f"CPU не больше {max_cpus}", "ok" if cpus <= max_cpus else "fail", f"{cpus}"),
        Check(
            f"память не больше {max_mem_gib:g} ГиБ",
            # ядро держит часть памяти себе, поэтому допускаем 10 % сверху
            "ok" if mem <= max_mem_gib * 1.1 else "fail",
            f"{mem:.2f} ГиБ",
        ),
    ]


def settings_from_app() -> tuple[str, list[str]]:
    try:
        from ai_task_manager.config import get_settings
    except ImportError:
        return os.environ.get("ATM_OLLAMA_BASE_URL", "http://host.docker.internal:11434"), []
    settings = get_settings()
    models = [
        settings.analyst_model,
        settings.planner_model,
        settings.fast_model,
        settings.validator_model,
        settings.decision_model,
    ]
    return settings.ollama_base_url, list(dict.fromkeys(models))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-dir", type=Path, default=Path("/opt/atm"))
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/atm"))
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=2.0)
    parser.add_argument(
        "--closed-host-ports",
        type=int,
        nargs="*",
        default=[8080, 6006],
        help="порты хоста, которые должны быть закрыты (Phoenix 6006 откроем в P14)",
    )
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = parser.parse_args(argv)

    ollama_url, models = settings_from_app()
    checks = [
        *check_user(),
        *check_files(args.app_dir, args.data_dir),
        *check_network(ollama_url, models, args.closed_host_ports),
        *check_docker(),
        *check_resources(args.cpus, args.memory_gib),
    ]

    if args.json:
        print(json.dumps([c.__dict__ for c in checks], ensure_ascii=False, indent=1))
    else:
        marks = {"ok": "OK  ", "fail": "FAIL", "warn": "WARN"}
        for check in checks:
            print(f"[{marks[check.status]}] {check.name}: {check.detail}")
    failed = [c for c in checks if c.status == "fail"]
    print(f"\nИтог: {len(checks) - len(failed)} из {len(checks)} проверок пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
