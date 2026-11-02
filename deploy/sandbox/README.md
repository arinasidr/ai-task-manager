# Изолированное окружение: Docker Sandboxes (P13)

Приложение (FastAPI + LangGraph + Tools + SQLite) работает внутри microVM Docker Sandboxes.
Ollama остаётся на хосте, потому что ей нужна видеокарта. Почему выбран этот вариант — P06.

```
Хост (Windows 11, ASUS)
├── Ollama :11434            ← модели и видеокарта
└── microVM «atm» (2 CPU, 2 ГБ)
    ├── /opt/atm             код, владелец root, для приложения только чтение
    ├── /var/lib/atm         SQLite: задачи, предпочтения, память диалога (единственное место записи)
    ├── пользователь atm     без sudo, не в группе docker
    ├── uvicorn 0.0.0.0:8000 → проброшен на localhost:8000 хоста
    └── сеть: запрещено всё, кроме host.docker.internal:11434 (Ollama)
```

## Что ограничено и чем

| Ограничение | Как сделано |
|---|---|
| Файлы хоста | песочница создана без папки (`sbx create shell` без пути): диски хоста не смонтированы вовсе. Код копируется архивом последнего коммита (`git archive`), без `.env`, `.venv`, `data/` |
| Файлы внутри VM | код в `/opt/atm` принадлежит root; приложение пишет только в `/var/lib/atm` |
| Права | приложение работает от `atm` без sudo; Docker Engine внутри VM ему недоступен; Docker хоста песочнице недоступен всегда |
| Сеть | общая политика `deny-all`. Песочнице разрешён только `localhost:11434` — так прокси называет Ollama на хосте. PyPI и Ubuntu открываются только на время установки и сразу закрываются |
| Ресурсы | `--cpus 2 --memory 2g` |
| Ядро | своё ядро microVM: побег из контейнера не даёт доступа к ядру хоста |

## Установка (Windows, один раз)

В PowerShell от администратора:

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All   # затем перезагрузка
winget install -h Docker.sbx
sbx login
```

Затем в папке проекта. Ollama должна быть запущена, а изменения закоммичены: в песочницу
попадает последний коммит.

```powershell
powershell -ExecutionPolicy Bypass -File deploy\sandbox\setup.ps1     # создать и установить
powershell -ExecutionPolicy Bypass -File deploy\sandbox\start.ps1     # API (окно остаётся занятым)
powershell -ExecutionPolicy Bypass -File deploy\sandbox\check.ps1     # в другом окне: проверка
```

После `start.ps1` откройте http://localhost:8000/docs или запустите `atm chat` (CLI ходит в
API на `localhost:8000`).

Обновить код в песочнице после нового коммита: `setup.ps1 -Recreate`.
Удалить песочницу: `sbx rm atm`.

## Что показывает check.ps1

`check_isolation.py` запускается внутри песочницы от `atm` и проверяет:

- **пользователь:** не root, без sudo;
- **файлы:** код только для чтения, данные на запись, нет папок хоста;
- **сеть:** Ollama доступна, а `example.com`, `pypi.org`, `1.1.1.1` и другие порты хоста — нет;
- **Docker:** нет доступа к Docker Engine;
- **ресурсы:** CPU ≤ 2, память ≈ 2 ГБ.

С хоста `check.ps1` проверяет, что API отвечает на проброшенном порту, и показывает политику
сети (`sbx policy ls atm`) и журнал решений прокси (`sbx policy log atm`).

## Если что-то не так

| Симптом | Что делать |
|---|---|
| `policy allow network atm …` — ошибка аргументов | в старых версиях sbx песочница задаётся флагом: `--sandbox atm` |
| «Модель недоступна» в ответах API | `sbx policy ls atm`: должно быть правило `localhost:11434`; Ollama запущена на хосте |
| pip в setup не скачивает пакеты | `sbx policy log atm`: какой адрес заблокирован — добавьте его в `$TemporaryHosts` в `setup.ps1` |
| `sbx` не запускается | `sbx diagnose`; проверить, что Windows Hypervisor Platform включена |
| Docker Sandboxes не работает совсем | запасной вариант из P06: Docker Engine в WSL2 + gVisor (`runsc`) |
