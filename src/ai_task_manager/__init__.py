"""AI Task Manager — мультиагентная система управления личными задачами."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-task-manager")
except PackageNotFoundError:  # пакет не установлен (например, запуск из исходников)
    __version__ = "0.0.0"
