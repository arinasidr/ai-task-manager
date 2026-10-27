from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from ai_task_manager.workflow.state import WorkflowState

_locks_guard = Lock()
_conversation_locks = WeakValueDictionary()


def _lock_for_conversation(checkpointer: SqliteSaver, conversation_id: str) -> Lock:
    with checkpointer.cursor(transaction=False) as cursor:
        database_path = cursor.execute("PRAGMA database_list").fetchone()[2]

    storage_key = str(Path(database_path).resolve()) if database_path else id(checkpointer)
    key = (storage_key, conversation_id)

    with _locks_guard:
        lock = _conversation_locks.get(key)
        if lock is None:
            lock = Lock()
            _conversation_locks[key] = lock
        return lock


@contextmanager
def open_checkpointer(db_path: str | Path) -> Iterator[SqliteSaver]:
    """Держит SQLite Checkpointer открытым, пока работает граф."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        prune_expired_conversations(checkpointer)
        yield checkpointer


def config_for_conversation(conversation_id: str) -> RunnableConfig:
    conversation_id = conversation_id.strip()
    if not conversation_id:
        raise ValueError("conversation_id не может быть пустым")
    return {"configurable": {"thread_id": conversation_id}}


def prune_expired_conversations(
    checkpointer: SqliteSaver,
    *,
    now: datetime | None = None,
) -> list[str]:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=7)
    latest: dict[str, datetime] = {}

    for saved in checkpointer.list(None):
        thread_id = str(saved.config["configurable"]["thread_id"])
        saved_at = datetime.fromisoformat(saved.checkpoint["ts"])
        if thread_id not in latest or saved_at > latest[thread_id]:
            latest[thread_id] = saved_at

    expired: list[str] = []
    for thread_id, saved_at in latest.items():
        if saved_at >= cutoff:
            continue

        with _lock_for_conversation(checkpointer, thread_id):
            current = checkpointer.get_tuple(config_for_conversation(thread_id))
            if current is None:
                continue
            if datetime.fromisoformat(current.checkpoint["ts"]) >= cutoff:
                continue

            checkpointer.delete_thread(thread_id)
            expired.append(thread_id)

    return expired


def run_turn(
    graph: CompiledStateGraph,
    state: WorkflowState,
    *,
    build_reply: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    if not isinstance(graph.checkpointer, SqliteSaver):
        raise ValueError("Граф должен быть собран с SQLite Checkpointer")

    conversation_id = state.get("conversation_id")
    if conversation_id is None:
        conversation_id = str(uuid4())

    config = config_for_conversation(conversation_id)
    normalized_id = str(config["configurable"]["thread_id"])

    with _lock_for_conversation(graph.checkpointer, normalized_id):
        saved = graph.checkpointer.get_tuple(config)
        cutoff = datetime.now(UTC) - timedelta(days=7)
        if saved is not None and datetime.fromisoformat(saved.checkpoint["ts"]) < cutoff:
            graph.checkpointer.delete_thread(normalized_id)

        result = graph.invoke({**state, "conversation_id": normalized_id}, config)
        if build_reply is None:
            return result

        reply = build_reply(result)
        if reply is None:
            return result
        if not reply.strip():
            raise ValueError("Ответ пользователю не может быть пустым")

        history = result.get("history") or []
        if history and history[-1]["role"] == "assistant":
            history = history[:-1]
        history = [
            *history[-5:],
            {
                "role": "assistant",
                "content": reply,
                "created_at": datetime.now(UTC).isoformat(),
            },
        ]
        graph.update_state(config, {"reply": reply, "history": history})
        return {**result, "reply": reply, "history": history}
