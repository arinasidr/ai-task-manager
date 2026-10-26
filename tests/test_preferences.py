import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_task_manager.db.database import connect, init_db
from ai_task_manager.db.preferences import (
    confirm_preference,
    delete_preference,
    get_preferences,
    propose_preference,
    reject_preference,
    update_preference,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.db"
    init_db(path)
    return path


def test_inferred_preference_is_inactive_until_confirmation(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        original = update_preference(connection, "preferred_work_time", "утром")
        proposal = propose_preference(connection, "preferred_work_time", "вечером")
        other = propose_preference(connection, "preferred_work_time", "ночью")
        assert get_preferences(connection) == {"preferred_work_time": "утром"}
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection) == {"preferred_work_time": "утром"}
        confirmed = confirm_preference(connection, proposal.id)
        assert confirmed.status == "confirmed"
        assert confirmed.source == "inferred"
        assert get_preferences(connection) == {"preferred_work_time": "вечером"}
        statuses = {
            row["id"]: row["status"]
            for row in connection.execute("SELECT id, status FROM preferences")
        }
        assert statuses == {
            original.id: "superseded",
            proposal.id: "confirmed",
            other.id: "superseded",
        }
        with pytest.raises(ValueError, match="уже обработано"):
            confirm_preference(connection, other.id)
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection) == {"preferred_work_time": "вечером"}


def test_explicit_change_replaces_value_and_delete_forgets_kind(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        old = update_preference(connection, "preferred_work_time", "утром")
        pending = propose_preference(connection, "preferred_work_time", "ночью")
        current = update_preference(connection, "preferred_work_time", "вечером")

        assert get_preferences(connection) == {"preferred_work_time": "вечером"}
        statuses = {
            row["id"]: row["status"]
            for row in connection.execute("SELECT id, status FROM preferences")
        }
        assert statuses == {
            old.id: "superseded",
            pending.id: "superseded",
            current.id: "confirmed",
        }
        with pytest.raises(ValueError, match="уже обработано"):
            confirm_preference(connection, pending.id)

        assert delete_preference(connection, "preferred_work_time")
        assert not delete_preference(connection, "preferred_work_time")
        assert get_preferences(connection) == {}
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection) == {}
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 0


def test_rejection_preserves_active_value_and_list_values_round_trip(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        update_preference(connection, "preferred_work_time", "утром")
        proposal = propose_preference(connection, "preferred_work_time", "вечером")
        assert reject_preference(connection, proposal.id)
        assert not reject_preference(connection, proposal.id)
        with pytest.raises(ValueError, match="уже обработано"):
            confirm_preference(connection, proposal.id)

        update_preference(connection, "working_days", [" понедельник ", "среда"])
        assert get_preferences(connection) == {
            "preferred_work_time": "утром",
            "working_days": ["понедельник", "среда"],
        }
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection)["working_days"] == ["понедельник", "среда"]


def test_invalid_preference_is_rejected_before_database_write(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        with pytest.raises(ValidationError):
            update_preference(connection, "unknown_kind", "вечером")
        with pytest.raises(ValidationError):
            propose_preference(connection, "preferred_work_time", "  ")
        with pytest.raises(ValidationError):
            propose_preference(connection, "working_days", ["понедельник", " "])
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 0


def test_rejected_value_cannot_be_proposed_again(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        proposal = propose_preference(connection, "preferred_work_time", "вечером")
        assert reject_preference(connection, proposal.id)
        connection.commit()

    with closing(connect(db_path)) as connection:
        with pytest.raises(ValueError, match="уже отклонил"):
            propose_preference(connection, "preferred_work_time", " вечером ")
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 1
        assert propose_preference(connection, "preferred_work_time", "утром").status == "proposed"


def test_rejection_covers_duplicate_proposals_and_normalized_values(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        first = propose_preference(connection, "preferred_work_time", "Вечером")
        assert propose_preference(connection, "preferred_work_time", " вечером ").id == first.id

        # Дубликат мог остаться в БД от старой версии приложения.
        duplicate = connection.execute(
            """
            INSERT INTO preferences (
                kind, value_json, status, source, created_at, updated_at
            )
            SELECT kind, value_json, status, source, created_at, updated_at
            FROM preferences WHERE id = ?
            """,
            (first.id,),
        )
        assert reject_preference(connection, first.id)
        with pytest.raises(ValueError, match="уже обработано"):
            confirm_preference(connection, duplicate.lastrowid)
        with pytest.raises(ValueError, match="уже отклонил"):
            propose_preference(connection, "preferred_work_time", "ВЕЧЕРОМ")
        assert get_preferences(connection) == {}

        days = propose_preference(connection, "working_days", ["Понедельник", "среда"])
        assert (
            propose_preference(connection, "working_days", [" СРЕДА ", "понедельник"]).id == days.id
        )
        assert reject_preference(connection, days.id)
        with pytest.raises(ValueError, match="уже отклонил"):
            propose_preference(connection, "working_days", ["среда", "ПОНЕДЕЛЬНИК"])


def test_automatic_invalidation_does_not_count_as_user_rejection(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        proposal = propose_preference(connection, "preferred_work_time", "вечером")
        update_preference(connection, "preferred_work_time", "утром")
        status = connection.execute(
            "SELECT status FROM preferences WHERE id = ?", (proposal.id,)
        ).fetchone()[0]
        assert status == "superseded"

        repeated = propose_preference(connection, "preferred_work_time", "вечером")
        assert repeated.id != proposal.id
        other = propose_preference(connection, "preferred_work_time", "ночью")
        confirm_preference(connection, repeated.id)
        other_status = connection.execute(
            "SELECT status FROM preferences WHERE id = ?", (other.id,)
        ).fetchone()[0]
        assert other_status == "superseded"
        assert propose_preference(connection, "preferred_work_time", "ночью").id != other.id


@pytest.mark.parametrize("commit_before_failure", [True, False])
def test_failed_replacement_rolls_back_even_if_caller_commits(
    db_path: Path, commit_before_failure: bool
) -> None:
    with closing(connect(db_path)) as connection:
        update_preference(connection, "preferred_work_time", "утром")
        if commit_before_failure:
            connection.commit()
        connection.execute("""
            CREATE TRIGGER block_preference_insert BEFORE INSERT ON preferences
            BEGIN SELECT RAISE(ABORT, 'forced insert failure'); END
        """)

        with pytest.raises(sqlite3.IntegrityError, match="forced insert failure"):
            update_preference(connection, "preferred_work_time", "вечером")
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection) == {"preferred_work_time": "утром"}
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 1


def test_failed_confirmation_preserves_active_and_pending_values(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        update_preference(connection, "preferred_work_time", "утром")
        proposal = propose_preference(connection, "preferred_work_time", "вечером")
        connection.commit()
        connection.execute("""
            CREATE TRIGGER block_confirmation BEFORE UPDATE OF status ON preferences
            WHEN OLD.status = 'proposed' AND NEW.status = 'confirmed'
            BEGIN SELECT RAISE(ABORT, 'forced confirmation failure'); END
        """)

        with pytest.raises(sqlite3.IntegrityError, match="forced confirmation failure"):
            confirm_preference(connection, proposal.id)
        connection.commit()

    with closing(connect(db_path)) as connection:
        assert get_preferences(connection) == {"preferred_work_time": "утром"}
        assert (
            connection.execute(
                "SELECT status FROM preferences WHERE id = ?", (proposal.id,)
            ).fetchone()[0]
            == "proposed"
        )


def test_confirmation_holds_write_lock_before_reading_proposal(db_path: Path) -> None:
    with closing(connect(db_path)) as first, closing(connect(db_path)) as other:
        update_preference(first, "preferred_work_time", "утром")
        proposal = propose_preference(first, "preferred_work_time", "ночью")
        first.commit()
        other.execute("PRAGMA busy_timeout = 50")

        class InterleavingConnection:
            blocked = False
            attempted = False

            @property
            def in_transaction(self) -> bool:
                return first.in_transaction

            @property
            def total_changes(self) -> int:
                return first.total_changes

            def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
                cursor = first.execute(sql, params)
                if not self.attempted and sql.strip().startswith(
                    "SELECT * FROM preferences WHERE id = ?"
                ):
                    self.attempted = True
                    try:
                        update_preference(other, "preferred_work_time", "вечером")
                    except sqlite3.OperationalError as exc:
                        assert "locked" in str(exc)
                        other.rollback()
                        self.blocked = True
                    else:
                        other.commit()
                return cursor

            def rollback(self) -> None:
                first.rollback()

        interleaving = InterleavingConnection()
        confirm_preference(interleaving, proposal.id)
        assert interleaving.attempted and interleaving.blocked
        first.commit()

        update_preference(other, "preferred_work_time", "вечером")
        other.commit()
        assert get_preferences(first) == {"preferred_work_time": "вечером"}


def test_noop_preference_calls_release_their_write_lock(db_path: Path) -> None:
    with closing(connect(db_path)) as first, closing(connect(db_path)) as second:
        proposal = propose_preference(first, "preferred_work_time", "вечером")
        first.commit()
        second.execute("PRAGMA busy_timeout = 50")

        assert propose_preference(first, "preferred_work_time", " ВЕЧЕРОМ ").id == proposal.id
        assert not first.in_transaction
        assert not reject_preference(first, 9999)
        assert not first.in_transaction
        assert not delete_preference(first, "working_days")
        assert not first.in_transaction

        second.execute("BEGIN IMMEDIATE")
        second.rollback()
