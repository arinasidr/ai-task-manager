"""P13: разбор результатов проверки изоляции (deploy/sandbox/check_isolation.py)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "deploy" / "sandbox" / "check_isolation.py"
spec = importlib.util.spec_from_file_location("check_isolation", SCRIPT)
check_isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_isolation)


def test_host_shares_are_found_in_mounts() -> None:
    mounts = "\n".join(
        [
            "overlay / overlay rw,relatime 0 0",
            "proc /proc proc rw,nosuid 0 0",
            "lxcfs /proc/meminfo fuse.lxcfs rw 0 0",
            "workspace /c/Users/user/project virtiofs rw,relatime 0 0",
            "skills /home/agent/.skills virtiofs ro,relatime 0 0",
        ]
    )
    assert check_isolation.host_mounts(mounts) == [
        ("/c/Users/user/project", "virtiofs", False),
        ("/home/agent/.skills", "virtiofs", True),
    ]


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (None, "URLError: Tunnel connection failed: 403 Forbidden", "blocked"),
        (403, "HTTP 403", "blocked"),
        (200, None, "open"),
        (404, "HTTP 404", "open"),  # сервер ответил — значит, сеть пропустила
    ],
)
def test_proxy_refusal_counts_as_blocked(status, error, expected) -> None:
    assert check_isolation.classify_response(status, error) == expected


def test_memory_limit_is_read_from_meminfo() -> None:
    assert check_isolation.mem_total_gib("MemTotal:  2097152 kB\nMemFree: 1 kB") == 2.0
