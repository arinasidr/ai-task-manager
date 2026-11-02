import logging
from collections import defaultdict, deque
from datetime import UTC, datetime
from threading import Lock
from time import monotonic

logger = logging.getLogger(__name__)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.request_duration_seconds_sum = 0.0
        self.agent_errors = defaultdict(int)
        self.retries = defaultdict(int)
        self._recent_errors = defaultdict(deque)
        self._last_alert_at = {}
        self.alerts = deque(maxlen=20)

    def record_request(self, duration_seconds: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.request_duration_seconds_sum += duration_seconds

    def record_agent_error(self, role: str) -> None:
        now = monotonic()
        with self._lock:
            self.agent_errors[role] += 1

            recent = self._recent_errors[role]
            recent.append(now)
            while recent and now - recent[0] > 300:
                recent.popleft()

            last_alert = self._last_alert_at.get(role)
            if len(recent) >= 3 and (last_alert is None or now - last_alert >= 300):
                self._last_alert_at[role] = now
                alert = {
                    "code": "agent_error_burst",
                    "role": role,
                    "count": len(recent),
                    "created_at": datetime.now(UTC).isoformat(),
                }
                self.alerts.append(alert)
                logger.warning("Всплеск ошибок агента %s: %s за 5 минут", role, len(recent))

    def record_retry(self, role: str) -> None:
        with self._lock:
            self.retries[role] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "request_duration_seconds_sum": self.request_duration_seconds_sum,
                "agent_errors": dict(self.agent_errors),
                "retries": dict(self.retries),
                "alerts": list(self.alerts),
            }
