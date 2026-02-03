"""Production monitoring for StudyBuddy v12.

Collects request metrics via FastAPI middleware and exposes them
in Prometheus text format at /api/metrics.

Tracks:
- Request counts by endpoint and method
- Response latency (average per endpoint)
- Error counts by status code
- Active concurrent requests
"""

import re
from collections import defaultdict
from threading import Lock


# UUID pattern for path normalization
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def normalize_path(path: str) -> str:
    """Replace UUIDs in paths with :id to avoid high-cardinality metrics."""
    return _UUID_PATTERN.sub(":id", path)


class MetricsCollector:
    """Thread-safe metrics collection for HTTP requests."""

    def __init__(self):
        self.request_count: dict[str, int] = defaultdict(int)
        self.error_count: dict[str, int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.active_requests: int = 0
        self._lock = Lock()

    def record_request(self, method: str, path: str, status: int, duration: float):
        """Record a completed HTTP request."""
        with self._lock:
            key = f"{method}:{path}"
            self.request_count[key] += 1
            self.latency_sum[key] += duration
            self.latency_count[key] += 1
            if status >= 400:
                self.error_count[f"{key}:{status}"] += 1

    def increment_active(self):
        """Increment active request count."""
        with self._lock:
            self.active_requests += 1

    def decrement_active(self):
        """Decrement active request count."""
        with self._lock:
            self.active_requests -= 1

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        lines.append("# HELP studybuddy_requests_total Total HTTP requests")
        lines.append("# TYPE studybuddy_requests_total counter")
        with self._lock:
            for key, count in sorted(self.request_count.items()):
                method, path = key.split(":", 1)
                lines.append(
                    f'studybuddy_requests_total{{method="{method}",path="{path}"}} {count}'
                )

            lines.append("")
            lines.append("# HELP studybuddy_request_duration_avg_seconds Average request latency")
            lines.append("# TYPE studybuddy_request_duration_avg_seconds gauge")
            for key in sorted(self.latency_sum.keys()):
                method, path = key.split(":", 1)
                avg = self.latency_sum[key] / max(self.latency_count[key], 1)
                lines.append(
                    f'studybuddy_request_duration_avg_seconds{{method="{method}",path="{path}"}} {avg:.4f}'
                )

            lines.append("")
            lines.append("# HELP studybuddy_errors_total Total HTTP errors (4xx/5xx)")
            lines.append("# TYPE studybuddy_errors_total counter")
            for key, count in sorted(self.error_count.items()):
                lines.append(f'studybuddy_errors_total{{endpoint="{key}"}} {count}')

            lines.append("")
            lines.append("# HELP studybuddy_active_requests Current active requests")
            lines.append("# TYPE studybuddy_active_requests gauge")
            lines.append(f"studybuddy_active_requests {self.active_requests}")

        return "\n".join(lines) + "\n"


# Global metrics instance
metrics = MetricsCollector()
