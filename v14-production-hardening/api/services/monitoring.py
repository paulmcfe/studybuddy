"""Production monitoring for StudyBuddy v14.

Collects request metrics via FastAPI middleware and exposes them
in Prometheus text format at /api/metrics.

v14 additions:
- P95/P99 latency tracking using sorted duration lists
- Error rate computation
- Active alert count metric

Tracks:
- Request counts by endpoint and method
- Response latency (average, P95, P99 per endpoint)
- Error counts by status code
- Error rate (percentage)
- Active concurrent requests
- Active alert count
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
        # v14: Store recent latencies for percentile computation
        self.latency_samples: dict[str, list[float]] = defaultdict(list)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.active_requests: int = 0
        self.active_alerts: int = 0
        self._lock = Lock()
        # Keep at most this many samples per endpoint for percentile calculations
        self._max_samples = 1000

    def record_request(self, method: str, path: str, status: int, duration: float):
        """Record a completed HTTP request."""
        with self._lock:
            key = f"{method}:{path}"
            self.request_count[key] += 1
            self.latency_sum[key] += duration
            self.latency_count[key] += 1
            self.total_requests += 1

            # v14: Track latency samples for percentiles
            samples = self.latency_samples[key]
            samples.append(duration)
            if len(samples) > self._max_samples:
                self.latency_samples[key] = samples[-self._max_samples:]

            if status >= 400:
                self.error_count[f"{key}:{status}"] += 1
                self.total_errors += 1

    def increment_active(self):
        """Increment active request count."""
        with self._lock:
            self.active_requests += 1

    def decrement_active(self):
        """Decrement active request count."""
        with self._lock:
            self.active_requests -= 1

    def set_active_alerts(self, count: int):
        """Update the active alert count."""
        with self._lock:
            self.active_alerts = count

    def get_error_rate(self) -> float:
        """Get the current error rate as a percentage."""
        with self._lock:
            if self.total_requests == 0:
                return 0.0
            return (self.total_errors / self.total_requests) * 100

    def get_p95_latency(self) -> float:
        """Get the overall P95 latency across all endpoints."""
        with self._lock:
            all_samples = []
            for samples in self.latency_samples.values():
                all_samples.extend(samples)
            if not all_samples:
                return 0.0
            all_samples.sort()
            idx = int(len(all_samples) * 0.95)
            return all_samples[min(idx, len(all_samples) - 1)]

    def get_p99_latency(self) -> float:
        """Get the overall P99 latency across all endpoints."""
        with self._lock:
            all_samples = []
            for samples in self.latency_samples.values():
                all_samples.extend(samples)
            if not all_samples:
                return 0.0
            all_samples.sort()
            idx = int(len(all_samples) * 0.99)
            return all_samples[min(idx, len(all_samples) - 1)]

    def _percentile(self, samples: list[float], pct: float) -> float:
        """Compute a percentile from a sorted list of samples."""
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        idx = int(len(sorted_s) * pct)
        return sorted_s[min(idx, len(sorted_s) - 1)]

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

            # v14: P95 and P99 latency per endpoint
            lines.append("")
            lines.append("# HELP studybuddy_request_duration_p95_seconds P95 request latency")
            lines.append("# TYPE studybuddy_request_duration_p95_seconds gauge")
            for key in sorted(self.latency_samples.keys()):
                method, path = key.split(":", 1)
                p95 = self._percentile(self.latency_samples[key], 0.95)
                lines.append(
                    f'studybuddy_request_duration_p95_seconds{{method="{method}",path="{path}"}} {p95:.4f}'
                )

            lines.append("")
            lines.append("# HELP studybuddy_request_duration_p99_seconds P99 request latency")
            lines.append("# TYPE studybuddy_request_duration_p99_seconds gauge")
            for key in sorted(self.latency_samples.keys()):
                method, path = key.split(":", 1)
                p99 = self._percentile(self.latency_samples[key], 0.99)
                lines.append(
                    f'studybuddy_request_duration_p99_seconds{{method="{method}",path="{path}"}} {p99:.4f}'
                )

            lines.append("")
            lines.append("# HELP studybuddy_errors_total Total HTTP errors (4xx/5xx)")
            lines.append("# TYPE studybuddy_errors_total counter")
            for key, count in sorted(self.error_count.items()):
                lines.append(f'studybuddy_errors_total{{endpoint="{key}"}} {count}')

            # v14: Error rate
            lines.append("")
            lines.append("# HELP studybuddy_error_rate Error rate percentage")
            lines.append("# TYPE studybuddy_error_rate gauge")
            error_rate = (self.total_errors / max(self.total_requests, 1)) * 100
            lines.append(f"studybuddy_error_rate {error_rate:.2f}")

            lines.append("")
            lines.append("# HELP studybuddy_active_requests Current active requests")
            lines.append("# TYPE studybuddy_active_requests gauge")
            lines.append(f"studybuddy_active_requests {self.active_requests}")

            # v14: Active alerts
            lines.append("")
            lines.append("# HELP studybuddy_active_alerts Number of active alerts")
            lines.append("# TYPE studybuddy_active_alerts gauge")
            lines.append(f"studybuddy_active_alerts {self.active_alerts}")

        return "\n".join(lines) + "\n"


# Global metrics instance
metrics = MetricsCollector()
