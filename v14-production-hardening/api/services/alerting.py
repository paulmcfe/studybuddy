"""Alerting service for StudyBuddy v14.

Evaluates alert rules against current metrics and budget data.
Returns triggered alerts for display in the UI and API.

Default rules cover: budget thresholds, error rates, and latency.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    BUDGET = "budget"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"


@dataclass
class AlertRule:
    name: str
    alert_type: AlertType
    threshold: float
    severity: AlertSeverity
    description: str


@dataclass
class TriggeredAlert:
    rule_name: str
    alert_type: str
    severity: str
    description: str
    current_value: float
    threshold: float
    triggered_at: str


# Default alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        "Budget Warning", AlertType.BUDGET, 0.8, AlertSeverity.WARNING,
        "Spending has exceeded 80% of monthly budget",
    ),
    AlertRule(
        "Budget Critical", AlertType.BUDGET, 0.95, AlertSeverity.CRITICAL,
        "Spending has exceeded 95% of monthly budget",
    ),
    AlertRule(
        "High Error Rate", AlertType.ERROR_RATE, 0.05, AlertSeverity.WARNING,
        "Error rate exceeds 5% of requests",
    ),
    AlertRule(
        "Critical Error Rate", AlertType.ERROR_RATE, 0.15, AlertSeverity.CRITICAL,
        "Error rate exceeds 15% of requests",
    ),
    AlertRule(
        "High Latency", AlertType.LATENCY, 5.0, AlertSeverity.WARNING,
        "Average response latency exceeds 5 seconds",
    ),
    AlertRule(
        "Critical Latency", AlertType.LATENCY, 15.0, AlertSeverity.CRITICAL,
        "Average response latency exceeds 15 seconds",
    ),
]


class AlertEvaluator:
    """Evaluates alert rules against current system state."""

    def evaluate_all(
        self,
        metrics_collector,
        db: Session,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Evaluate all alert rules and return triggered alerts.

        Args:
            metrics_collector: The MetricsCollector instance with current metrics
            db: Database session for budget queries
            user_id: Optional user ID for user-specific budget alerts

        Returns:
            List of triggered alert dicts.
        """
        triggered = []

        for rule in DEFAULT_ALERT_RULES:
            if rule.alert_type == AlertType.ERROR_RATE:
                triggered.extend(self._check_error_rate(rule, metrics_collector))
            elif rule.alert_type == AlertType.LATENCY:
                triggered.extend(self._check_latency(rule, metrics_collector))
            elif rule.alert_type == AlertType.BUDGET:
                triggered.extend(self._check_budget(rule, db, user_id))

        return triggered

    def _check_error_rate(
        self, rule: AlertRule, metrics_collector
    ) -> list[dict]:
        """Check if error rate exceeds threshold."""
        total_requests = sum(metrics_collector.request_count.values())
        total_errors = sum(metrics_collector.error_count.values())

        if total_requests == 0:
            return []

        error_rate = total_errors / total_requests
        if error_rate >= rule.threshold:
            return [self._make_alert(rule, error_rate)]

        return []

    def _check_latency(
        self, rule: AlertRule, metrics_collector
    ) -> list[dict]:
        """Check if average latency exceeds threshold."""
        total_latency = sum(metrics_collector.latency_sum.values())
        total_count = sum(metrics_collector.latency_count.values())

        if total_count == 0:
            return []

        avg_latency = total_latency / total_count
        if avg_latency >= rule.threshold:
            return [self._make_alert(rule, avg_latency)]

        return []

    def _check_budget(
        self, rule: AlertRule, db: Session, user_id: Optional[str]
    ) -> list[dict]:
        """Check if spending exceeds budget threshold."""
        from ..database.models import BudgetAlert, TokenUsage

        if not user_id:
            return []

        budget = (
            db.query(BudgetAlert)
            .filter_by(user_id=user_id, is_active=True)
            .first()
        )
        if not budget:
            return []

        # Calculate current month's spending
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        total_spend = (
            db.query(func.sum(TokenUsage.cost_cents))
            .filter(
                TokenUsage.user_id == user_id,
                TokenUsage.created_at >= month_start,
            )
            .scalar()
        ) or 0.0

        if budget.monthly_budget_cents <= 0:
            return []

        spend_ratio = total_spend / budget.monthly_budget_cents
        if spend_ratio >= rule.threshold:
            return [self._make_alert(rule, spend_ratio)]

        return []

    def _make_alert(self, rule: AlertRule, current_value: float) -> dict:
        """Create a triggered alert dict."""
        return asdict(TriggeredAlert(
            rule_name=rule.name,
            alert_type=rule.alert_type.value,
            severity=rule.severity.value,
            description=rule.description,
            current_value=round(current_value, 4),
            threshold=rule.threshold,
            triggered_at=datetime.utcnow().isoformat(),
        ))

    def get_rules(self) -> list[dict]:
        """Return all alert rules as dicts."""
        return [asdict(rule) for rule in DEFAULT_ALERT_RULES]


# Singleton instance
alert_evaluator = AlertEvaluator()
