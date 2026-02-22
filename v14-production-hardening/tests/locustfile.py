"""Load test script for StudyBuddy v14.

Usage:
    pip install locust
    locust -f tests/locustfile.py --host=http://localhost:8000

    Then open http://localhost:8089 to configure and start the test.

Tests:
- User registration and login flows
- Program CRUD operations
- Chat with tutor
- Flashcard generation
- Health and metrics endpoints
- Cost analytics and production checklist
"""

import uuid

from locust import HttpUser, task, between


class StudyBuddyUser(HttpUser):
    """Simulates a typical StudyBuddy user session."""

    wait_time = between(1, 5)
    token: str = ""
    program_id: str = ""

    def on_start(self):
        """Register a new user and log in."""
        email = f"loadtest-{uuid.uuid4().hex[:8]}@test.com"
        resp = self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "loadtest123"},
        )
        if resp.status_code == 200:
            data = resp.json()
            self.token = data.get("access_token", "")

    def _headers(self):
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    @task(3)
    def list_programs(self):
        """List all learning programs."""
        self.client.get("/api/programs", headers=self._headers())

    @task(1)
    def create_program(self):
        """Create a new learning program."""
        resp = self.client.post(
            "/api/programs",
            json={
                "name": f"Load Test Program {uuid.uuid4().hex[:6]}",
                "description": "Created by load test",
            },
            headers=self._headers(),
        )
        if resp.status_code == 200:
            self.program_id = resp.json().get("id", "")

    @task(5)
    def chat_with_tutor(self):
        """Send a chat message to the tutor."""
        if not self.program_id:
            return
        self.client.post(
            f"/api/programs/{self.program_id}/chat",
            json={
                "message": "Explain the concept of spaced repetition briefly",
                "history": [],
            },
            headers=self._headers(),
        )

    @task(3)
    def generate_flashcard(self):
        """Generate a flashcard."""
        if not self.program_id:
            return
        self.client.post(
            f"/api/programs/{self.program_id}/flashcards/generate",
            json={"topic": "Load test topic"},
            headers=self._headers(),
        )

    @task(2)
    def check_health(self):
        """Hit the health check endpoint."""
        self.client.get("/api/health")

    @task(1)
    def get_metrics(self):
        """Fetch Prometheus metrics."""
        self.client.get("/api/metrics")

    @task(2)
    def get_costs(self):
        """Fetch cost data."""
        self.client.get("/api/costs?days=7", headers=self._headers())

    @task(1)
    def get_cost_analytics(self):
        """Fetch per-feature cost breakdown."""
        self.client.get("/api/costs/by-feature?days=7", headers=self._headers())

    @task(1)
    def get_production_checklist(self):
        """Run production readiness checks."""
        self.client.get("/api/production-checklist", headers=self._headers())

    @task(1)
    def get_alerts(self):
        """Fetch current alerts."""
        self.client.get("/api/alerts", headers=self._headers())
