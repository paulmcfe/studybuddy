"""Rate limiting for StudyBuddy v12.

Uses slowapi for per-user request throttling.
Limits are keyed by user ID extracted from the JWT token,
falling back to IP address for unauthenticated requests.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def get_user_identifier(request: Request) -> str:
    """Extract user identifier for rate limiting.

    Uses the JWT user_id if available, falls back to IP address.
    This ensures rate limits are per-user rather than per-IP,
    so users behind shared IPs aren't unfairly throttled.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from .auth import decode_access_token
            token = auth_header.split(" ", 1)[1]
            payload = decode_access_token(token)
            return f"user:{payload['user_id']}"
        except Exception:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_user_identifier)
