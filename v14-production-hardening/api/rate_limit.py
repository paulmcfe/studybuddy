"""Tiered rate limiting for StudyBuddy v14.

Supports per-tier limits (free/pro/enterprise) and graceful degradation
when approaching rate limits. Upgrades v12's flat per-user limits with
tier awareness and dynamic limit resolution.

Rate limits are keyed by user ID (from JWT) or IP address for
unauthenticated requests.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

# Tier-based rate limit definitions
TIER_LIMITS = {
    "free": {
        "auth": "5/minute",
        "chat": "10/minute",
        "flashcard_generate": "5/minute",
        "document_upload": "3/minute",
        "benchmark": "2/hour",
        "default": "30/minute",
    },
    "pro": {
        "auth": "10/minute",
        "chat": "60/minute",
        "flashcard_generate": "30/minute",
        "document_upload": "20/minute",
        "benchmark": "10/hour",
        "default": "120/minute",
    },
    "enterprise": {
        "auth": "20/minute",
        "chat": "200/minute",
        "flashcard_generate": "100/minute",
        "document_upload": "50/minute",
        "benchmark": "30/hour",
        "default": "500/minute",
    },
}

# At this percentage of the rate limit, start degrading service
DEGRADATION_THRESHOLD = 0.8


def get_user_identifier(request: Request) -> str:
    """Extract user identifier for rate limiting.

    Uses the JWT user_id if available, falls back to IP address.
    This ensures rate limits are per-user rather than per-IP,
    so users behind shared IPs aren't unfairly throttled.

    The return value also encodes the user's tier so that the
    dynamic_limit callable can resolve the correct limit without
    needing direct access to the request object. Format:
        "user:<id>:tier:<tier>"  or  "ip:<addr>:tier:free"
    """
    tier = "free"
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from .auth import decode_access_token
            token = auth_header.split(" ", 1)[1]
            payload = decode_access_token(token)
            user_id = payload["user_id"]
            tier = getattr(request.state, "user_tier", "free") or "free"
            return f"user:{user_id}:tier:{tier}"
        except Exception:
            pass
    return f"ip:{get_remote_address(request)}:tier:free"


def get_tier_limit(tier: str, endpoint_category: str) -> str:
    """Get the rate limit string for a tier and endpoint category."""
    tier_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    return tier_config.get(endpoint_category, tier_config["default"])


def dynamic_limit(endpoint_category: str):
    """Create a dynamic limit function for an endpoint category.

    Returns a callable that slowapi invokes per-request to determine
    the rate limit string based on the user's tier.

    slowapi passes the result of key_func(request) as the ``key``
    argument when the limit provider declares a ``key`` parameter.
    We extract the tier from the key string (set by get_user_identifier).

    Usage:
        @limiter.limit(dynamic_limit("chat"))
        async def chat_endpoint(request: Request, ...):
            ...
    """
    def _limit(key: str) -> str:
        # key format: "user:<id>:tier:<tier>" or "ip:<addr>:tier:<tier>"
        tier = "free"
        if ":tier:" in key:
            tier = key.rsplit(":tier:", 1)[1]
        return get_tier_limit(tier, endpoint_category)
    return _limit


limiter = Limiter(key_func=get_user_identifier)
