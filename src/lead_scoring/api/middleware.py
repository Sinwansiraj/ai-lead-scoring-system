"""
Lightweight in-memory rate limiting.

A fixed-window limiter keyed by client IP, intended to protect the public
scoring endpoints from accidental abuse on the free hosting tier. It keeps
state in process memory, which is correct for the single-worker Render
deployment; for a multi-worker or multi-instance setup, back it with a shared
store (e.g. Redis) instead.

The limiter fails open: any internal error lets the request through rather than
blocking legitimate traffic.
"""

from __future__ import annotations

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)

# Paths that must never be rate limited: health is polled by the frontend and by
# the keep-warm job; the docs/schema routes are harmless and aid evaluation.
_EXEMPT_PATHS = frozenset({"/", "/docs", "/redoc", "/openapi.json"})
_EXEMPT_PREFIXES = ("/api/v1/health",)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP fixed-window rate limiter for the scoring endpoints."""

    def __init__(self, app: ASGIApp, *, limit: int, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    @staticmethod
    def _client_ip(request: Request) -> str:
        # Render and Vercel sit behind a proxy, so prefer the forwarded client IP.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            path = request.url.path
            if path in _EXEMPT_PATHS or path.startswith(_EXEMPT_PREFIXES):
                return await call_next(request)

            now = time.monotonic()
            cutoff = now - self.window
            ip = self._client_ip(request)

            bucket = self._hits.setdefault(ip, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = int(self.window - (now - bucket[0])) + 1
                logger.warning("Rate limit exceeded for %s on %s", ip, path)
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit exceeded ({self.limit} requests/min). Retry in ~{retry_after}s."},
                    headers={"Retry-After": str(retry_after)},
                )

            bucket.append(now)

            # Bound memory: if the IP table grows unreasonably, reset it. A legitimate
            # demo never approaches this; it only trips under a scripted flood.
            if len(self._hits) > 10_000:
                self._hits.clear()
        except Exception:  # fail open: never block legitimate traffic on a limiter bug
            logger.exception("Rate limiter error; allowing request through")

        return await call_next(request)
