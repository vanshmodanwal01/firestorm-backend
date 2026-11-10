"""Async sliding-window rate limiter.

Keeps us under a self-imposed requests-per-window budget so we never hammer
NASA POWER. POWER's real limits are not formally documented and have been
reported to differ per endpoint, so this is deliberately conservative.
"""
import asyncio
import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        self.max_requests = max_requests
        self.window = window_seconds
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a request slot is free. Returns seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.max_requests:
                    self._stamps.append(now)
                    return waited
                sleep_for = self.window - (now - self._stamps[0])
                await asyncio.sleep(max(sleep_for, 0.01))
                waited += sleep_for
