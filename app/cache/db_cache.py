"""Persistent response cache backed by SQLAlchemy.

SQLite locally, PostgreSQL on Railway (via DATABASE_URL). Railway's
filesystem is ephemeral, so the cache must live in the database, not on disk.
Raw upstream payloads are stored untouched so analysis can always be traced
back to exactly what NASA returned.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class CachedResponse(Base):
    __tablename__ = "cached_responses"

    key = Column(String(512), primary_key=True)
    payload = Column(Text, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)


class DbCache:
    def __init__(self, database_url: str, ttl_hours: int):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
        Base.metadata.create_all(self.engine, tables=[CachedResponse.__table__])
        self.ttl = timedelta(hours=ttl_hours)

    def _get_sync(self, key: str) -> dict | None:
        with Session(self.engine) as s:
            row = s.execute(select(CachedResponse).where(CachedResponse.key == key)).scalar_one_or_none()
            if row is None:
                return None
            fetched = row.fetched_at
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - fetched > self.ttl:
                return None
            return json.loads(row.payload)

    def _set_sync(self, key: str, payload: dict) -> None:
        with Session(self.engine) as s:
            row = s.get(CachedResponse, key)
            now = datetime.now(timezone.utc)
            text = json.dumps(payload)
            if row is None:
                s.add(CachedResponse(key=key, payload=text, fetched_at=now))
            else:
                row.payload = text
                row.fetched_at = now
            s.commit()

    async def get(self, key: str) -> dict | None:
        try:
            return await asyncio.to_thread(self._get_sync, key)
        except Exception:  # cache must never break a request
            logger.exception("cache read failed for %s", key)
            return None

    async def set(self, key: str, payload: dict) -> None:
        try:
            await asyncio.to_thread(self._set_sync, key, payload)
        except Exception:
            logger.exception("cache write failed for %s", key)
