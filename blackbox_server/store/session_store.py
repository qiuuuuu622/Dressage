from __future__ import annotations

import asyncio
from collections.abc import Callable

from blackbox_server.core.errors import SessionCapacityError
from blackbox_server.core.models import SessionContext, SessionState, TurnStatus, utcnow


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_index_lock = asyncio.Lock()

    async def get(self, session_id: str) -> SessionContext | None:
        return self._sessions.get(session_id)

    async def get_lock(self, session_id: str) -> asyncio.Lock | None:
        return self._locks.get(session_id)

    async def get_or_create(
        self,
        session_id: str,
        *,
        factory: Callable[[], SessionContext],
        max_sessions: int,
    ) -> tuple[SessionContext, asyncio.Lock, bool]:
        async with self._session_index_lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing, self._locks[session_id], False

            if len(self._sessions) >= max_sessions:
                self._evict_oldest_aborted_locked()

            if len(self._sessions) >= max_sessions:
                raise SessionCapacityError()

            session = factory()
            lock = asyncio.Lock()
            self._sessions[session_id] = session
            self._locks[session_id] = lock
            return session, lock, True

    async def counts(self) -> dict[str, int]:
        active = 0
        desynced = 0
        aborted = 0
        for session in self._sessions.values():
            if session.state == SessionState.ACTIVE:
                active += 1
            elif session.state == SessionState.DESYNCED:
                desynced += 1
            elif session.state == SessionState.ABORTED:
                aborted += 1
        return {
            "active_count": active,
            "desynced_count": desynced,
            "aborted_count": aborted,
            "total_count": len(self._sessions),
        }

    async def has_open_sessions(self) -> bool:
        return any(
            session.state in {SessionState.ACTIVE, SessionState.DESYNCED}
            for session in self._sessions.values()
        )

    async def has_inflight_turns(self) -> bool:
        return any(
            any(turn.status == TurnStatus.INFLIGHT for turn in session.turn_ledger.values())
            for session in self._sessions.values()
        )

    async def mark_all_non_aborted_desynced(self) -> None:
        now = utcnow()
        for session in self._sessions.values():
            if session.state != SessionState.ABORTED:
                session.state = SessionState.DESYNCED
                session.updated_at = now

    async def reset_all(self) -> None:
        async with self._session_index_lock:
            self._sessions = {}
            self._locks = {}

    def _evict_oldest_aborted_locked(self) -> None:
        aborted = [session for session in self._sessions.values() if session.state == SessionState.ABORTED]
        if not aborted:
            return
        oldest = min(aborted, key=lambda item: (item.updated_at, item.created_at, item.session_id))
        self._sessions.pop(oldest.session_id, None)
        self._locks.pop(oldest.session_id, None)
