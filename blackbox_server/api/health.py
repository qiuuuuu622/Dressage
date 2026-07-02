from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from blackbox_server.core.models import ServerState
from blackbox_server.core.models import StatusResponse


router = APIRouter()


@router.get("/health", response_model=None)
async def health(request: Request):
    server = getattr(request.app.state, "server", None)
    state = getattr(server, "state", None)
    healthy = state not in {ServerState.ERROR, ServerState.SHUTTING_DOWN}
    payload = {
        "healthy": healthy,
        "version": "1.0.0",
        "state": state.value if state is not None else None,
    }
    if healthy:
        return payload
    return JSONResponse(payload, status_code=503)


@router.get("/v1/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.status()
    response.request_id = request.state.request_id
    return response
