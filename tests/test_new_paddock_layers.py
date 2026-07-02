from __future__ import annotations

import asyncio
from typing import Any

import httpx

from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.paddock.whitebox.paddock import WhiteboxToolPaddock
from dressage.sandbox.types import CommandResult, SandboxEndpoint, SandboxLease, SandboxSpec


class FakeProvider:
    name = "local_bwrap"

    def __init__(self) -> None:
        self.created: list[SandboxSpec] = []
        self.terminated: list[SandboxLease] = []
        self.commands: list[tuple[SandboxLease, str | list[str], dict[str, Any]]] = []
        self.files: dict[str, str] = {}

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        self.created.append(spec)
        paddock_mode = spec.metadata.get("paddock_mode")
        lease_index = len(self.created)
        lease = SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=f"lease-{spec.trajectory_id}-{lease_index}",
            capabilities=(
                {"command", "file", "public_url"}
                if paddock_mode == "blackbox"
                else {"command", "file"}
            ),
            metadata={"node_ip": "10.0.0.12", "port": 31000},
        )
        if paddock_mode == "blackbox":
            lease.endpoints["blackbox"] = SandboxEndpoint(
                url="http://sandbox.test",
                headers={"x-test": "1"},
            )
        return lease

    async def terminate(self, lease):
        self.terminated.append(lease)
        return {"terminated": True}

    async def get_public_url(self, lease, *, port, service_name=None):
        return lease.endpoints[service_name or "blackbox"]

    async def run_command(self, lease, command, **kwargs):
        self.commands.append((lease, command, kwargs))
        return CommandResult(cmd=command, stdout="ran\n", stderr="", returncode=0)

    async def read_file(self, lease, path, *, encoding="utf-8", max_bytes=None):
        value = self.files.get(path, "")
        return value if max_bytes is None else value[:max_bytes]

    async def write_file(self, lease, path, content, *, encoding="utf-8", append=False):
        text = content.decode(encoding or "utf-8") if isinstance(content, bytes) else str(content)
        self.files[path] = self.files.get(path, "") + text if append else text
        return {"path": path, "bytes": len(text)}


def test_blackbox_agent_paddock_uses_provider_and_blackbox_client():
    asyncio.run(_run_blackbox_agent_paddock_uses_provider_and_blackbox_client())


async def _run_blackbox_agent_paddock_uses_provider_and_blackbox_client():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/rollout/register":
            assert request.headers["x-test"] == "1"
            body = request.content.decode()
            assert "bound_instance_id" in body
            assert "runtime_root" not in body
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/v1/sessions/traj-1/messages":
            return httpx.Response(200, json={"response": "done"})
        if request.url.path == "/v1/sessions/traj-1/execute_cmd":
            return httpx.Response(200, json={"returncode": 0, "stdout": "Python\n"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        blackbox_client=None,
        wait_health=False,
    )
    # Replace the internally created client with one backed by MockTransport.
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    state = await paddock.init("traj-1")
    assert state.sandbox_url == "http://sandbox.test"
    assert provider.created[0].services[0].name == "blackbox"
    assert provider.created[0].metadata == {"paddock_mode": "blackbox"}

    assert await paddock.register_agent(state, instance_id="inst", session_id="traj-1") == {"ok": True}
    assert (await paddock.call_agent(state, session_id="traj-1", messages=[]))["response"] == "done"
    assert (await paddock.execute_cmd(state, session_id="traj-1", cmd="python -V"))["returncode"] == 0

    await client.aclose()


def test_blackbox_register_recycles_slot_after_retryable_failure(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_REGISTER_RECYCLE_ATTEMPTS", "1")
    asyncio.run(_run_blackbox_register_recycles_slot_after_retryable_failure())


async def _run_blackbox_register_recycles_slot_after_retryable_failure():
    register_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal register_attempts
        if request.url.path == "/v1/rollout/register":
            register_attempts += 1
            if register_attempts == 1:
                return httpx.Response(502, json={"detail": "backend_error"})
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    state = await paddock.init("traj-1")
    assert await paddock.register_agent(state, instance_id="inst", session_id="traj-1") == {"ok": True}

    assert register_attempts == 2
    assert len(provider.created) == 2
    assert len(provider.terminated) == 1
    assert provider.terminated[0].sandbox_id == "lease-traj-1-1"
    assert paddock._states["traj-1"].sandbox_id == "lease-traj-1-2"

    await client.aclose()


def test_blackbox_register_recycles_slot_after_read_error(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_REGISTER_RECYCLE_ATTEMPTS", "1")
    asyncio.run(_run_blackbox_register_recycles_slot_after_read_error())


async def _run_blackbox_register_recycles_slot_after_read_error():
    register_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal register_attempts
        if request.url.path == "/v1/rollout/register":
            register_attempts += 1
            if register_attempts == 1:
                raise httpx.ReadError("server dropped register response", request=request)
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    state = await paddock.init("traj-1")
    assert await paddock.register_agent(state, instance_id="inst", session_id="traj-1") == {"ok": True}

    assert register_attempts == 2
    assert len(provider.created) == 2
    assert len(provider.terminated) == 1
    assert provider.terminated[0].sandbox_id == "lease-traj-1-1"
    assert paddock._states["traj-1"].sandbox_id == "lease-traj-1-2"

    await client.aclose()



def test_blackbox_terminate_backgrounds_abort_and_release(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BACKGROUND_TERMINATE", "1")
    asyncio.run(_run_blackbox_terminate_backgrounds_abort_and_release())


async def _run_blackbox_terminate_backgrounds_abort_and_release():
    abort_started = asyncio.Event()
    allow_abort = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions/traj-1/abort":
            abort_started.set()
            await allow_abort.wait()
            return httpx.Response(200, json={"state": "aborted"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    await paddock.init("traj-1")
    result = await paddock.terminate("traj-1")

    assert result["terminated"] is True
    assert result["cleanup_background"] is True
    assert result["release_queued"] is True
    assert "traj-1" not in paddock._states
    assert "traj-1" not in paddock._leases
    assert "traj-1" not in paddock._specs

    await asyncio.wait_for(abort_started.wait(), timeout=1.0)
    assert provider.terminated == []

    allow_abort.set()
    await paddock.drain_cleanup_tasks()
    assert len(provider.terminated) == 1
    assert provider.terminated[0].sandbox_id == "lease-traj-1-1"

    await client.aclose()


def test_blackbox_terminate_skips_abort_for_clean_session(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BACKGROUND_TERMINATE", "0")
    asyncio.run(_run_blackbox_terminate_skips_abort_for_clean_session())


async def _run_blackbox_terminate_skips_abort_for_clean_session():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    await paddock.init("traj-1")
    result = await paddock.terminate("traj-1", {"blackbox_skip_abort": True})

    assert result == {"terminated": True}
    assert len(provider.terminated) == 1
    assert provider.terminated[0].sandbox_id == "lease-traj-1-1"

    await client.aclose()


def test_blackbox_terminate_can_run_synchronously(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BACKGROUND_TERMINATE", "0")
    asyncio.run(_run_blackbox_terminate_can_run_synchronously())


async def _run_blackbox_terminate_can_run_synchronously():
    abort_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal abort_calls
        if request.url.path == "/v1/sessions/traj-1/abort":
            abort_calls += 1
            return httpx.Response(200, json={"state": "aborted"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FakeProvider()
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    from dressage.paddock.blackbox.client import BlackboxServerClient

    paddock._client = BlackboxServerClient(client=client)

    await paddock.init("traj-1")
    result = await paddock.terminate("traj-1")

    assert result == {"terminated": True}
    assert abort_calls == 1
    assert len(provider.terminated) == 1
    assert paddock._cleanup_tasks == set()

    await client.aclose()

def test_whitebox_tool_paddock_maps_tools_to_provider():
    asyncio.run(_run_whitebox_tool_paddock_maps_tools_to_provider())


async def _run_whitebox_tool_paddock_maps_tools_to_provider():
    provider = FakeProvider()
    paddock = WhiteboxToolPaddock(provider=provider)
    await paddock.init("traj-1")

    assert provider.created[0].services == ()
    assert provider.created[0].metadata == {"paddock_mode": "whitebox"}

    text, meta = await paddock.tool_call("traj-1", "shell.exec", {"cmd": "echo hi"})
    assert text == "ran\n"
    assert meta["returncode"] == 0

    await paddock.tool_call("traj-1", "file.write", {"path": "/workspace/a.txt", "content": "abc"})
    text, meta = await paddock.tool_call("traj-1", "file.read", {"path": "/workspace/a.txt"})
    assert text == "abc"
    assert meta["chars"] == 3
