#!/usr/bin/env python3
"""Start or stop SGLang torch profiler on all HTTP workers behind a router."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any
from urllib import request


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _worker_urls(router_url: str) -> list[str]:
    data = _json_request(router_url.rstrip("/") + "/workers")
    workers = data.get("workers")
    if not isinstance(workers, list):
        raise RuntimeError(f"invalid /workers response: {data!r}")
    urls: list[str] = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        if not worker.get("is_healthy"):
            continue
        if str(worker.get("connection_mode") or "").lower() != "http":
            continue
        url = str(worker.get("url") or "").rstrip("/")
        if url:
            urls.append(url.replace("0.0.0.0", "127.0.0.1"))
    return sorted(set(urls))


def _start(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(run_dir),
        "num_steps": args.num_steps,
        "activities": args.activities,
        "profile_by_stage": args.profile_by_stage,
        "with_stack": args.with_stack,
        "record_shapes": args.record_shapes,
        "merge_profiles": args.merge_profiles,
        "profile_prefix": args.profile_prefix,
    }
    if args.profile_stages:
        payload["profile_stages"] = args.profile_stages
    if args.start_step is not None:
        payload["start_step"] = args.start_step

    print(f"profile output_dir={run_dir}")
    for url in _worker_urls(args.router_url):
        print(f"start {url}")
        print(json.dumps(_json_request(url + "/start_profile", payload), indent=2))


def _stop(args: argparse.Namespace) -> None:
    for url in _worker_urls(args.router_url):
        print(f"stop {url}")
        print(json.dumps(_json_request(url + "/stop_profile", {}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["start", "stop"])
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="/root/Dressage/profiles/sglang")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--activities", nargs="+", default=["CPU", "GPU"])
    parser.add_argument("--profile-by-stage", action="store_true", default=True)
    parser.add_argument("--no-profile-by-stage", dest="profile_by_stage", action="store_false")
    parser.add_argument("--profile-stages", nargs="*", default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--with-stack", action="store_true", default=False)
    parser.add_argument("--record-shapes", action="store_true", default=False)
    parser.add_argument("--merge-profiles", action="store_true", default=False)
    parser.add_argument("--profile-prefix", default="rollout")
    args = parser.parse_args()

    if args.mode == "start":
        _start(args)
    else:
        _stop(args)


if __name__ == "__main__":
    main()
