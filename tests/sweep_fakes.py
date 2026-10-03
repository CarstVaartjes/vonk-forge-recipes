"""A fake ``vonkctl`` and a fake gateway for the sweep tests.

``FakeFleet.runner`` has the ``Runner`` signature of ``spark_sweep.vonkctl``:
it takes the argv the sweep would run, simulates the Controller with a fake
clock, and returns the exit code and JSON document the real vonkctl prints.
It models what the sweep depends on: paginated libraries, download operations
(models shared between recipes), whole-fleet profiles (a load stops every run
that is not in the profile), review verdicts with memory fit, applications
with success and failure, and the fleet's loaded runs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SPARK_MEMORY = 130_663_231_488
GB = 10**9


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0
        self.hooks: list[Callable[[float], None]] = []
        self.sleeps = 0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += seconds
        for hook in list(self.hooks):
            hook(self.t)


@dataclass
class FakeModel:
    digest: str
    bytes: int = 20 * GB
    local: str = "not_cached"

    @property
    def selector(self) -> str:
        return f"hf/{self.digest}"


@dataclass
class FakeRecipe:
    slug: str
    digests: tuple[str, ...] = ("m1",)
    node_count: int = 1
    memory: int = 40 * GB
    engine: str = "vllm"
    adapter: str = "openai"
    port: int = 8000
    local: str = "not_cached"
    content: str = "c1"
    updated: str = "2026-10-01T00:00:00Z"
    load_seconds: float = 120.0
    fail_load: str | None = None  # status reason of a failed load
    fail_phase: str = "runtime-install"
    fail_download: str | None = None
    alias: str | None = None

    @property
    def key(self) -> str:
        return f"vonk-forge/{self.slug}"


@dataclass
class Gateway:
    """An OpenAI-compatible endpoint on loopback; aliases in ``broken`` answer 500."""

    broken: set[str] = field(default_factory=set)
    requests: list[tuple[str, str]] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    def start(self) -> str:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                return

            def _send(
                self, status: int, body: bytes, kind: str = "application/json"
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                alias = self.path.split("/")[1]
                gateway.requests.append(("GET", self.path))
                if alias in gateway.broken:
                    self._send(500, b'{"error":"boom"}')
                    return
                self._send(200, json.dumps({"data": [{"id": alias}]}).encode())

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                alias = self.path.split("/")[1]
                gateway.requests.append(("POST", self.path))
                if alias in gateway.broken:
                    self._send(500, b'{"error":"boom"}')
                    return
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(8):
                        chunk = {"choices": [{"delta": {"content": f"w{i} "}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    usage = {"choices": [], "usage": {"completion_tokens": 8}}
                    self.wfile.write(
                        f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode()
                    )
                    return
                reply = {
                    "model": alias,
                    "choices": [
                        {"message": {"content": "391"}, "finish_reason": "stop"}
                    ],
                }
                self._send(200, json.dumps(reply).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server = server
        threading.Thread(target=lambda: server.serve_forever(0.01), daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()


class FakeFleet:
    def __init__(
        self,
        clock: FakeClock,
        recipes: list[FakeRecipe],
        models: list[FakeModel],
        api_root: str = "http://127.0.0.1:1",
    ) -> None:
        self.clock = clock
        self.recipes = {r.key: r for r in recipes}
        self.models = {m.digest: m for m in models}
        self.api_root = api_root
        self.sparks = [("spk_a", "spark-a"), ("spk_b", "spark-b")]
        self.profiles: dict[int, dict[str, Any]] = {}
        self.apps: dict[str, dict[str, Any]] = {}
        self.ops: dict[str, dict[str, Any]] = {}
        self.by_request: dict[str, str] = {}
        self.runs: list[dict[str, Any]] = []
        self.calls: list[tuple[int | None, tuple[str, ...]]] = []
        self.call_times: list[float] = []
        self.download_seconds = 60.0
        self.page_size = 2
        self.next_id = 1
        self.reject_pins = False
        # One-shot injected errors: (command prefix, error code, detail), consumed on first match.
        self.faults: list[tuple[tuple[str, ...], str, str]] = []
        self.client_build: dict[str, Any] = {"version": "1.0", "source_sha": "a" * 40}
        self.accepted_version = "1.0"
        self.release_sha = "a" * 40  # the accepted Controller release
        # Scripted library trouble, consumed one library call at a time: "timeout", "cursor" or None.
        self.library_faults: list[str | None] = []
        # The library's local cache state lags what the NAS holds (it keeps saying not_cached).
        self.library_lag = False
        # Rows carry the library's assessment, as the real Controller's do when it has time.
        self.assessments = False

    # -- helpers -------------------------------------------------------------

    def _id(self, prefix: str) -> str:
        self.next_id += 1
        return f"{prefix}-{self.next_id:04d}"

    def writes(self, number: int) -> list[tuple[str, ...]]:
        return [
            c
            for p, c in self.calls
            if p == number
            and c[0] == "profile"
            and c[1] in ("add", "remove", "load", "cancel")
        ]

    def owner_load(self, number: int, duration: float) -> None:
        """The owner loads one of their profiles: it stops every run and takes ``duration`` seconds."""
        app_id = self._id("owner")
        self.apps[app_id] = {
            "id": app_id,
            "profile": number,
            "created": self.clock.now(),
            "duration": duration,
            "owner": True,
            "assign": [],
            "state": "running",
        }
        self.profiles.setdefault(number, self._empty())["latest"] = app_id
        self.runs.clear()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"revision": 0, "assignments": [], "latest": None}

    # -- world -----------------------------------------------------------------

    def _tick(self) -> None:
        now = self.clock.now()
        for op in self.ops.values():
            if (
                op["state"] in ("queued", "running")
                and now >= op["started"] + op["duration"]
            ):
                recipe = self.recipes[op["recipe"]]
                if recipe.fail_download:
                    op["state"] = "failed"
                    continue
                op["state"] = "succeeded"
                recipe.local = "cached"
                for digest in recipe.digests:
                    self.models[digest].local = "cached"
        for app in self.apps.values():
            if app["state"] not in ("queued", "running"):
                continue
            if app.get("owner"):
                if now >= app["created"] + app["duration"]:
                    app["state"] = "succeeded"
                continue
            pending = [a for a in app["assign"] if not a["resolved"]]
            for item in pending:
                if now >= item["at"]:
                    item["resolved"] = True
                    if item["fail"]:
                        self.runs = [
                            r for r in self.runs if r["alias"] != item["alias"]
                        ]
                        app["reason"] = item["fail"]
                        app["phase"] = item["phase"]
                    else:
                        for run in self.runs:
                            if run["alias"] == item["alias"]:
                                run["ready"] = True
            if all(a["resolved"] for a in app["assign"]):
                failed = any(a["fail"] for a in app["assign"])
                app["state"] = "failed" if failed else "succeeded"

    # -- the runner ----------------------------------------------------------------

    def runner(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        args = list(argv[1:])
        profile: int | None = None
        clean: list[str] = []
        i = 0
        while i < len(args):
            if args[i] in ("--json", "--no-input"):
                i += 1
            elif args[i] == "--profile":
                profile = int(args[i + 1])
                i += 2
            else:
                clean.append(args[i])
                i += 1
        self.calls.append((profile, tuple(clean)))
        self.call_times.append(self.clock.now())
        self._tick()
        for index, (prefix, code, detail) in enumerate(self.faults):
            # Pin-profile edits are best effort and have their own tests: faults aim at the sweep.
            if profile != 13 and tuple(clean[: len(prefix)]) == prefix:
                del self.faults[index]
                _, document = self._error(code, detail)
                return 2, json.dumps(document), ""
        if clean[:1] == ["--version"]:
            return 0, json.dumps(self.client_build), ""
        if clean[:1] == ["update"]:
            drift = self.accepted_version != self.client_build["version"]
            return (
                0,
                json.dumps(
                    {
                        "current": self.client_build,
                        "accepted_version": self.accepted_version,
                        "accepted_source_sha": self.release_sha,
                        "update_available": drift,
                        "updated": False,
                    }
                ),
                "",
            )
        code, document = self._dispatch(profile, clean)
        return code, json.dumps(document), ""

    def _error(self, code: str, detail: str) -> tuple[int, dict[str, Any]]:
        return 2, {
            "error": detail,
            "error_type": "control_api",
            "code": code,
            "detail": detail,
        }

    def _dispatch(self, profile: int | None, a: list[str]) -> tuple[int, Any]:
        noun, verb = a[0], a[1] if len(a) > 1 else ""
        if noun == "fleet" and not verb:
            return 0, self._fleet()
        if noun == "fleet" and verb == "evidence":
            Path(a[a.index("--output") + 1]).write_text(
                json.dumps({"operation": a[2]}), encoding="utf-8"
            )
            return 0, {"output": a[a.index("--output") + 1]}
        if noun in ("recipe", "model") and verb == "library" and self.library_faults:
            fault = self.library_faults.pop(0)
            if fault == "timeout":
                return self._error(
                    "controller.transport_timeout",
                    "GET /api/recipe/library control API request failed [retry]",
                )
            if fault == "cursor":
                return self._error(
                    "controller.invalid_request",
                    "library cursor is invalid or the selection changed; restart without a cursor",
                )
        if noun == "recipe" and verb == "library":
            return 0, self._library(
                a, "recipes", self._recipe_row, list(self.recipes.values())
            )
        if noun == "model" and verb == "library":
            return 0, self._library(
                a, "models", self._model_row, list(self.models.values())
            )
        if noun == "recipe" and verb == "download":
            return self._download(a)
        if noun == "recipe" and verb == "progress":
            return self._op_progress(a)
        if noun == "profile":
            return self._profile(profile, verb, a[2:])
        return self._error("unsupported", " ".join(a))

    # -- libraries -------------------------------------------------------------

    def _library(
        self,
        a: list[str],
        name: str,
        row: Callable[[Any], dict[str, Any]],
        items: list[Any],
    ) -> dict[str, Any]:
        start = int(a[a.index("--cursor") + 1][1:]) if "--cursor" in a else 0
        page = items[start : start + self.page_size]
        more = start + self.page_size < len(items)
        return {
            name: [row(i) for i in page],
            "next_cursor": f"c{start + self.page_size}" if more else None,
            "library": {"commit": "lib-commit-1", "version": "2.1.0"},
        }

    def _recipe_row(self, r: FakeRecipe) -> dict[str, Any]:
        local = (
            "preparing"
            if any(
                o["recipe"] == r.key and o["state"] in ("queued", "running")
                for o in self.ops.values()
            )
            else ("not_cached" if self.library_lag else r.local)
        )
        return {
            **self._assessment(r),
            "selector": r.key,
            "identity": {
                "content_sha256": r.content,
                "recipe_revision_id": f"rev-{r.content}",
                "title": r.slug,
            },
            "node_count": r.node_count,
            "usage": ["chat"],
            "engine": r.engine,
            "creator": "test",
            "model_selectors": [self.models[d].selector for d in r.digests],
            "resources": {
                "memory_bytes": r.memory,
                "image_bytes": 10 * GB,
                "disk_bytes": 80 * GB,
            },
            "local": {"controller": local, "running_on": []},
            "updated_at": r.updated,
            "document": {
                "models": [
                    {"model": {"publisher": "hf", "slug": d, "content_sha256": d}}
                    for d in r.digests
                ],
                "interfaces": [
                    {
                        "adapter": r.adapter,
                        "port": r.port,
                        "model_aliases": [r.alias or r.slug],
                    }
                ],
                "topology": {
                    "node_count": r.node_count,
                    "roles": [{"resources": {"disk": {"artifact_bytes": 20 * GB}}}],
                },
                "execution": {"build": {"base_image": {"digest": f"base-{r.engine}"}}},
                "validation": {"serving": {"checks": []}},
                "runtime": {"engine": r.engine},
            },
        }

    def _assessment(self, r: FakeRecipe) -> dict[str, Any]:
        if not self.assessments:
            return {}
        foreign = any(run["alias"] not in self._ours() for run in self.runs)
        cache = "ready" if r.local == "cached" else "blocked"
        return {
            "assessment": {
                "cache": {"state": cache},
                # The advisory fit and readiness cannot hold while someone else's workload fills the Sparks.
                "fit": {"allowed": not foreign},
                "readiness": {
                    "state": "blocked" if foreign or cache != "ready" else "ready"
                },
            }
        }

    def _ours(self) -> set[str]:
        return {
            x["assignment_name"]
            for n in (10,)
            for x in self.profiles.get(n, {}).get("assignments", [])
        }

    def _model_row(self, m: FakeModel) -> dict[str, Any]:
        local = (
            "preparing"
            if any(
                m.digest in self.recipes[o["recipe"]].digests
                and o["state"] in ("queued", "running")
                for o in self.ops.values()
            )
            else m.local
        )
        return {
            "selector": m.selector,
            "identity": {"content_sha256": m.digest},
            "resources": {"disk_bytes": m.bytes},
            "local": {"controller": local},
        }

    # -- downloads -------------------------------------------------------------

    def _download(self, a: list[str]) -> tuple[int, Any]:
        key = a[2]
        request = a[a.index("--request-key") + 1]
        if request in self.by_request:
            return 0, self._op_doc(self.ops[self.by_request[request]])
        recipe = self.recipes[key]
        if recipe.fail_download == "policy":
            return self._error(
                "controller.unavailable",
                "dockerfile.heredoc_forbidden: Dockerfile heredocs are not accepted",
            )
        model_missing = any(self.models[d].local != "cached" for d in recipe.digests)
        op_id = self._id("op")
        self.ops[op_id] = {
            "id": op_id,
            "recipe": key,
            "state": "running",
            "started": self.clock.now(),
            "duration": self.download_seconds if model_missing else 5.0,
            "request": request,
            "bytes": sum(self.models[d].bytes for d in recipe.digests),
        }
        self.by_request[request] = op_id
        return 0, self._op_doc(self.ops[op_id])

    def _op_doc(self, op: dict[str, Any]) -> dict[str, Any]:
        recipe = self.recipes[op["recipe"]]
        elapsed = max(0.0, self.clock.now() - op["started"])
        done = (
            op["bytes"]
            if op["state"] == "succeeded"
            else int(op["bytes"] * min(elapsed / op["duration"], 0.99))
        )
        doc: dict[str, Any] = {
            "id": op["id"],
            "state": op["state"],
            "progress": {
                "completed_bytes": done,
                "total_bytes": op["bytes"],
                "smoothed_bytes_per_second": op["bytes"] / op["duration"]
                if op["state"] == "running"
                else 0,
            },
            "children": [],
        }
        if op["state"] == "failed":
            doc["failure"] = {
                "code": "model.download_failed",
                "detail": recipe.fail_download,
                "recovery_actions": [],
            }
            doc["children"] = [{"kind": "model-cache", "state": "failed"}]
        return doc

    def _op_progress(self, a: list[str]) -> tuple[int, Any]:
        if "--request-key" in a:
            op_id = self.by_request.get(a[a.index("--request-key") + 1])
        else:
            op_id = a[2]
        if op_id not in self.ops:
            return self._error("not_found", "unknown operation")
        return 0, self._op_doc(self.ops[op_id])

    # -- fleet ---------------------------------------------------------------------

    def _fleet(self) -> dict[str, Any]:
        nodes = []
        for node_id, name in self.sparks:
            loaded = [
                {
                    "alias": run["alias"],
                    "run_id": run["run_id"],
                    "healthy": run["ready"],
                    "route_state": "published" if run["ready"] else "pending",
                    "run_state": "running" if run["ready"] else "starting",
                }
                for run in self.runs
                if node_id in run["node_ids"]
            ]
            nodes.append(
                {
                    "id": node_id,
                    "display_name": name,
                    "connection": {"online_state": "online"},
                    "inventory": {
                        "host_memory_total_bytes": SPARK_MEMORY,
                        "disk_free_bytes": 10**12,
                    },
                    "loaded": loaded,
                }
            )
        return {"nodes": nodes}

    # -- profiles --------------------------------------------------------------------

    def _profile(self, number: int | None, verb: str, a: list[str]) -> tuple[int, Any]:
        n = number or 1
        data = self.profiles.setdefault(n, self._empty())
        if verb == "add" and n == 13 and self.reject_pins:
            return self._error("profile.refused", "assignment limit reached")
        if verb == "add":
            key, alias = a[0], a[a.index("--as") + 1]
            names = [a[i + 1] for i, x in enumerate(a) if x == "--spark"]
            data["assignments"] = [
                x for x in data["assignments"] if x["assignment_name"] != alias
            ]
            data["assignments"].append(
                {
                    "recipe_selector": key,
                    "spark_ids": names,
                    "assignment_name": alias,
                    "desired_state": a[a.index("--state") + 1],
                }
            )
            data["revision"] += 1
            return 0, {"revision": data["revision"]}
        if verb == "remove":
            before = len(data["assignments"])
            data["assignments"] = [
                x for x in data["assignments"] if x["assignment_name"] != a[0]
            ]
            if len(data["assignments"]) == before:
                return self._error(
                    "selector.absent", "assignment is absent or ambiguous"
                )
            data["revision"] += 1
            return 0, {"revision": data["revision"]}
        if verb == "export":
            return 0, {
                "assignments": data["assignments"],
                "name": f"profile {n}",
                "labels": data.get("labels", {}),
            }
        if verb == "configure":
            for item in a:
                if "=" in item and not item.startswith("-"):
                    name, _, value = item.partition("=")
                    data.setdefault("labels", {})[name] = value
            data["revision"] += 1
            return 0, {"revision": data["revision"]}
        if verb == "endpoint":
            alias = a[0] if a else ""
            return 0, {
                "assignments": [
                    {
                        "alias": x["assignment_name"],
                        "state": "published",
                        "endpoint": {
                            "alias": x["assignment_name"],
                            "api_base": f"{self.api_root}/{x['assignment_name']}/v1",
                        },
                    }
                    for x in data["assignments"]
                    if x["assignment_name"] == alias or not alias
                ]
            }
        if verb == "progress":
            return self._app_progress(data, a)
        if verb == "cancel":
            app = self.apps.get(a[0])
            if app and app["state"] in ("queued", "running"):
                app["state"] = "cancelled"
                app["cancel"] = True
                stopping = {
                    item["alias"] for item in app["assign"] if not item["resolved"]
                }
                self.runs = [r for r in self.runs if r["alias"] not in stopping]
            return 0, self._app_doc(app) if app else {}
        if verb == "load":
            return self._load(n, data, a)
        return self._error("unsupported", f"profile {verb}")

    def _desired_by_spark(
        self, data: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for x in data["assignments"]:
            if x["desired_state"] != "running":
                continue
            for name in x["spark_ids"]:
                out.setdefault(name, []).append(x)
        return out

    def _load(self, n: int, data: dict[str, Any], a: list[str]) -> tuple[int, Any]:
        by_spark = self._desired_by_spark(data)
        running = {r["alias"]: r for r in self.runs}
        names = {name: nid for nid, name in self.sparks}
        blocked: dict[str, list[dict[str, str]]] = {}
        for spark, items in by_spark.items():
            total = (
                sum(self.recipes[i["recipe_selector"]].memory for i in items) + 8 * GB
            )
            if total > SPARK_MEMORY:
                for i in items:
                    if i["assignment_name"] not in running:
                        blocked.setdefault(i["assignment_name"], []).append(
                            {
                                "code": "run-switch.resource.insufficient_capacity",
                                "detail": f"{spark} lacks memory",
                            }
                        )
        desired_aliases = {
            x["assignment_name"]
            for x in data["assignments"]
            if x["desired_state"] == "running"
        }
        if "--review" in a:
            doc = {
                "allowed": not blocked,
                "waits_for_preparation": False,
                "effects": {
                    "runs": [
                        {
                            "alias": r["alias"],
                            "run_id": r["run_id"],
                            "node_ids": r["node_ids"],
                            "action": "keep"
                            if r["alias"] in desired_aliases
                            else "stop",
                        }
                        for r in self.runs
                    ]
                },
                "admission_decisions": [
                    {"alias": alias, "allowed": False, "blockers": found}
                    for alias, found in blocked.items()
                ],
                "reasons": [],
            }
            return (0 if not blocked else 2), doc
        request = a[a.index("--request-key") + 1]
        if request in self.by_request:
            return 0, self._app_doc(self.apps[self.by_request[request]])
        self.runs = [r for r in self.runs if r["alias"] in desired_aliases]
        app_id = self._id("app")
        assign = []
        for x in data["assignments"]:
            alias = x["assignment_name"]
            if x["desired_state"] != "running" or alias in running:
                continue
            recipe = self.recipes[x["recipe_selector"]]
            self.runs.append(
                {
                    "alias": alias,
                    "recipe": recipe.key,
                    "run_id": self._id("run"),
                    "node_ids": [names[s] for s in x["spark_ids"]],
                    "ready": False,
                }
            )
            assign.append(
                {
                    "alias": alias,
                    "at": self.clock.now() + recipe.load_seconds,
                    "fail": recipe.fail_load,
                    "phase": recipe.fail_phase,
                    "resolved": False,
                }
            )
        self.apps[app_id] = {
            "id": app_id,
            "profile": n,
            "created": self.clock.now(),
            "assign": assign,
            "state": "running",
            "request_key": request,
        }
        data["latest"] = app_id
        self.by_request[request] = app_id
        if not assign:
            self.apps[app_id]["state"] = "succeeded"
        return 0, self._app_doc(self.apps[app_id])

    def _app_doc(self, app: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": app["id"],
            "request_key": app.get("request_key", ""),
            "state": app["state"],
            "status_reason": app.get("reason"),
            "current_operation_id": app["id"],
            "progress": {
                "child_progress": {"phase": app.get("phase")},
                "switch_adapter": {"assignment_failures": []},
            },
            **(
                {"cancellation": {"cause": app.get("cause", "operator")}}
                if app.get("cancel")
                else {}
            ),
        }

    def _app_progress(self, data: dict[str, Any], a: list[str]) -> tuple[int, Any]:
        if "--application" in a:
            app = self.apps.get(a[a.index("--application") + 1])
        elif "--request-key" in a:
            app = self.apps.get(
                self.by_request.get(a[a.index("--request-key") + 1], "")
            )
        else:
            app = self.apps.get(data.get("latest") or "")
        if app is None:
            return self._error("not_found", "no application")
        return 0, self._app_doc(app)


# -- harness -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def make_definitions(recipes: list[FakeRecipe]) -> Any:
    """Definitions for fake recipes: the repository's real M0 and A391 service cases."""
    from spark_sweep.definitions import Definitions

    shared = json.loads(
        (ROOT / "qualification" / "shared.json").read_text(encoding="utf-8")
    )
    document = {
        "fixtures": shared["fixtures"],
        "service_case_templates": shared["service_case_templates"],
        "service_recipes": {
            r.key: {"alias": r.alias or r.slug, "smoke_cases": ["M0", "A391"]}
            for r in recipes
            if r.adapter == "openai"
        },
        "recipes": {},
    }
    return Definitions(document, ROOT / "qualification")


def make_sweep(
    tmp_path: Path,
    recipes: list[FakeRecipe],
    models: list[FakeModel],
    *,
    gateway: Gateway | None = None,
    fleet: FakeFleet | None = None,
    **overrides: Any,
) -> tuple[Any, FakeFleet, FakeClock]:
    from spark_sweep.prefetch import PrefetchConfig
    from spark_sweep.run import Sweep, SweepConfig
    from spark_sweep.state import ResultsLog, State
    from spark_sweep.vonkctl import Vonkctl

    if fleet is None:
        clock = FakeClock()
        root = gateway.start() if gateway else "http://127.0.0.1:1"
        fleet = FakeFleet(clock, recipes, models, root)
    clock = fleet.clock
    prefetch = overrides.pop("prefetch", PrefetchConfig(pin_profile=13))
    settings: dict[str, Any] = {
        "poll_seconds": 10.0,
        "catalog_seconds": 10.0,
        "status_seconds": 30.0,
        "owner_poll_seconds": 10.0,
        "readiness_grace": 60.0,
        "retry_delay": 30.0,
        "defer_delay": 60.0,
        "boost": (),
        **overrides,
    }
    config = SweepConfig(
        state_dir=tmp_path, authority_id="test-authority", prefetch=prefetch, **settings
    )
    sweep = Sweep(
        config,
        Vonkctl("vonkctl", runner=fleet.runner),
        State.load(tmp_path / "state.json", clock.now),
        ResultsLog(tmp_path / "results.jsonl", "test-authority"),
        make_definitions(recipes),
        clock,
    )
    return sweep, fleet, clock
