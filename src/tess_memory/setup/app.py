"""The setup web app: a JSON API and one page, on this machine only.

It listens on 127.0.0.1 and trusts nobody by default. Every API call must
carry the one-time token printed with the link (so another local user or
program cannot drive setup), the ``Host`` header must name this listener (so
a hostile web page cannot reach it by rebinding a domain name to 127.0.0.1),
and a request that changes anything must come from this page's own origin.
Keys arrive once, go straight to the keychain and are never returned, logged
or echoed in an error.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import agents, apply, detect, embeddings, guidance, keepcurrent, models, reembedding
from .jobs import FirstIngest
from .providers import ProviderFile, SetupError
from .vault import Vault

STATIC = Path(__file__).parent / "static"
MAX_BODY = 64 * 1024
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
           "Referrer-Policy": "no-referrer",
           "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"}


class _Fresh(StaticFiles):
    """Static files the browser checks again each time, so a page never runs
    against a script from an earlier version."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Setup:
    """Everything one setup session holds."""

    def __init__(self, home: Path, *, keychain: bool = True, user_home: Path | None = None, vault: Vault | None = None) -> None:
        self.home = Path(home).expanduser().resolve()
        self.user_home = Path(user_home) if user_home else Path.home()
        self.vault = vault or Vault(self.home, keychain=keychain)
        self.providers = ProviderFile(self.vault)
        self.chatgpt = models.ChatGPT(self.vault, self.providers)
        self.local_models = embeddings.LocalModels(self.vault)
        self.settings = self.home / apply.SETTINGS
        self.ingest = FirstIngest(self.settings)
        self.ingest.after = self._keep_current
        self.scheduler_error: dict[str, str] | None = None
        self.reembedding = reembedding.Reembedding(self.vault)
        self.token = secrets.token_urlsafe(32)
        self.on_quit: Any = None

    def _keep_current(self) -> None:
        """Once the first batch is in, hand the rest to the scheduler, so the
        history is read whether or not this page stays open."""
        if not self.vault.draft.get("keep_current"):
            return
        current = keepcurrent.status(self.home)
        if not current["supported"] or current["installed"]:
            return
        try:
            keepcurrent.install(self.home, self.settings)
            self.scheduler_error = None
        except SetupError as error:
            self.scheduler_error = {"code": error.code, "detail": error.detail}

    def agent_homes(self) -> list[dict[str, Any]]:
        """Every agent that can be given the memory: all the agent homes on
        this machine, plus any folder added by hand as a source. Which
        sources are read and which agents are connected are separate choices."""
        homes = detect.agent_homes(self.user_home)
        known = {item["home"] for item in homes}
        return homes + [source for source in self.chosen_sources() if source.get("home") not in known]

    def chosen_sources(self) -> list[dict[str, Any]]:
        saved = self.vault.draft.get("sources")
        if saved is not None:
            return saved
        return [{**item, "enabled": True} for item in detect.detect_sources(self.user_home)]

    async def state(self) -> dict[str, Any]:
        draft = self.vault.draft
        chatgpt_ok, chatgpt_reason = await self.chatgpt.available()
        connection = {k: v for k, v in (draft.get("model") or {}).items() if k != "auth"} or None
        embedding = {k: v for k, v in (draft.get("embedding") or {}).items() if k != "auth"} or None
        sources = self.chosen_sources()
        return {
            "home": str(self.home), "display_home": detect._tilde(str(self.home), self.user_home),
            "settings": str(self.settings), "existing_settings": self.settings.exists() and not self.vault.meta.get("applied"),
            "applied": self.vault.meta.get("applied"), "secrets": await self.vault.mode(),
            "sources": sources, "sources_saved": draft.get("sources") is not None,
            "model": {"connection": connection, "chatgpt": {"available": chatgpt_ok, "reason": chatgpt_reason,
                                                            "attempt": self.chatgpt.outcome, "disclosure": models.DISCLOSURE}},
            "embedding": {"choice": embedding, "local": self.local_models.catalogue(), "job": self.local_models.job,
                          "presets": embeddings.PRESETS},
            "agents": {"targets": agents.targets(self.agent_homes(), self.settings),
                       "entry": {"mcpServers": {agents.SERVER: agents.server_entry(self.settings)}},
                       "guidance": guidance.text("claude-code")},
            "ingest": self.ingest.status,
            "scheduler": {**keepcurrent.status(self.home), "wanted": bool(draft.get("keep_current", True)),
                          "error": self.scheduler_error},
            "reembed": {"pending": self.reembedding.info, "job": self.reembedding.job},
            "missing": apply.missing({**draft, "sources": sources if draft.get("sources") is not None else []}),
        }

    # -- actions ----------------------------------------------------------

    async def save_sources(self, body: dict[str, Any]) -> None:
        items = body.get("sources")
        if not isinstance(items, list) or len(items) > 64:
            raise SetupError("invalid_sources")
        known = {(item["type"], item["path"]): item for item in detect.detect_sources(self.user_home)}
        chosen, names = [], set()
        for item in items:
            if not isinstance(item, dict):
                raise SetupError("invalid_sources")
            key = (item.get("type"), item.get("path"))
            try:
                source = dict(known[key]) if key in known else detect.describe_folder(*key)
            except (ValueError, TypeError) as error:
                raise SetupError(str(error) if str(error) in ("unknown_source_type", "folder_not_found") else "invalid_sources") from None
            base, n = source["name"], 2
            while source["name"] in names:
                source["name"], n = f"{base}-{n}", n + 1
            names.add(source["name"])
            source["enabled"] = bool(item.get("enabled", True))
            chosen.append(source)
        if not any(source["enabled"] for source in chosen):
            raise SetupError("no_source_selected")
        self.vault.draft["sources"] = chosen
        self.vault.save()

    async def apply(self) -> dict[str, Any]:
        """Write the settings, unless the store must first be moved to another
        embedding model: then nothing is written and the re-embed is offered."""
        if self.vault.draft.get("sources") is None:
            raise SetupError("incomplete", "sources")
        gaps = apply.missing(self.vault.draft)
        if gaps:
            raise SetupError("incomplete", ", ".join(gaps))
        registry = await self.providers.save()
        waiting = await asyncio.to_thread(reembedding.pending, self.home, registry, self.vault.draft)
        self.reembedding.offer(waiting)
        if not waiting:
            return await self._write(registry)
        # Only the embedding model has to wait. Everything else chosen (sources,
        # provider, models) is written now, around the embedding the store runs on.
        kept = reembedding.current_embedding(self.home, self.providers.profiles)
        if kept is None:
            waiting["held"] = "everything"
            return {"reembed": waiting}
        waiting["held"] = "embedding"
        return {**apply.write(self.vault, registry, keep_embedding=kept), "reembed": waiting}

    async def _write(self, registry: Path) -> dict[str, Any]:
        result = apply.write(self.vault, registry)
        embeddings.prune(self.vault, self.providers)
        await self.providers.save()
        while self.vault.pending_forget:
            await self.vault.forget(self.vault.pending_forget.pop())
        return result

    async def start_reembed(self) -> None:
        registry = await self.providers.path()
        # Runs on the re-embed thread, once the store has switched.
        self.reembedding.start(registry, after=lambda: asyncio.run(self._write(registry)))


def create_app(setup: Setup, *, port: int) -> Starlette:
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{host}" for host in hosts}

    def guard(request: Request, *, api: bool) -> Response | None:
        if request.headers.get("host") not in hosts:
            return JSONResponse({"error": "wrong_host"}, status_code=421, headers=HEADERS)
        if not api:
            return None
        if not hmac.compare_digest(request.headers.get("x-setup-token", ""), setup.token):
            return JSONResponse({"error": "token_required"}, status_code=401, headers=HEADERS)
        origin = request.headers.get("origin")
        if request.method != "GET" and (origin is None or origin not in origins):
            return JSONResponse({"error": "wrong_origin"}, status_code=403, headers=HEADERS)
        return None

    def api(handler):
        async def endpoint(request: Request) -> Response:
            refused = guard(request, api=True)
            if refused is not None:
                return refused
            body: dict[str, Any] = {}
            if request.method == "POST":
                raw = await request.body()
                if len(raw) > MAX_BODY:
                    return JSONResponse({"error": "too_large"}, status_code=413, headers=HEADERS)
                try:
                    body = json.loads(raw) if raw else {}
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return JSONResponse({"error": "invalid_json"}, status_code=400, headers=HEADERS)
            try:
                result = await handler(body)
            except SetupError as error:
                return JSONResponse({"error": error.code, "detail": error.detail}, status_code=422, headers=HEADERS)
            payload = await setup.state()
            if result is not None:
                payload["result"] = result
            return JSONResponse(payload, headers=HEADERS)
        return endpoint

    async def state(_body):
        return None

    async def folder(body):
        try:
            return detect.describe_folder(body.get("type"), body.get("path") or "")
        except (ValueError, TypeError) as error:
            raise SetupError(str(error) if str(error) in ("unknown_source_type", "folder_not_found") else "folder_not_found") from None

    async def model_key(body):
        await models.connect_with_key(setup.vault, setup.providers, body.get("kind"), api_key=body.get("api_key"),
                                      base_url=body.get("base_url"))

    async def chatgpt_start(_body):
        return await setup.chatgpt.start()

    async def chatgpt_cancel(_body):
        await setup.chatgpt.cancel()

    async def model_choose(body):
        await models.choose_models(setup.vault, setup.providers, distill_model=body.get("distill_model"),
                                   search_model=body.get("search_model"))

    async def local_download(body):
        setup.local_models.start(body.get("name"))

    async def local_cancel(_body):
        setup.local_models.cancel()

    async def local_use(body):
        embeddings.choose_local(setup.vault, body.get("name"))

    async def hosted(body):
        await embeddings.connect_hosted(setup.vault, setup.providers, preset=body.get("preset"), base_url=body.get("base_url"),
                                        model=body.get("model"), api_key=body.get("api_key"))

    async def apply_settings(_body):
        return await setup.apply()

    async def reembed_start(_body):
        await setup.start_reembed()

    async def reembed_stop(_body):
        setup.reembedding.stop()

    async def reembed_cancel(_body):
        setup.reembedding.cancel()

    async def register(body):
        if not setup.vault.meta.get("applied"):
            raise SetupError("settings_not_written")
        return await asyncio.to_thread(agents.register, body.get("id"), setup.agent_homes(), setup.settings)

    async def ingest_start(body):
        if not setup.vault.meta.get("applied"):
            raise SetupError("settings_not_written")
        if "keep_current" in body:
            setup.vault.draft["keep_current"] = bool(body["keep_current"])
            setup.vault.save()
        setup.ingest.start(body.get("limit"))

    async def ingest_stop(_body):
        setup.ingest.stop()

    async def scheduler_install(_body):
        if not setup.vault.meta.get("applied"):
            raise SetupError("settings_not_written")
        await asyncio.to_thread(keepcurrent.install, setup.home, setup.settings)
        setup.scheduler_error = None
        setup.vault.draft["keep_current"] = True
        setup.vault.save()

    async def scheduler_remove(_body):
        await asyncio.to_thread(keepcurrent.remove, setup.home)
        setup.vault.draft["keep_current"] = False
        setup.vault.save()

    async def quit_setup(_body):
        await setup.chatgpt.cancel()
        if setup.on_quit:
            setup.on_quit()

    async def index(request: Request) -> Response:
        refused = guard(request, api=False)
        if refused is not None:
            return refused
        return Response((STATIC / "index.html").read_bytes(), media_type="text/html; charset=utf-8", headers=HEADERS)

    post = lambda path, handler: Route(path, api(handler), methods=["POST"])  # noqa: E731
    return Starlette(routes=[
        Route("/", index, methods=["GET"]),
        Route("/api/state", api(state), methods=["GET"]),
        post("/api/sources", setup.save_sources), post("/api/sources/folder", folder),
        post("/api/model/key", model_key), post("/api/model/chatgpt/start", chatgpt_start),
        post("/api/model/chatgpt/cancel", chatgpt_cancel), post("/api/model/choose", model_choose),
        post("/api/embedding/local/download", local_download), post("/api/embedding/local/cancel", local_cancel),
        post("/api/embedding/local/use", local_use), post("/api/embedding/hosted", hosted),
        post("/api/apply", apply_settings), post("/api/agents/register", register),
        post("/api/reembed/start", reembed_start), post("/api/reembed/stop", reembed_stop),
        post("/api/reembed/cancel", reembed_cancel),
        post("/api/ingest/start", ingest_start), post("/api/ingest/stop", ingest_stop),
        post("/api/scheduler/install", scheduler_install), post("/api/scheduler/remove", scheduler_remove),
        post("/api/quit", quit_setup),
        Mount("/static", _Fresh(directory=STATIC), name="static"),
    ])


def run(home: Path, *, port: int = 0, open_browser: bool = True, keychain: bool = True) -> int:
    """Serve setup until the person finishes or presses Ctrl+C."""
    import socket
    import sys
    import uvicorn

    setup = Setup(home, keychain=keychain)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", port))
    except OSError:
        listener.bind(("127.0.0.1", 0))
    actual = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(setup, port=actual), log_level="warning", access_log=False))
    setup.on_quit = lambda: setattr(server, "should_exit", True)
    url = f"http://127.0.0.1:{actual}/?token={setup.token}"
    print(f"Tess Memory setup is running for {setup.home}\n\n  {url}\n\nOpen that link in a browser on this machine. "
          "Press Ctrl+C to stop.", file=sys.stderr, flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        asyncio.run(server.serve(sockets=[listener]))
    except KeyboardInterrupt:
        pass
    return 0
