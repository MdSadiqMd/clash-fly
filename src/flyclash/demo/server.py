"""Websocket broadcaster for the demo page plus a static file server for it."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from functools import partial
from pathlib import Path

import websockets

WEB = Path(__file__).parent / "web"


class Broadcaster:
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def start(cls, port: int = 8765) -> "Broadcaster":
        b = cls(port)
        threading.Thread(target=b._run, daemon=True).start()
        return b

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async def handler(ws):
            self.clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self.clients.discard(ws)

        async with websockets.serve(handler, "localhost", self.port):
            await asyncio.Future()

    def publish(self, msg: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(msg)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send(data), self.loop)


def serve_static(port: int = 8000) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    http.server.ThreadingHTTPServer(("localhost", port), handler).serve_forever()
