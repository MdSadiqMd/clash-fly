"""WebSocket server for the TanStack Start UI: streams game frames and brain state, takes run commands.

One game thread owns the Playwright session (the sync API is bound to its thread) and
serves a command queue; the websocket loop runs in another thread and only exchanges JSON.

server -> client
  {type: "status", running, seconds_left, session, message?}
  {type: "frame", jpg: <base64 jpeg>}
  {type: "step", action, reward, score, elapsed, kc_idx, scores, loss, steps}
  {type: "episode", ...EpisodeLog}
  {type: "history", episodes: [EpisodeLog...]}
client -> server
  {cmd: "run", seconds: 30, learning: true}   {cmd: "stop"}
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import websockets

from flyclash.brain.circuit import Circuit
from flyclash.brain.learn import Plastic
from flyclash.config import ASSETS, DATA
from flyclash.subway.agent import SUBWAY_DIR, SurfPolicy, run_episodes
from flyclash.subway.browser import LiveEnv, PokiSession

PLASTIC = SUBWAY_DIR / "plastic.npz"
CSV_PATH = SUBWAY_DIR / "live.csv"


class Hub:
    """Websocket fan-out plus inbound command queue."""

    def __init__(self, port: int):
        self.port, self.clients, self.commands = port, set(), queue.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_status: dict = {
            "type": "status",
            "running": False,
            "seconds_left": 0,
            "session": "starting",
        }

    def start(self) -> "Hub":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async def handler(ws):
            self.clients.add(ws)
            try:
                await ws.send(json.dumps(self.last_status))
                await ws.send(
                    json.dumps({"type": "history", "episodes": read_history()})
                )
                async for raw in ws:
                    try:
                        self.commands.put(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
            finally:
                self.clients.discard(ws)

        async with websockets.serve(
            handler, "localhost", self.port, max_size=4_000_000
        ):
            await asyncio.Future()

    def publish(self, msg: dict) -> None:
        if msg.get("type") == "status":
            self.last_status = msg
        if self.loop is None:
            return
        data = json.dumps(msg)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send(data), self.loop)


def read_history(limit: int = 50) -> list[dict]:
    if not CSV_PATH.exists():
        return []
    rows = list(csv.DictReader(open(CSV_PATH)))[-limit:]
    return [
        {k: (float(v) if k not in ("dead",) else v == "True") for k, v in r.items()}
        for r in rows
    ]


def encode_frame(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


class GameThread(threading.Thread):
    def __init__(self, hub: Hub, headless: bool, weights: Path, frame_every: int = 1):
        super().__init__(daemon=True)
        self.hub, self.headless, self.weights, self.frame_every = (
            hub,
            headless,
            weights,
            frame_every,
        )
        self.stop_flag = threading.Event()

    def status(
        self,
        running: bool,
        seconds_left: float,
        session: str = "ready",
        message: str | None = None,
    ) -> None:
        msg = {
            "type": "status",
            "running": running,
            "seconds_left": round(seconds_left, 1),
            "session": session,
        }
        if message:
            msg["message"] = message
        self.hub.publish(msg)

    def run(self) -> None:
        try:
            session = PokiSession(headless=self.headless).start()
        except Exception as e:  # keep serving status so the UI can show the error
            self.status(False, 0, "error", f"browser failed: {e}")
            return
        env = LiveEnv(
            session,
            self.weights,
            on_frame=lambda f: self.hub.publish(
                {"type": "frame", "jpg": encode_frame(f)}
            ),
        )
        circuit = (
            Circuit.load(DATA / "circuit.npz")
            if (DATA / "circuit.npz").exists()
            else Circuit.random()
        )
        self.status(False, 0, "ready")
        while True:
            try:
                cmd = self.hub.commands.get(timeout=0.5)
            except queue.Empty:
                self.hub.publish(
                    {"type": "frame", "jpg": encode_frame(session.frame())}
                )
                continue
            if cmd.get("cmd") != "run":
                continue
            # one RUN click = a play session of AT LEAST `seconds` of in-run time:
            # crashes auto-restart, and the session only stops between episodes
            seconds = float(cmd.get("seconds", 30))
            learning = bool(cmd.get("learning", True))
            plastic = (
                Plastic.load(PLASTIC, circuit) if learning else Plastic.fresh(circuit)
            )
            plastic.epsilon = 0.02
            policy = SurfPolicy(circuit, plastic, learning=learning)
            self.stop_flag.clear()
            played = [0.0]

            def on_step(msg: dict) -> None:
                msg["type"] = "step"
                self.hub.publish(msg)
                if msg["steps"] % self.frame_every == 0:
                    self.hub.publish(
                        {"type": "frame", "jpg": encode_frame(session.frame())}
                    )
                    self.status(True, max(seconds - played[0] - msg["elapsed"], 0))
                while not self.hub.commands.empty():
                    if self.hub.commands.get_nowait().get("cmd") == "stop":
                        self.stop_flag.set()

            try:
                from flyclash.subway.browser import in_run

                if in_run(session.frame()):
                    # a leftover run from the previous session would start us mid-flight
                    # at high speed; let it die so this session begins a fresh run
                    self.status(True, seconds, "ready", "clearing leftover run")
                    env.park()
                n_ep = 0
                while played[0] < seconds and not self.stop_flag.is_set():
                    self.status(
                        True,
                        max(seconds - played[0], 0),
                        "ready",
                        "restarting" if n_ep else "starting run",
                    )
                    logs = run_episodes(
                        env,
                        policy,
                        1,
                        tag="live",
                        # cap each run generously so a good live run is never truncated mid-play;
                        # the session still ends once total `played` reaches the target
                        max_seconds=max(seconds, 60.0),
                        on_step=on_step,
                        should_stop=self.stop_flag.is_set,
                    )
                    for log in logs:
                        played[0] += log.wall_s
                        n_ep += 1
                        self.hub.publish({"type": "episode", **asdict(log)})
                if learning:
                    plastic.save(PLASTIC)
                # park so the next click starts from the beginning of a fresh run
                self.status(False, 0, "ready", "parking")
                env.park()
                self.status(False, 0, "ready")
            except Exception as e:
                self.status(False, 0, "ready", f"run failed: {e}")


def serve(
    port: int = 8765,
    headless: bool = True,
    weights: Path = ASSETS / "yolo" / "subway.pt",
) -> None:
    hub = Hub(port).start()
    GameThread(hub, headless, weights).start()
    print(f"flysurf server on ws://localhost:{port}  (headless={headless})")
    while True:
        time.sleep(3600)
