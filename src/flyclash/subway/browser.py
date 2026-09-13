"""Playwright session on the official HTML5 Subway Surfers at poki.com, and the live Env.

Keys: ArrowLeft / ArrowRight move, ArrowUp jumps, ArrowDown rolls, Space hoverboard.
Frames come from CDP Page.captureScreenshot clipped to the game canvas (~50 fps).

Screens seen in Poki's build (recorded 2026-09-13, headless Chromium 1280x800):
  intro (character) -> run starts on any arrow key -> tutorial pauses at each obstacle
  with "Press Arrow Key Up/Down/Left/Right" until that key is pressed -> free play ->
  crash: dimmed "Save me!" dialog with countdown -> "New High Score / Press Space to
  continue" -> menu. The blue pause button (top-left) and the score HUD (top-right)
  are present only while playing; that is the alive signal.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from flyclash.config import DATA
from flyclash.game import ui
from flyclash.game.base_reader import Detection
from flyclash.subway.classes import canonical_class
from flyclash.subway.reader import Geometry, vectorize

URL = "https://poki.com/en/g/subway-surfers"
KEYS = {
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "jump": "ArrowUp",
    "roll": "ArrowDown",
    "none": None,
}
PROMPT_KEYS = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "space": "Space",
}
PROFILE = (
    DATA / "subway" / "profile"
)  # persistent browser profile: tutorial and progress survive restarts
PAUSE_BOX = (0.01, 0.02, 0.07, 0.13)
SCORE_BOX = (0.86, 0.02, 0.995, 0.10)
CENTER_BOX = (0.25, 0.35, 0.75, 0.65)


class PokiSession:
    def __init__(
        self,
        headless: bool = False,
        scale: float = 0.5,
        viewport=(1280, 800),
        profile: Path = PROFILE,
    ):
        self.headless, self.scale, self.viewport, self.profile = (
            headless,
            scale,
            viewport,
            profile,
        )
        self.canvas_box = None

    def start(self) -> "PokiSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self.profile.mkdir(parents=True, exist_ok=True)
        self.context = self._pw.chromium.launch_persistent_context(
            str(self.profile),
            headless=self.headless,
            viewport={"width": self.viewport[0], "height": self.viewport[1]},
            args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"],
        )
        self.page = (
            self.context.pages[0] if self.context.pages else self.context.new_page()
        )
        self.page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        self.cdp = self.page.context.new_cdp_session(self.page)
        self._find_canvas()
        return self

    def _find_canvas(self, timeout: float = 90.0) -> None:
        """Poki loads the game inside an iframe from poki-gdn.com; wait for its canvas."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            for frame in self.page.frames:
                try:
                    box = frame.locator("canvas").first.bounding_box(timeout=1000)
                except Exception:
                    box = None
                if box and box["width"] > 300 and box["height"] > 300:
                    self.canvas_box, self.game_frame = box, frame
                    return
            time.sleep(1.0)
        raise RuntimeError(
            "game canvas not found; is the page blocked or still loading?"
        )

    def frame(self) -> np.ndarray:
        b = self.canvas_box
        res = self.cdp.send(
            "Page.captureScreenshot",
            {
                "format": "jpeg",
                "quality": 80,
                "clip": {
                    "x": b["x"],
                    "y": b["y"],
                    "width": b["width"],
                    "height": b["height"],
                    "scale": self.scale,
                },
            },
        )
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(res["data"]), np.uint8), cv2.IMREAD_COLOR
        )

    def click(self, xf: float = 0.5, yf: float = 0.5) -> None:
        b = self.canvas_box
        self.page.mouse.click(b["x"] + b["width"] * xf, b["y"] + b["height"] * yf)

    def focus(self) -> None:
        self.click(0.5, 0.5)

    def press(self, action: str) -> None:
        key = KEYS.get(action, action)
        if key:
            self.page.keyboard.press(key)

    def try_dom_close(self) -> bool:
        """Poki overlays (sign-in nag, consent) are DOM, not canvas: close them by element."""
        for sel in (
            "button[aria-label*='lose' i]",
            "button:has-text('×')",
            "button:has-text('✕')",
            "[class*='close' i]:visible",
        ):
            for fr in self.page.frames:
                try:
                    loc = fr.locator(sel).first
                    if loc.is_visible(timeout=300):
                        loc.click(timeout=800)
                        return True
                except Exception:
                    continue
        return False

    def close(self) -> None:
        self.context.close()
        self._pw.stop()


def alive(frame: np.ndarray) -> bool:
    """In play: blue pause button top-left (0.42 blue fraction) AND the dark score box with white
    digits top-right (dark >= 0.5, white ~0.2). The main menu has a blue icon but no score box;
    the run intro and the end screens have neither. Measured on recorded frames, see docs/subway-plan.md."""
    hsv = cv2.cvtColor(ui.crop(frame, PAUSE_BOX), cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[..., 0] > 95)
        & (hsv[..., 0] < 125)
        & (hsv[..., 1] > 120)
        & (hsv[..., 2] > 120)
    ).mean()
    return bool(blue > 0.15)


def in_run(frame: np.ndarray) -> bool:
    """alive() plus readable score digits; excludes the main menu, whose blue gear icon sits where
    the pause button is. OCR costs ~300 ms, so this is for reset, not the step loop."""
    return alive(frame) and read_score(frame) is not None


def read_score(frame: np.ndarray) -> int | None:
    try:
        return ui.ocr_number(ui.crop(frame, SCORE_BOX))
    except Exception:
        return None


def dialog_showing(frame: np.ndarray) -> bool:
    """The 'Save me!', 'Need hoverboards?' or Poki 'Sign in' dialog: a bright card in the centre."""
    c = ui.crop(frame, CENTER_BOX)
    return bool((c.min(axis=2) > 225).mean() > 0.25)


def dialog_bbox(frame: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fractional bbox of the biggest bright card on screen, for aiming at its close button.
    Different dialogs put the X in different corners (hoverboard: top-left; Poki sign-in: top-right)."""
    bright = (frame.min(axis=2) > 225).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return None
    i = int(np.argmax(stats[1:, 4])) + 1
    x, y, w, h, area = stats[i]
    fh, fw = frame.shape[:2]
    if area < 0.03 * fh * fw:
        return None
    return x / fw, y / fh, (x + w) / fw, (y + h) / fh


PLAY_BUTTON_BOX = (
    0.51,
    0.87,
    0.67,
    0.98,
)  # green PLAY on the leaderboard / results screen
BOTTOM_TEXT_BOX = (
    0.25,
    0.86,
    0.75,
    0.99,
)  # "Press Space to continue" on the high-score screen


def classify(frame: np.ndarray) -> str:
    """play | dialog | menu | results | space | other. Thresholds measured on recorded frames."""
    if alive(frame):
        return "play" if read_score(frame) is not None else "menu"
    if dialog_showing(frame):
        return "dialog"
    g = cv2.cvtColor(ui.crop(frame, PLAY_BUTTON_BOX), cv2.COLOR_BGR2HSV)
    green = (
        (g[..., 0] > 45) & (g[..., 0] < 85) & (g[..., 1] > 120) & (g[..., 2] > 120)
    ).mean()
    if green > 0.3:
        return "results"
    try:
        if "space" in ui.ocr_text(ui.crop(frame, BOTTOM_TEXT_BOX)).lower():
            return "space"
    except Exception:
        pass
    return "other"


def tutorial_key(frame: np.ndarray) -> str | None:
    """Key named by a tutorial prompt in the centre of the screen, if any."""
    try:
        txt = ui.ocr_text(ui.crop(frame, CENTER_BOX)).lower()
    except Exception:
        return None
    m = re.search(r"\b(up|down|left|right|space)\b", txt)
    return PROMPT_KEYS[m.group(1)] if m else None


def occupancy_detections(frame: np.ndarray, geo: Geometry) -> list[Detection]:
    """Train/blocker vision without ML: a lane cell that shows little of the track bed is
    covered by something. Track pixels = tan/brown sleepers or grey rails; measured on
    recorded frames, open cells read 0.7-1.0 track fraction, cells behind a train, wall
    or crate read 0.1-0.35. Catches train fronts (which defeat texture cues) too."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tan = (
        (hsv[..., 0] >= 8)
        & (hsv[..., 0] <= 30)
        & (hsv[..., 1] >= 60)
        & (hsv[..., 2] >= 90)
    )
    grey = (hsv[..., 1] < 40) & (hsv[..., 2] >= 70) & (hsv[..., 2] <= 180)
    teal = (
        (hsv[..., 0] >= 78)
        & (hsv[..., 0] <= 105)
        & (hsv[..., 1] >= 40)
        & (hsv[..., 2] >= 60)
    )  # tunnel floor
    track = (tan | grey | teal).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    yf, xf = ys / h, xs / w
    t = (geo.near_y - yf) / max(geo.near_y - geo.far_y, 1e-6)
    depth = np.clip((t * 5).astype(int), 0, 4)
    width = geo.near_width + (geo.far_width - geo.near_width) * np.clip(t, 0, 1)
    rel = (xf - geo.center_x) / np.maximum(width, 1e-6) + 0.5
    lane = np.clip((rel * 3).astype(int), 0, 2)
    inside = (rel >= 0) & (rel < 1) & (yf < geo.near_y) & (yf > geo.far_y)
    dets: list[Detection] = []
    for d in range(5):
        frac = np.array(
            [
                track[inside & (depth == d) & (lane == ln)].mean()
                if (inside & (depth == d) & (lane == ln)).any()
                else np.nan
                for ln in range(3)
            ]
        )
        if not np.isfinite(frac).any() or np.nanmax(frac) < 0.65:
            continue  # whole row obscured (tunnel, dialog): no lane is readable, say nothing
        for ln in range(3):
            if np.isfinite(frac[ln]) and frac[ln] < 0.45:
                yc = geo.near_y - (d + 0.5) / 5 * (geo.near_y - geo.far_y)
                xc = geo.center_x + ((ln + 0.5) / 3 - 0.5) * geo.width_at(yc)
                dets.append(
                    Detection(
                        "train", xc * w, yc * h, min(0.5 + (0.45 - frac[ln]), 0.95)
                    )
                )
    return dets


def heuristic_detections(frame: np.ndarray) -> list[Detection]:
    """Fallback eyes until YOLO weights exist: the red-and-white striped barriers are found by
    colour (red pixels next to white pixels in the track area). Trains are not detected, so the
    fly still dies at the first train. ponytail: replace with the trained detector, see docs."""
    h, w = frame.shape[:2]
    roi = frame[int(0.35 * h) : int(0.95 * h), int(0.15 * w) : int(0.85 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 140, 120), (8, 255, 255)) | cv2.inRange(
        hsv, (172, 140, 120), (180, 255, 255)
    )
    white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    k = np.ones((7, 7), np.uint8)
    stripes = cv2.dilate(red, k) & cv2.dilate(
        white, k
    )  # red and white within a few pixels of each other
    stripes = cv2.morphologyEx(stripes, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(stripes)
    dets = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 40 or bw < 12:
            continue
        cx, cy = cents[i]
        fx, fy = (cx + 0.15 * w) / w, (cy + 0.35 * h) / h
        if fy > 0.68 and 0.36 < fx < 0.64:
            continue  # the player and the chasing guard (red-striped shirt) occupy this box
        # barriers sit low and wide; the near ones are big. Both barrier kinds look alike from behind, so call them low (jump).
        dets.append(
            Detection(
                "low_barrier",
                cx + int(0.15 * w),
                cy + int(0.35 * h),
                min(0.5 + area / 2000, 0.95),
            )
        )
    return dets


class LiveEnv:
    """Gym-like wrapper. Vision is sprite-template matching plus coin blobs (YOLO weights
    override when present). No OCR in the decision loop: it cost 300 ms exactly when the
    fly needed to dodge; the score is read once per episode from the last alive frame."""

    def __init__(
        self,
        session: PokiSession,
        weights: Path,
        geo: Geometry = Geometry(),
        collect_dir: Path | None = None,
        step_period: float = 0.08,
        debug_dir: Path | None = None,
        on_frame=None,
    ):
        self.s, self.weights, self.geo, self.collect_dir = (
            session,
            weights,
            geo,
            collect_dir,
        )
        self.on_frame = on_frame  # called with every frame grabbed during reset/park walks, so a UI can keep streaming; Playwright is single-threaded, so no separate pump thread is possible
        self.step_period = (
            step_period  # ~9-10 decisions/s with ~50 ms template matching
        )
        self.debug_dir = debug_dir  # when set: every frame + a JSON line with detections and the vector's lane/depth cells
        self.last_dets: list[Detection] = []
        self.model = None
        if Path(weights).exists():
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
        from flyclash.subway.templates import (
            TEMPLATE_DIR,
            build_templates,
            load_templates,
        )

        if not any(TEMPLATE_DIR.glob("*.png")):
            build_templates()
        self.templates = load_templates()
        self.n_frames = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self.lane, self.score, self.speed = 1, 0, 0.0
        self.dead_frames = 0
        self.run_start_t = time.monotonic()
        self.prev_frame: np.ndarray | None = None
        self.still_since: float | None = None
        self.coins_near = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from flyclash.subway.templates import coin_detections, match_obstacles

        if self.model is None:
            # templates: precise, fire near. occupancy: coarse "lane blocked" that fires at
            # range and in any lighting (the fly was dying to trains templates only saw at
            # impact). coins give it something to chase between hazards.
            obstacles = match_obstacles(
                frame, self.templates, self.geo
            ) + occupancy_detections(frame, self.geo)
            return obstacles + coin_detections(frame, self.geo)
        out = self.model.predict(frame, conf=0.35, verbose=False)[0]
        dets = []
        for box in out.boxes:
            name = canonical_class(out.names[int(box.cls)])
            if name is None:
                continue
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            dets.append(Detection(name, (x0 + x1) / 2, (y0 + y1) / 2, float(box.conf)))
        return dets

    def _observe(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, int]:
        """Returns the vector and the number of coins directly ahead in the player's lane."""
        dets = self.detect(frame)
        # the station platform planks read as blocked lanes to occupancy; suppress blockers
        # for the first second (the real station-exit train arrives ~1.4 s in, seen by then)
        if t - self.run_start_t < 1.0:
            dets = [d for d in dets if d.cls not in ("train", "wall", "boxtrain")]
        self.last_dets = dets
        h, w = frame.shape[:2]
        self.coins_near = sum(
            1
            for d in dets
            if d.cls == "coin"
            and self.geo.lane_of(d.x / w, d.y / h) == self.lane
            and self.geo.depth_of(d.y / h) <= 1
        )
        # stall guard: an unseen dialog or tutorial prompt freezes the scene while the HUD stays
        if (
            self.prev_frame is not None
            and self.prev_frame.shape == frame.shape
            and float(np.mean(cv2.absdiff(frame, self.prev_frame))) < 1.0
        ):
            self.still_since = self.still_since or t
            if t - self.still_since > 2.5:
                self.s.press("jump")
                self.still_since = None
        else:
            self.still_since = None
        self.speed = min(
            (t - self.run_start_t) / 60.0, 1.0
        )  # game speed grows with run time
        v, self.lane = vectorize(
            dets,
            (frame.shape[1], frame.shape[0]),
            self.geo,
            self.lane,
            self.speed * 200,
            False,
        )
        if self.collect_dir is not None and self.n_frames % 10 == 0:
            self.collect_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(self.collect_dir / f"frame_{int(time.time() * 1000)}.jpg"), frame
            )
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.debug_dir / f"{self.n_frames:04d}.jpg"), frame)
            cells = [
                (
                    int(i) // 15,
                    int(i) % 15 // 3,
                    int(i) % 3,
                    float(round(float(v[i]), 2)),
                )
                for i in np.flatnonzero(v[:45])
            ]
            with open(self.debug_dir / "steps.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "n": self.n_frames,
                            "t": round(t, 3),
                            "lane": self.lane,
                            "score": self.score,
                            "dets": [
                                (
                                    d.cls,
                                    int(d.x),
                                    int(d.y),
                                    float(round(float(d.conf), 2)),
                                )
                                for d in dets
                            ],
                            "cells(lane,depth,kind,val)": cells,
                        }
                    )
                    + "\n"
                )
        self.n_frames += 1
        return v, self.coins_near

    def _stable_alive(self) -> bool:
        """One confirmed frame is enough: the run is already moving, and every 300 ms OCR spent
        here is 300 ms closer to the first barrier."""
        return in_run(self.s.frame())

    def reset(self, timeout: float = 60.0) -> np.ndarray:
        """Walk through whatever end/menu/intro screen is showing until the HUD is back.
        Never presses Space: with no hoverboards it opens a 'Need hoverboards?' dialog that only
        its X closes. Dialogs ('Save me!' countdown, hoverboard shop) are waited out, then X-ed."""
        s = self.s
        t0 = time.monotonic()
        since: dict[str, float] = {}
        debug = DATA / "subway" / "reset_debug"
        debug.mkdir(parents=True, exist_ok=True)
        n = 0
        while not self._stable_alive():
            n += 1
            frame = s.frame()
            if self.on_frame is not None:
                self.on_frame(frame)
            kind = classify(frame)
            cv2.imwrite(str(debug / f"{n:02d}_{kind}.jpg"), frame)
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(
                    f"could not get back into a run (stuck on '{kind}'); see {debug}"
                )
            since.setdefault(kind, time.monotonic())
            waited = time.monotonic() - since[kind]
            if kind == "menu":
                s.click(
                    0.5, 0.5
                )  # PRESS TO PLAY; the click also gives the canvas keyboard focus
                time.sleep(1.4)  # run intro before the HUD appears
            elif kind == "results":
                s.click(0.58, 0.92)  # green PLAY
                time.sleep(1.4)
            elif kind == "space":
                s.press("Space")  # only screen where Space is safe
                time.sleep(1.0)
            elif kind == "dialog":
                # Save me! counts down by itself; other dialogs need their close button.
                # Never click card centres: that is "Sign in" / "buy hoverboard" territory.
                if waited > 6.0:
                    if not s.try_dom_close():
                        box = dialog_bbox(frame)
                        if box is not None:
                            x0, y0, x1, _ = box
                            corner = (
                                (x1 - 0.025, y0 + 0.03)
                                if (n % 2 == 0)
                                else (x0 + 0.025, y0 + 0.03)
                            )
                            s.click(*corner)
                    since.pop("dialog", None)
                time.sleep(1.0)
            else:  # intro, attract replay, transitions: wait, then nudge
                if waited > 6.0:
                    s.click(0.5, 0.5)
                    since.pop("other", None)
                time.sleep(1.0)
            if kind != "dialog" and kind != "other":
                since.pop(kind, None)
        self._reset_state()
        v, _ = self._observe(s.frame(), time.monotonic())
        return v

    def park(self, timeout: float = 30.0) -> None:
        """Let a leftover run end on its own (no keys pressed). Unattended, the runner
        crashes within a few seconds; afterwards the game sits on an end screen, so the
        next reset() starts a fresh run: zero speed, score 0, clear track. Without this,
        the next session would join the old run mid-flight at high speed."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            frame = self.s.frame()
            if self.on_frame is not None:
                self.on_frame(frame)
            if not alive(frame):
                return
            time.sleep(0.4)

    def step(self, action: str) -> tuple[np.ndarray, float, bool, dict]:
        if action == "left":
            self.lane = max(self.lane - 1, 0)
        elif action == "right":
            self.lane = min(self.lane + 1, 2)
        self.s.press(action)
        time.sleep(self.step_period)
        frame = self.s.frame()
        t = time.monotonic()
        self.dead_frames = 0 if alive(frame) else self.dead_frames + 1
        dead = self.dead_frames >= 2  # two throttled frames = ~0.2 s without the HUD
        if dead and self.prev_frame is not None:
            self.score = read_score(self.prev_frame) or int(
                (t - self.run_start_t) * 8
            )  # one OCR per episode, off the hot path
        v, coins_ahead = self._observe(frame, t)
        self.prev_frame = frame
        reward = 0.01 + 0.05 * min(coins_ahead, 2) + (-1.0 if dead else 0.0)
        return (
            v,
            reward,
            dead,
            {"dead": dead, "score": self.score, "coins": coins_ahead},
        )
