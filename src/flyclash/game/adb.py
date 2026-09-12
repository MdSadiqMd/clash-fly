"""Thin ADB wrapper: screenshots in, taps out. One phone over USB."""

from __future__ import annotations

import subprocess

import cv2
import numpy as np


class AdbError(RuntimeError):
    pass


class Adb:
    def __init__(self, serial: str | None = None, timeout: float = 10.0):
        self.serial = serial
        self.timeout = timeout

    def _cmd(self, *args: str) -> list[str]:
        base = ["adb"] + (["-s", self.serial] if self.serial else [])
        return base + list(args)

    def _run(self, *args: str, binary: bool = False) -> bytes | str:
        try:
            out = subprocess.run(
                self._cmd(*args), capture_output=True, timeout=self.timeout, check=True
            )
        except FileNotFoundError as e:
            raise AdbError(
                "adb not found on PATH; install Android platform-tools"
            ) from e
        except subprocess.CalledProcessError as e:
            raise AdbError(
                f"adb {' '.join(args)} failed: {e.stderr.decode(errors='ignore')}"
            ) from e
        return out.stdout if binary else out.stdout.decode(errors="ignore")

    @staticmethod
    def devices() -> list[str]:
        out = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=10
        ).stdout
        return [
            line.split()[0]
            for line in out.splitlines()[1:]
            if line.strip().endswith("device")
        ]

    def screencap(self) -> np.ndarray:
        """Returns a BGR uint8 array of the current screen."""
        png = self._run("exec-out", "screencap", "-p", binary=True)
        frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise AdbError("screencap returned no image")
        return frame

    def size(self) -> tuple[int, int]:
        out = self._run("shell", "wm", "size")
        w, h = out.strip().split()[-1].split("x")
        return int(w), int(h)

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self._run(
            "shell",
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(ms),
        )

    def keep_awake(self) -> None:
        self._run("shell", "svc", "power", "stayon", "usb")
