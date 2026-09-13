"""flyclash command line.

flyclash capture [--out shots/]      save a screenshot from the phone (for templates and YOLO labels)
flyclash calibrate                   print screen size and the layout regions in pixels
flyclash build-circuit               extract the olfactory circuit from FlyWire into data/circuit.npz
flyclash sim [--attacks 200]         offline learning curve, real wiring vs random wiring
flyclash run [--attacks 5]           the real attack loop on the connected phone
flyclash export-somas                write demo/web/somas.json from the FlyWire annotations
flyclash fetch-dataset               download the public CoC detection set and convert to YOLO (data/yolo/coc)
flyclash train-yolo [--epochs 50]    fine-tune yolov8n on data/yolo/coc, weights -> assets/yolo/best.pt
flyclash demo                        serve the demo page and websocket
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from flyclash.config import ASSETS, DATA, ROOT, load_config


def cmd_capture(a):
    from flyclash.game.adb import Adb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = Adb(a.serial).screencap()
    path = out / f"shot_{int(time.time())}.png"
    cv2.imwrite(str(path), frame)
    print(path, frame.shape[1], "x", frame.shape[0])


def cmd_calibrate(a):
    from flyclash.game.adb import Adb

    cfg = load_config()
    w, h = Adb(a.serial).size()
    print(f"screen {w}x{h}")
    for name, reg in vars(cfg.layout).items():
        if isinstance(reg, tuple) and len(reg) == 4:
            print(
                f"{name:16s} px=({int(reg[0] * w)},{int(reg[1] * h)})-({int(reg[2] * w)},{int(reg[3] * h)})"
            )


def cmd_build_circuit(a):
    from flyclash.brain.circuit import Circuit

    c = Circuit.from_flywire(cache=DATA / "circuit.npz")
    print(
        f"glomeruli {len(c.glomeruli)}  PNs {c.W_glom_pn.shape[1]}  KCs {c.n_kc}  MBON types {c.n_mbon}"
    )
    print("approach MBONs:", [t for t, v in zip(c.mbon_types, c.mbon_valence) if v > 0])
    print("avoid MBONs:", [t for t, v in zip(c.mbon_types, c.mbon_valence) if v < 0])
    print("action MBONs:", [c.mbon_types[i] for i in c.action_mbons])


def _circuit(random_ok: bool):
    from flyclash.brain.circuit import Circuit

    cache = DATA / "circuit.npz"
    if cache.exists():
        return Circuit.load(cache)
    try:
        return Circuit.from_flywire(cache=cache)
    except FileNotFoundError:
        if not random_ok:
            raise
        print("FlyWire data missing, using a synthetic circuit")
        return Circuit.random()


def cmd_sim(a):
    from flyclash.agent import sim

    c = _circuit(random_ok=True)
    r = sim.run(c, a.attacks, seed=0)
    print(json.dumps({k: v for k, v in r.items() if k != "rewards"}, indent=1))
    if a.ablation:
        print(json.dumps(sim.ablation(c, a.attacks, a.seeds), indent=1))


def cmd_run(a):
    from flyclash.agent.loop import Agent
    from flyclash.brain.learn import Plastic, Policy
    from flyclash.game.actions import RealGame
    from flyclash.game.adb import Adb
    from flyclash.game.base_reader import read_base

    cfg = load_config()
    circuit = _circuit(random_ok=False)
    plastic = Plastic.load(cfg.plastic_path, circuit)
    policy = Policy(circuit, plastic, attack_bias=a.attack_bias)
    adb = Adb(a.serial)
    adb.keep_awake()
    publish = None
    if a.demo:
        from flyclash.demo.server import Broadcaster

        publish = Broadcaster.start().publish
    agent = Agent(
        RealGame(adb, cfg),
        policy,
        cfg,
        lambda f, loot: read_base(f, loot, a.th, cfg.yolo_weights),
        publish=publish,
    )
    for rec in agent.run(a.attacks, plastic_path=cfg.plastic_path):
        print(json.dumps(rec.__dict__))


def cmd_export_somas(a):
    import pandas as pd

    ann = pd.read_csv(
        DATA / "raw" / "flywire" / "neuron_annotations_783.tsv",
        sep="\t",
        low_memory=False,
    )
    ann = ann.dropna(subset=["soma_x"])
    cls = ann.cell_class.fillna(ann.super_class).astype(str)
    keep = cls.isin(["olfactory", "ALPN", "Kenyon_Cell", "MBON", "DAN", "APL"]) | (
        np.random.default_rng(0).random(len(ann)) < a.frac
    )
    ann, cls = ann[keep], cls[keep]
    classes = sorted(cls.unique())
    out = {
        "classes": classes,
        "xyz": np.stack([ann.soma_x, ann.soma_y, ann.soma_z], 1).astype(int).tolist(),
        "cls": [classes.index(c) for c in cls],
    }
    p = ROOT / "src" / "flyclash" / "demo" / "web" / "somas.json"
    p.write_text(json.dumps(out))
    print(p, len(out["xyz"]), "somas")


def cmd_fetch_dataset(a):
    from flyclash.game import dataset

    if a.roboflow_export:
        print("merged", dataset.add_yolo_export(Path(a.roboflow_export)))
    else:
        print("hugging face", dataset.fetch_huggingface())
    print("data.yaml:", dataset.YOLO_ROOT / "data.yaml", "classes:", dataset.CLASSES)


def cmd_train_yolo(a):
    from flyclash.game import dataset

    cfg = load_config()
    print(
        "weights:",
        dataset.train(
            dataset.YOLO_ROOT / "data.yaml",
            cfg.yolo_weights,
            epochs=a.epochs,
            imgsz=a.imgsz,
        ),
    )


def _surf_policy(
    learning: bool = True,
    epsilon: float = 0.02,
    seed: int = 0,
    random_wiring: bool = False,
):
    from flyclash.brain.learn import Plastic
    from flyclash.subway.agent import SUBWAY_DIR, SurfPolicy

    c = _circuit(random_ok=True)
    if random_wiring:
        c = c.with_random_pn_kc(seed)
    plastic_path = SUBWAY_DIR / (
        "plastic_random.npz" if random_wiring else "plastic.npz"
    )
    p = Plastic.load(plastic_path, c) if learning else Plastic.fresh(c)
    p.epsilon = epsilon
    return SurfPolicy(
        c, p, rng=np.random.default_rng(seed), learning=learning
    ), plastic_path


def cmd_surf_sim(a):
    from flyclash.subway import sim
    from flyclash.subway.agent import SUBWAY_DIR, plot_curves, run_episodes

    class FakeClock:  # sim steps are 0.1 s apart so the eligibility window means the same thing as live
        t = 0.0

        def __call__(self):
            self.t += 0.1
            return self.t

    q = max(a.episodes // 4, 1)
    summary = {}
    for tag, learning, rw in (
        ("sim_real", True, False),
        ("sim_random_wiring", True, True),
        ("sim_no_learning", False, False),
    ):
        if a.only and tag != a.only:
            continue
        firsts, lasts = [], []
        for seed in range(a.seed, a.seed + a.seeds):
            policy, _ = _surf_policy(learning=learning, seed=seed, random_wiring=rw)
            policy.plastic = policy.plastic.fresh(policy.circuit, epsilon=a.epsilon)
            (SUBWAY_DIR / f"{tag}_s{seed}.csv").unlink(missing_ok=True)
            logs = run_episodes(
                sim.SimEnv(seed=seed),
                policy,
                a.episodes,
                tag=f"{tag}_s{seed}",
                clock=FakeClock(),
            )
            firsts.append(np.mean([l.steps for l in logs[:q]]))
            lasts.append(np.mean([l.steps for l in logs[-q:]]))
            plot_curves(
                SUBWAY_DIR / f"{tag}_s{seed}.csv", SUBWAY_DIR / f"{tag}_s{seed}.png"
            )
        summary[tag] = {
            "first_quarter_steps": float(np.mean(firsts)),
            "last_quarter_steps": float(np.mean(lasts)),
            "last_quarter_sd": float(np.std(lasts)),
            "seeds": a.seeds,
        }
        print(
            f"{tag:18s} survival steps: first quarter {np.mean(firsts):6.1f} -> last quarter {np.mean(lasts):6.1f} ± {np.std(lasts):5.1f} over {a.seeds} seeds"
        )
    (SUBWAY_DIR / "sim_summary.json").write_text(json.dumps(summary, indent=1))
    print(
        "curves: data/subway/<tag>_s<seed>.png, summary: data/subway/sim_summary.json"
    )


def cmd_surf_run(a):
    from flyclash.subway.agent import SUBWAY_DIR, plot_curves, run_episodes
    from flyclash.subway.browser import LiveEnv, PokiSession

    cfg = load_config()
    policy, plastic_path = _surf_policy(learning=not a.no_learning, epsilon=a.epsilon)
    publish = None
    if a.demo:
        from flyclash.demo.server import Broadcaster

        publish = Broadcaster.start().publish
    session = PokiSession(headless=a.headless).start()
    try:
        env = LiveEnv(
            session,
            ASSETS / "yolo" / "subway.pt",
            collect_dir=SUBWAY_DIR / "frames" if a.collect else None,
        )
        for log in run_episodes(env, policy, a.episodes, tag="live", on_step=publish):
            print(json.dumps(log.__dict__))
            policy.plastic.save(plastic_path)
    finally:
        session.close()
    print("curves:", plot_curves(SUBWAY_DIR / "live.csv", SUBWAY_DIR / "live.png"))


def cmd_surf_collect(a):
    """Save canvas frames while pressing random keys, for labelling."""
    import random

    from flyclash.subway.agent import SUBWAY_DIR
    from flyclash.subway.browser import PokiSession
    from flyclash.subway.reader import ACTIONS

    out = SUBWAY_DIR / "frames"
    out.mkdir(parents=True, exist_ok=True)
    s = PokiSession(headless=a.headless).start()
    try:
        s.focus()
        for _ in range(3):
            s.page.keyboard.press("Space")
            time.sleep(0.5)
        for i in range(a.frames):
            s.press(random.choice(ACTIONS))
            cv2.imwrite(str(out / f"frame_{int(time.time() * 1000)}.jpg"), s.frame())
            time.sleep(a.interval)
            if i % 40 == 39:
                s.page.keyboard.press("Space")
    finally:
        s.close()
    print(out, len(list(out.glob("*.jpg"))), "frames")


def cmd_surf_dataset(a):
    from flyclash.game import dataset
    from flyclash.subway.classes import SUBWAY_CLASSES, canonical_class

    root = DATA / "yolo" / "subway"
    print(
        "merged",
        dataset.add_yolo_export(
            Path(a.roboflow_export), root, SUBWAY_CLASSES, canonical_class
        ),
    )
    print("data.yaml:", root / "data.yaml", "classes:", SUBWAY_CLASSES)


def cmd_surf_train(a):
    from flyclash.game import dataset

    print(
        "weights:",
        dataset.train(
            DATA / "yolo" / "subway" / "data.yaml",
            ASSETS / "yolo" / "subway.pt",
            epochs=a.epochs,
            imgsz=a.imgsz,
            run_name="subway",
        ),
    )


def cmd_surf_serve(a):
    from flyclash.subway.server import serve

    serve(port=a.port, headless=not a.headed)


def cmd_surf_curves(a):
    from flyclash.subway.agent import SUBWAY_DIR, plot_curves

    print(plot_curves(SUBWAY_DIR / f"{a.tag}.csv", SUBWAY_DIR / f"{a.tag}.png"))


def cmd_demo(a):
    from flyclash.demo.server import Broadcaster, serve_static

    b = Broadcaster.start()
    print("websocket on ws://localhost:8765, page on http://localhost:8000")
    serve_static()


def main():
    p = argparse.ArgumentParser(prog="flyclash")
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--out", default="shots")
    c.add_argument("--serial")
    c.set_defaults(f=cmd_capture)
    c = sp.add_parser("calibrate")
    c.add_argument("--serial")
    c.set_defaults(f=cmd_calibrate)
    c = sp.add_parser("build-circuit")
    c.set_defaults(f=cmd_build_circuit)
    c = sp.add_parser("sim")
    c.add_argument("--attacks", type=int, default=200)
    c.add_argument("--ablation", action="store_true")
    c.add_argument("--seeds", type=int, default=5)
    c.set_defaults(f=cmd_sim)
    c = sp.add_parser("run")
    c.add_argument("--attacks", type=int, default=5)
    c.add_argument("--serial")
    c.add_argument("--th", type=int, default=3)
    c.add_argument("--attack-bias", type=float, default=0.0)
    c.add_argument("--demo", action="store_true")
    c.set_defaults(f=cmd_run)
    c = sp.add_parser("export-somas")
    c.add_argument("--frac", type=float, default=0.15)
    c.set_defaults(f=cmd_export_somas)
    c = sp.add_parser("fetch-dataset")
    c.add_argument(
        "--roboflow-export", help="folder of a Roboflow YOLOv8 export to merge"
    )
    c.set_defaults(f=cmd_fetch_dataset)
    c = sp.add_parser("train-yolo")
    c.add_argument("--epochs", type=int, default=50)
    c.add_argument("--imgsz", type=int, default=960)
    c.set_defaults(f=cmd_train_yolo)
    c = sp.add_parser("surf-sim")
    c.add_argument("--episodes", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--epsilon", type=float, default=0.02)
    c.add_argument("--only")
    c.set_defaults(f=cmd_surf_sim)
    c = sp.add_parser("surf-run")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--headless", action="store_true")
    c.add_argument("--no-learning", action="store_true")
    c.add_argument("--epsilon", type=float, default=0.02)
    c.add_argument("--collect", action="store_true")
    c.add_argument("--demo", action="store_true")
    c.set_defaults(f=cmd_surf_run)
    c = sp.add_parser("surf-collect")
    c.add_argument("--frames", type=int, default=300)
    c.add_argument("--interval", type=float, default=0.3)
    c.add_argument("--headless", action="store_true")
    c.set_defaults(f=cmd_surf_collect)
    c = sp.add_parser("surf-dataset")
    c.add_argument("--roboflow-export", required=True)
    c.set_defaults(f=cmd_surf_dataset)
    c = sp.add_parser("surf-train")
    c.add_argument("--epochs", type=int, default=60)
    c.add_argument("--imgsz", type=int, default=640)
    c.set_defaults(f=cmd_surf_train)
    c = sp.add_parser("surf-curves")
    c.add_argument("--tag", default="live")
    c.set_defaults(f=cmd_surf_curves)
    c = sp.add_parser("surf-serve")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--headed", action="store_true")
    c.set_defaults(f=cmd_surf_serve)
    c = sp.add_parser("demo")
    c.set_defaults(f=cmd_demo)
    a = p.parse_args()
    a.f(a)
