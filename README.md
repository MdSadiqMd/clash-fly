# flyclash

A fruit fly's olfactory circuit, wired straight from the FlyWire connectome, plays games with the fly's own learning rule. Two games:

- **Subway Surfers** (current, no phone needed): the official HTML5 build on
  poki.com driven by Playwright. See docs/subway-plan.md.
- **Clash of Clans** (kept, needs an Android phone): decides whether to attack
  a base and where to drop the army. See docs/clash-plan.md.

## Subway Surfers quick start

```bash
uv sync --extra ocr --extra yolo && uv run playwright install chromium
uv run pytest                                  # 19 tests: circuit, UI, loop, sim, trace
uv run flyclash surf-sim --episodes 150 --seeds 5   # offline RL curves + wiring ablation -> data/subway/*.png
uv run flyclash surf-run --headless --episodes 3 --no-learning   # live loop on poki.com, logs data/subway/live.csv
uv run flyclash surf-curves --tag live         # plot survival / reward / loss for the live run
```

Detector weights (`assets/yolo/subway.pt`) need the Roboflow set, see docs/subway-plan.md. Without them the fly is blind to trains and dies at the first one; the loop, death detection, restart and learning still run.

### Dashboard (TanStack Start, `client/`)

```bash
uv run flyclash surf-serve          # ws://localhost:8765, owns the headless Poki browser
cd client && pnpm install && pnpm dev   # http://localhost:3000
```

Press RUN 30s: the fly plays one 30-second run (or until it crashes). Left pane streams the game, top-right is the FlyWire soma cloud with the active Kenyon cells and MBON/DAN pulses, bottom-right shows the fly at the keyboard pressing the chosen arrow key. Stats and the episode list update live; `data/subway/live.csv` keeps the learning curve.

## Clash of Clans (original 
The fly decides whether to attack a Clash of Clans base and where to drop the army. Only the Kenyon-cell to MBON synapses learn, with the fly's own dopamine three-factor rule. No backprop.

The base is encoded as a 53-dim "odour" (8 sectors × 4 defense groups + loot, trophies, town hall), pushed through ORN → PN → KC → APL → MBON, and the approach-minus-avoid MBON balance is the attack-or-Next decision.

## Warning

Automating Clash of Clans violates Supercell's terms of service and gets accounts permanently banned. Use a throwaway account on a spare Android phone. The agent is rate-limited to one attack per 4 minutes and stops on unknown screens.

## Setup

```bash
uv sync --group dev            # core
uv sync --extra ocr --extra yolo   # add OCR (rapidocr) and YOLO (ultralytics)
```

Data (already in data/raw if you ran the research session):
`data/raw/flywire/neuron_annotations_783.tsv`, `data/raw/shiu/Connectivity_783.parquet`.

```bash
uv run flyclash build-circuit   # extracts ORN/PN/KC/MBON matrices into data/circuit.npz
uv run pytest                   # circuit asserts, UI matching, full loop on a fake game
uv run flyclash sim --ablation  # offline learning curve, real PN→KC wiring vs random
```

## Phone

The `flyclash` command lives in the project venv: run it as `uv run flyclash ...`
or `source .venv/bin/activate` first.

1. Mac: `brew install --cask android-platform-tools` (gives `adb`).
   Phone: Settings → About phone → tap "Build number" 7 times → Settings → System → Developer options → USB debugging ON. Plug in with a USB-C data cable (Apple's USB-C charge cable carries USB 2.0 data; a USB-C-to-Lightning cable does not fit an Android phone). Choose "File transfer" if the phone asks, accept the "Allow USB debugging?" prompt, then `adb devices` must show the serial with the word `device`. `adb shell wm size` prints the resolution
2. `uv run flyclash capture` a few times on each screen: home, attack window, scouting, battle, result. Crop these templates from the screenshots at native resolution into `assets/templates/`:    `attack_button, multiplayer_tab, find_match, next_button, end_battle, okay, return_home, star`
3. `uv run flyclash calibrate` prints the layout regions in pixels; adjust `assets/layout.json` so the loot numbers, result numbers and troop bar slots line up
4. Detector. `uv run flyclash fetch-dataset` downloads the only public labelled set (find-this-base, 125 high-TH images, CC-BY 4.0, mirrored on Hugging Face as keremberke/clash-of-clans-object-detection) and converts it to YOLO under `data/yolo/coc` with our class names. `uv run flyclash train-yolo --epochs 80` fine-tunes YOLOv8n on the Apple GPU and writes `assets/yolo/best.pt`.
That set has Cannon, Mortar, WizardTower, AirDefense, ClanCastle, TownHall but no ArcherTower, so for a TH2 account label ~100 of your own scout screenshots (Roboflow or Label Studio, classes as in `CANONICAL_CLASSES`), export as "YOLOv8", then `uv run flyclash fetch-dataset --roboflow-export <folder>` merges them and retrain. Without weights the fly only smells loot and trophies and drops on a default edge. 5. Train the army by hand. Start on the Single Player campaign.

```bash
uv run flyclash run --attacks 3 --th 3           # attack loop
uv run flyclash export-somas && uv run flyclash demo   # then run with --demo
```

## Offline result so far (synthetic bases, `flyclash sim --attacks 400 --ablation --seeds 5`)

Real FlyWire wiring (53 glomeruli, 277 uniglomerular PNs, 5,177 KCs, 35 MBON types; 28 approach / 7 avoid by neurotransmitter) versus degree-matched random PN→KC wiring, same learning rule, reward in the last 100 of 400 attacks:

| Wiring | mean reward | sd over 5 seeds |
|---|---|---|
| real PN→KC | 0.29 | 0.10 |
| random PN→KC | 0.15 | 0.03 |

Hard-base skip rate rises from ~0.25 to ~0.46 over 400 attacks while easy bases are still attacked 100% of the time. Weakest-edge choice is worse with real wiring (0.40 vs 0.66 hit rate), so the reward gain comes from the attack/skip decision, not from edge selection. This is on synthetic bases; the phone-side numbers are still to be collected.
