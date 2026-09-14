# Fly brain plays Clash of Clans: attack loop

## Context

Pivot from cancer sniffing. Same asset (FlyWire olfactory + mushroom body
circuit, already extracted and verified in data/raw/), new task: the fly
brain decides whether to attack a base and where to drop the army the user
trained. Demo look stays as in demo_videos/ (3D fly left, live MaleCNS soma
cloud right). Nothing exists in the repo yet except src/cancer_sniffing_fly/__init__.py.

Expected behaviour (user): tap the attack button, find a match, look at the
base and decide attack or skip within 30 s, if attack then analyse and deploy,
else press Next and repeat.

## Game facts verified (Sept 2026)

| Fact | Value | Source |
|---|---|---|
| Entry | Home village "Attack!" button (bottom left) opens attack window; Multiplayer tab; "Find a Match" | MyBot PrepareSearch.au3: findButton("AttackButton"), MultiplayerTab, "FindMatchNormal" |
| Matchmaking | "clouds" screen of variable length; bots wait for clouds to clear | MyBot WaitForClouds.au3 |
| Scouting | 30 s once the village appears. Battle starts on first deployment. At 0 s Next disappears and the 3:00 battle timer starts automatically | Fandom Multiplayer Battles; MyBot AttackRemainingTime comment |
| Battle | 3 minutes | same |
| Next | Costs gold by Town Hall (TH9 750, TH11+ 1,000; lower THs less). No elixir cost | Fandom, forum thread |
| Scout screen | Top-left: available Gold, Elixir, Dark Elixir (if storage present), Trophies +/-; MyBot OCRs them at fixed window coords (860x732 window: gold 48,69; elixir 48,98; DE 48,126; trophies 45,168) | MyBot GetResources.au3 |
| Deploy zone | Troops cannot spawn within 1 tile of any visible building; the red line marks it; bots detect the red line by colour and drop on 4 edges (TL, BL, BR, TR) | MyBot _GetRedArea.au3; wiki no-spawn rule |
| Troop bar | Bottom bar; slots found by template matching, counts by OCR under each icon; more than 11 slots need a drag | MyBot GetAttackBar.au3 |
| End | "End Battle" button while fighting, then "Okay" confirm; after result screen "Return Home" button | MyBot ReturnHome.au3 |
| Legend League | No Next, 8 attacks per day, base pre-picked. Stay below 5,000 trophies | Supercell FAQ |
| Single Player campaign | Same attack UI, no trophies, no time limit, does not break shield, 50 levels, replayable | Fandom |
| Automation policy | Supercell ToS bans bots, emulators, automation; permanent ban waves. Official API is read-only | supercell.com ToS, fair play posts |
| Mac platform | Google Play Games on PC is Windows only. BlueStacks Air runs CoC on Apple Silicon; ADB on Air unconfirmed. An Android phone over USB ADB is the deterministic path | BlueStacks, Supercell support |
| Base detection data | Roboflow "find-this-base" CoC dataset, 16 classes (Canon, WizzTower, Xbow, AD, Mortar, Inferno, Scattershot, AirSweeper, BombTower, ClanCastle, Eagle, KingPad, QueenPad, RcPad, TH13, WardenPad), CC-BY 4.0, YOLO exports | universe.roboflow.com |

## Architecture

```
Android (phone or BlueStacks Air)
   │ adb exec-out screencap -p        │ adb shell input tap/swipe
   ▼                                  ▲
game/screen.py  ──►  game/ui.py (button finder, OCR)  ──►  game/actions.py
   │
   ▼
game/base_reader.py  (YOLO defenses + red line + loot OCR) ──► 53-dim base vector
   │
   ▼
brain/circuit.py  (ORN→PN→KC→APL→MBON from FlyWire, rate model)
brain/learn.py    (three-factor KC→MBON rule, DAN = reward)
   │
   ▼
agent/loop.py  state machine with hard deadlines
   │
   ▼
demo/  (Three.js: fly + MaleCNS soma cloud, counters)
```

One Python package `src/flyclash/`. Delete or rename `cancer_sniffing_fly`.

## State machine (agent/loop.py)

| State | Action | Deadline | Exit |
|---|---|---|---|
| HOME | verify home screen (Attack! button template), read army bar is non-empty on the attack window later | 10 s | tap Attack! |
| ATTACK_WINDOW | verify Multiplayer tab, tap Find a Match | 10 s | clouds |
| CLOUDS | poll screenshot every 0.5 s until loot numbers OCR non-empty | 120 s then abort to HOME | scouting |
| SCOUT_DECIDE | t0 = first frame with loot. Capture 1 frame, run base_reader (target under 3 s), run brain forward (ms), read MBON valence | decide by t0 + 20 s, hard stop 25 s | ATTACK or tap Next → CLOUDS |
| ANALYSE | already inside the 30 s. Pick side and drop order from MBON compartment winners. Optional zoom-out first (MyBot CheckZoomOut) | finish by t0 + 28 s | DEPLOY |
| DEPLOY | first tap starts the 3:00 timer. Drop troops along chosen red-line edge, spacing per MyBot MakeDropLine; heroes then abilities; clan castle; spells | 150 s | END |
| END | when troop bar empty or 150 s: tap End Battle, Okay, wait result screen, read stars/percent/loot, tap Return Home | 30 s | LEARN |
| LEARN | reward = stars (0-3) plus loot fraction; Next presses cost -0.1; update KC→MBON | ms | HOME |

Every state has a screenshot-based precondition check and a timeout that
returns to HOME via ReturnHome logic. Any unknown screen for 60 s: stop and
alert, do not tap blindly.

## Base vector (53 glomeruli, matches the 53 ORN types in FlyWire)

- YOLO on the scout frame (start from Roboflow dataset, fine-tune with ~200
  screenshots from the user's Town Hall range; TH class must be extended).
- Red line polygon from colour segmentation (MyBot colour variation 40).
- Isometric grid: map detections to tile coords using the red-line diamond
  as the frame.
- 8 sectors around the Town Hall × 6 defense groups (point, splash, air,
  inferno/scatter/eagle, heroes, clan castle) = 48 features, each a
  distance-weighted count normalised to [0,1].
- Plus 5 scalars: gold, elixir, dark elixir (OCR, log-scaled), trophies
  offered, opponent TH level (from detection or the TH sprite). Total 53.

## Fly brain mapping (brain/)

- Input: 53-dim vector → ORN firing rates → real ORN→PN weights → divisive
  normalisation → real PN→KC sparse projection (median 5 PNs per KC,
  verified) → APL winner-take-all to 5-10% active KCs.
- Decision: KC→MBON weights, initialised uniform. MBONs grouped by Aso 2014
  valence: approach group = ATTACK, avoid group = NEXT. Sign of the
  difference decides. This is the same "attack or not" readout the fly uses
  for odours.
- Attack plan: the 15 MB compartments are 15 actions: 4 edges × 3 drop
  patterns (spread line, two-point pincer, single point) plus 3 troop orders.
  Winner compartment per slot. Ties broken by innate prior (drop on the edge
  with the fewest defenses).
- Learning: three-factor rule. After the battle, DAN activity = reward
  prediction error (stars minus running mean). Depress KC→MBON synapses in
  the winning compartment for negative RPE, potentiate for positive, only on
  KCs that were active for that base. Reference: Springer & Nawrot 2021
  (github nawrotlab). Next = avoid action rewarded with -0.1 (gold cost).
- Speed: rate model, under 10 ms. Optional Brian2 LIF on the 8.6k-neuron
  olfactory+MB subgraph for the demo visuals only, run after the decision.
- Ablation that makes it a result: real PN→KC vs degree-matched random,
  learning curve (stars per attack) over 200 attacks.

## Files

```
src/flyclash/
  game/adb.py          screencap, tap, swipe, device discovery
  game/ui.py           template matching (OpenCV) for buttons, OCR (tesseract or paddleocr) for numbers
  game/base_reader.py  YOLO (ultralytics) + red line + grid mapping → 53-vector
  game/actions.py      attack, find_match, next, drop_line, end_battle, return_home
  brain/circuit.py     load FlyWire matrices (reuse extraction from docs/plan.md, Shiu parquet)
  brain/learn.py       three-factor rule + replay buffer of (kc_code, compartment, reward)
  agent/loop.py        state machine with deadlines and logging
  agent/reward.py      parse result screen: stars, percent, loot
  demo/server.py       websocket frames: state, KC actives, MBON values, DN "taps"
  demo/web/            Three.js fly + soma cloud (MaleCNS body-annotations feather)
assets/templates/      button crops at the chosen resolution
assets/yolo/           fine-tuned weights
data/raw/              existing downloads (FlyWire, MaleCNS, Shiu)
tests/test_circuit.py  KC sparseness, decorrelation, conditioning asserts
tests/test_ui.py       template matches on saved screenshots (offline)
```

Reuse: circuit extraction code already run in this session (docs/plan.md
"Circuit extraction"), Shiu Connectivity_783.parquet at
/tmp/shiu_check (copy into data/raw/), MaleCNS annotations in data/raw/malecns.

## Fixed resolution

Lock the device to one resolution (phone native or BlueStacks Air window),
capture all templates at that size, never scale. MyBot's coordinates above
are for its 860x732 window and are reference only.

## Risk and account policy

- Automation violates Supercell ToS. Use a dedicated throwaway account and
  device. Rate-limit to human pace: at most one attack per 4 minutes, random
  delays 0.3-1.2 s between taps, sessions under 40 minutes.
- Develop and test the whole loop on the Single Player campaign first (same
  UI, no trophies, no 30 s limit, replayable), then Goblin maps for the
  30 s timing with a stopwatch, then multiplayer.
- Stay under 5,000 trophies (Legend removes Next and the flow changes).
- Do not touch Clan War, Raid Weekend or Builder Base screens; abort on
  unknown screen.

## Verification

1. Offline: 50 saved screenshots across states; ui.py finds every button
   with 0 false positives; OCR loot matches hand-labelled values.
2. Offline: base_reader on 30 scout screenshots; sector counts match manual
   labels within ±1 defense.
3. Circuit asserts (tests/test_circuit.py): 5-10% KC active for random
   base vectors; conditioning demo learns attack vs skip on synthetic
   "easy" vs "hard" bases in under 30 trials.
4. Timing: SCOUT_DECIDE end-to-end under 20 s on the Mac, measured 20 runs.
5. Single Player: 10 consecutive levels completed autonomously, no manual
   intervention, correct Return Home each time.
6. Multiplayer, throwaway account: 20 attacks, log per attack
   (decision, sectors, stars, loot, elapsed), zero stuck states.
7. Learning: stars per attack over 200 attacks, real wiring vs random.

## Order of work

1. adb.py + ui.py + templates, Single Player loop without brain (scripted
   drop on weakest edge). Proves control.
2. base_reader.py with fine-tuned YOLO.
3. brain/ from existing matrices, asserts.
4. agent/loop.py with deadlines; multiplayer on throwaway.
5. demo/.

## Decisions (confirmed by user)

- Platform: Android phone over USB ADB. `adb devices`, `adb exec-out
  screencap -p`, `adb shell input tap X Y`, `adb shell input swipe`.
  Enable USB debugging, keep screen on (`adb shell svc power stayon usb`),
  lock rotation, set the phone's resolution once and capture all templates
  at that size.
- Account: brand-new throwaway. User plays the base-building side and trains
  the army by hand; the bot only runs the attack loop.
- Town Hall: starts at TH1 and grows. Consequences:
  - The Roboflow dataset (TH13 defenses) does not fit. Collect our own
    scout screenshots from Single Player levels and early multiplayer, label
    with the low-TH classes: Cannon, Archer Tower, Mortar, Wizard Tower,
    Air Defense, Hidden Tesla (unseen, skip), Walls, Town Hall, Clan Castle,
    Storages, Collectors. Label 150-300 frames in Roboflow or Label Studio,
    fine-tune YOLOv8n. Retrain as new defenses unlock.
  - Base vector shrinks: 8 sectors × 4 defense groups (point, splash, air,
    other) = 32, plus loot and TH scalars. Pad to 53 glomeruli with zeros or
    map several sectors per glomerulus; keep the 53-dim interface fixed so
    the brain never changes shape.
  - Army: Barbarians and Archers only at first, later Giants, Wall Breakers,
    Goblins. Actions collapse to 4 edges × drop pattern; Barch (barbarians
    front, archers behind) is the default drop pattern.
  - Next cost is tiny at low TH (tens of gold), so skipping is cheap and
    the -0.1 penalty is right.
  - The tutorial must be completed by hand; the bot starts after the first
    Single Player levels are unlocked.
  - New-account shield and league: attacking breaks the shield; at low
    trophies matchmaking is fast and bases are weak, good for learning.
- Sandbox: Single Player campaign first, then Goblin-map levels for the
  30 s timing, then multiplayer.
