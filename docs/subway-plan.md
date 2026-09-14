# Fly brain plays Subway Surfers (web, no phone)

Implemented 2026-09-13 in `src/flyclash/subway/`. The Clash of Clans code in
`src/flyclash/game` and `src/flyclash/agent` is untouched (see clash-plan.md).

## Why Subway Surfers

| | Subway Surfers | Beach Buggy Racing 2 |
|---|---|---|
| Runs on the Mac | official SYBO HTML5 build on poki.com, arrow keys | Android/iOS/Steam-Windows only |
| Labelled data | Roboflow "subway surfers object dataset" (horizon), 2,328 images; two smaller sets | none |
| Prior bots | 5+ GitHub projects, screenshot -> CNN -> 5 keys | none |
| Account risk | none, single player, no login | n/a |

## What is built

- `browser.py` Playwright session on poki.com with a persistent profile
  (`data/subway/profile`), canvas capture via CDP at ~50 fps, `alive()` from
  the blue pause button (0.42 blue fraction in play, 0.00 on end screens),
  score OCR once a second, tutorial prompt following ("Press Arrow Key Up"
  pauses the run until pressed), reset that walks Space/ArrowUp until the HUD
  is back. `LiveEnv` is gym-like: `reset()`, `step(action)`.
- `reader.py` detections -> 53-dim vector: 3 lanes x 5 depth bins x 3 obstacle
  kinds + coins per lane + player lane + speed + powerup. Track geometry
  measured on the real canvas (rails 55% of width at the player, 12% at the
  horizon, player at y = 0.80). `innate_prior()` is the reflex: low barrier ->
  jump, high -> roll, blocker -> emptier neighbouring lane.
- `sim.py` offline env with the same vector: obstacles scroll toward the
  player, speed rises, coins, death rules per obstacle kind.
- `agent.py` one RL loop for both envs (`run_episodes`), `SurfPolicy`
  (circuit forward -> 5 action MBONs, per-MBON running-mean normalisation,
  reflex prior, epsilon-greedy), learning via `brain/learn.Trace`: death
  punishes the last 1.5 s of (KC, action-MBON) pairs with exp decay (PPL1),
  score gain credits the last 0.5 s (PAM). Logs per episode to CSV (steps,
  reward, coins, score, dead, loss = mean |reward prediction error|,
  decisions/s) and `plot_curves()` draws survival / reward / loss.
- CLI: `surf-sim`, `surf-run`, `surf-collect`, `surf-dataset`, `surf-train`, `surf-curves`.

## Results so far (simulator, FlyWire circuit, 150 episodes, 5 seeds)

| arm | survival first quarter | last quarter |
|---|---|---|
| real PN->KC wiring, learning | 137 | 163 ± 17 |
| degree-matched random wiring, learning | 118 | 157 ± 35 |
| real wiring, no learning | 117 | 119 ± 13 |

Two findings on the way: (1) random epsilon exploration is lethal here (a random
lane change into a train), so epsilon is 0.02 and exploration is left to the
reflex; (2) the five action MBON types receive up to 23x different structural
KC input, so without per-MBON output normalisation the wiring, not learning,
chose the action (untrained real wiring survived 117 vs 170 for random). With
normalisation real wiring learns and edges out random wiring.

Curves: `data/subway/sim_<arm>_s<seed>.png`, summary `data/subway/sim_summary.json`.

## Train vision without ML (2026-09-14)

`occupancy_detections` reads the fraction of track-coloured pixels (tan
sleepers, grey rails) per lane x depth cell: open cells read 0.7-1.0,
cells covered by a train/wall/crate read below 0.45. Blind in teal tunnels.
Combined with the stripe-barrier heuristic and a never-zero dodge reflex
(blocker/(1+side load)), live mean survival went 0.9 s -> 2.7 s (best 7.6 s).
Four other fixes mattered as much: roadside umbrellas are no longer clipped
into lane 2 (off-track detections discarded), the eligibility trace records
'none' so idle crashes punish idleness (otherwise the fly learns helplessness),
the trace window is 0.6 s so punishment lands on the fatal decision, and OCR
score glitches are clamped before they reach the reward.

## Session semantics and start-zone fixes (2026-09-14)

One RUN click = at least 30 s of accumulated play: crashes auto-restart and
the server parks the game (lets a leftover run die) before and after each
session, so every episode starts a fresh run from the station. Restart
overhead is ~9 s (the game's own Save-me countdown plus menus). The Poki
'Sign in' dialog is dismissed via DOM close buttons or card-corner clicks;
never click card centres. The station platform floor misreads as covered
lanes, so occupancy train detections are suppressed for the first 1.0 s of
a run (1.8 s hid the real station-exit train). The circuit head's deviation
is damped 0.5x (running-mean drift caused one action to be spammed to
death), the hold-lane floor is 0.3 and the blocker reflex is proximity
weighted [1.5, 1.2, 0.8]. Live: ~3 s mean per episode, best 7.6 s; this is
the heuristic-vision ceiling, YOLO weights are the next lever.

## Live loop status

Verified headless on this Mac: canvas found in ~11 s, 57 fps capture, keys
move the player, tutorial prompts are followed, crash -> "Save me!" -> results
-> restart. Without detector weights the fly only sees its lane and speed,
so it plays by the reflex on an empty vector (i.e. nothing) and dies at the
first obstacle after the tutorial. That is the expected baseline.

## Detector (needs a free Roboflow account)

1. Export "subway surfers object dataset" (horizon, 2,328 images) as YOLOv8.
2. `uv run flyclash surf-dataset --roboflow-export <folder>` (classes mapped
   to train / wall / boxtrain / low_barrier / high_barrier / coin / player / powerup).
3. `uv run flyclash surf-train --epochs 60` -> `assets/yolo/subway.pt` (Apple GPU, ~10 min).
4. If Poki frames differ from the mobile screenshots in that set, collect
   `uv run flyclash surf-collect --frames 300`, auto-label in Roboflow with the
   first model, merge, retrain.

Then `uv run flyclash surf-run --episodes 50` learns live; `--demo` streams
the soma cloud. Baselines for the writeup: `--no-learning`, and `surf-sim`
random wiring.
