"""Full attack cycle against a fake game with a fake clock: deadlines and transitions."""

import numpy as np

from flyclash.agent.loop import Agent, State
from flyclash.brain.circuit import Circuit
from flyclash.brain.learn import Plastic, Policy
from flyclash.config import Config
from flyclash.game.actions import Result
from flyclash.game.base_reader import BaseObs, Loot


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeGame:
    def __init__(self, clock, clouds_for=3.0, stars=2):
        self.clock, self.clouds_for, self.stars = clock, clouds_for, stars
        self.calls = []
        self.search_t = None
        self.troops = [10, 10]

    def _c(self, name):
        self.calls.append(name)

    def at_home(self):
        return True

    def open_attack_window(self):
        self._c("attack")
        return True

    def find_match(self):
        self._c("find")
        self.search_t = self.clock.now()
        return True

    def next_base(self):
        self._c("next")
        self.search_t = self.clock.now()
        return True

    def scout_frame(self):
        return (
            np.zeros((100, 200, 3), np.uint8)
            if self.clock.now() - self.search_t >= self.clouds_for
            else None
        )

    def scout_loot(self, frame):
        return Loot(1000, 1000, 0, 10)

    def troop_counts(self):
        return list(self.troops)

    def drop(self, slot, points):
        self._c(f"drop{slot}:{len(points)}")
        self.troops[slot] = 0

    def end_battle(self):
        self._c("end")
        return True

    def result(self):
        return Result(self.stars, 60, 500, 500)

    def return_home(self):
        self._c("home")
        return True


def make_agent(game, clock, cfg, attack_bias):
    c = Circuit.random()
    policy = Policy(
        c,
        Plastic.fresh(c, epsilon=0.0),
        rng=np.random.default_rng(0),
        attack_bias=attack_bias,
    )
    return Agent(
        game,
        policy,
        cfg,
        lambda f, loot: BaseObs(
            np.random.default_rng(0).uniform(0, 1, 53), [], (100, 50), {}, loot
        ),
        clock=clock.now,
        sleep=clock.sleep,
    )


def test_attack_cycle_logs_and_learns(tmp_path):
    clock, cfg = FakeClock(), Config()
    cfg.log_path = tmp_path / "attacks.jsonl"
    game = FakeGame(clock)
    rec = make_agent(game, clock, cfg, attack_bias=1.0).run_once()
    assert (
        rec.attacked and rec.stars == 2 and rec.reward == 2.0
    )  # 2 stars + full loot - army cost
    assert (
        game.calls[:2] == ["attack", "find"]
        and "end" in game.calls
        and game.calls[-1] == "home"
    )
    assert any(c.startswith("drop0") for c in game.calls) and any(
        c.startswith("drop1") for c in game.calls
    )
    assert rec.elapsed_decide < cfg.timing.decide
    assert cfg.log_path.read_text().count("\n") == 1


def test_skips_until_bias_forces_attack(tmp_path):
    clock, cfg = FakeClock(), Config()
    cfg.log_path = tmp_path / "attacks.jsonl"
    game = FakeGame(clock)
    agent = make_agent(game, clock, cfg, attack_bias=-0.1)
    # each skip depresses the avoid synapses a little, so the fly attacks eventually
    rec = agent.run_once()
    assert 1 <= rec.skips < cfg.timing.max_skips and rec.attacked
    assert game.calls.count("next") == rec.skips


def test_skip_cap_forces_attack(tmp_path):
    clock, cfg = FakeClock(), Config()
    cfg.log_path = tmp_path / "attacks.jsonl"
    game = FakeGame(clock)
    rec = make_agent(game, clock, cfg, attack_bias=-10.0).run_once()
    assert rec.skips == cfg.timing.max_skips and rec.attacked


def test_late_decision_attacks_instead_of_next(tmp_path):
    clock, cfg = FakeClock(), Config()
    cfg.log_path = tmp_path / "attacks.jsonl"
    game = FakeGame(clock)
    agent = make_agent(game, clock, cfg, attack_bias=-0.5)
    agent.read_base = lambda f, loot: (
        clock.sleep(30),
        BaseObs(np.ones(53), [], (100, 50), {}, loot),
    )[1]
    rec = agent.run_once()
    assert rec.skips == 0 and rec.attacked and agent.state == State.LEARN
