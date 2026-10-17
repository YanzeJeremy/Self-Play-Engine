"""SelfPlayEnv end to end with fake seats in place of verifiers' agents. Needs verifiers
(Linux, prime-rl's environment); skipped elsewhere."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

from self_play.chat_model import ModelCallFailed  # noqa: E402
from self_play_verifiers.env import SelfPlayEnv, SelfPlayEnvConfig  # noqa: E402
from self_play_verifiers.taskset import SelfPlayData, SelfPlayTask  # noqa: E402


def reply(*, answer):
    return f"<thinking>scratch</thinking><answer>{answer}</answer>[STOP]"


# Copycat over 2 rounds: think, talk + move (ACTION), think, move (RESPONSE_WINDOW).
REPLIES = [reply(answer="notes"), reply(answer="SILENCE"), reply(answer="scissors"),
           reply(answer="NO CHANGE"), reply(answer="rock")]


class FakeTrace:
    def __init__(self):
        self.rewards, self.metrics, self.info = {}, {}, {}

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value

    def record_metric(self, name, value):
        self.metrics[name] = value


class FakeInteraction:
    """Replies in order; after `fail_at` replies the seat's run ends (a terminated segment)."""

    def __init__(self, *, replies, fail_at=None):
        self.replies, self.fail_at, self.turns = list(replies), fail_at, 0
        self.trace = FakeTrace()

    async def turn(self, messages):
        if self.turns == self.fail_at:
            return SimpleNamespace(terminated=True, last_reply="")
        self.turns += 1
        return SimpleNamespace(terminated=False, last_reply=self.replies.pop(0))


class FakeAgent:
    def __init__(self, *, interaction=None):
        self.trainable = True
        self.opened = False
        self._interaction = interaction

    @asynccontextmanager
    async def interaction(self, task):
        self.opened = True
        yield self._interaction


def fake_agents(*, interaction):
    agents = SimpleNamespace(**{f"seat{i}": FakeAgent() for i in range(6)})
    agents.seat0 = FakeAgent(interaction=interaction)
    return agents


def make_env(*, checkpoint_dir=None, **extra):
    config = SelfPlayEnvConfig.model_validate({
        "taskset": {"id": "self-play-verifiers"},
        "game": "copycat_rps",
        "game_config": {"rounds": 2},
        "checkpoint_dir": checkpoint_dir,
        **extra,
    })
    env = SelfPlayEnv(config)
    asyncio.run(env.start())
    return env


TASK = SelfPlayTask(SelfPlayData(name="game#0", prompt=None, info={"seed": 0}))


def test_a_game_runs_through_seat0_and_fills_its_trace():
    env = make_env()
    interaction = FakeInteraction(replies=REPLIES)
    agents = fake_agents(interaction=interaction)
    asyncio.run(env.run(TASK, agents))
    trace = interaction.trace
    assert interaction.turns == len(REPLIES)
    assert set(trace.rewards) == {"game"} and trace.metrics == {"invalid": 0.0}
    game = trace.info["game"]
    assert game["game"] == "copycat_rps" and game["game_config"] == {"rounds": 2, "seed": 0}
    assert game["seats"] == {"player": "seat0"} and game["player_id"] == "player"
    assert [e["call"] for e in game["record"] if e["kind"] == "call"] == ["think", "talk", "answer", "think", "answer"]
    assert [(s["call"], s["step_id"]) for s in trace.info["step_rewards"]] == [
        ("think", 0), ("talk", 1), ("answer", 1), ("think", 2), ("answer", 3)
    ]
    assert not any(getattr(agents, f"seat{i}").opened for i in range(1, 6))  # unused seats stay closed


def test_a_crashed_game_resumes_from_its_checkpoint(tmp_path):
    env = make_env(checkpoint_dir=str(tmp_path))
    with pytest.raises(ModelCallFailed):
        asyncio.run(env.run(TASK, fake_agents(interaction=FakeInteraction(replies=REPLIES, fail_at=3))))
    assert len(list(tmp_path.glob("*.json"))) == 1 and len(list(tmp_path.glob("meta_*.yaml"))) == 1

    rest = FakeInteraction(replies=REPLIES[3:])
    asyncio.run(env.run(TASK, fake_agents(interaction=rest)))
    assert rest.turns == len(REPLIES) - 3  # only the calls the crash left unmade
    game = rest.trace.info["game"]
    assert game["resumed"]
    reference = FakeInteraction(replies=REPLIES)
    asyncio.run(make_env().run(TASK, fake_agents(interaction=reference)))
    assert game["record"] == reference.trace.info["game"]["record"]
    assert rest.trace.rewards == reference.trace.rewards


def test_pinned_seats_are_not_trainable():
    env = make_env(seat1={"model": "some/other-model"})
    agents = fake_agents(interaction=None)
    asyncio.run(env.setup(agents))
    assert agents.seat0.trainable and not agents.seat1.trainable


def predicting(answer):
    return f"<thinking>scratch</thinking><prediction>{{\"rock\": 0.5, \"paper\": 0.25, \"scissors\": 0.25}}</prediction><answer>{answer}</answer>[STOP]"


# Copycat over 2 rounds, prediction on: think, talk, move (with forecast), think, move (no forecast).
PREDICTING_REPLIES = [reply(answer="notes"), reply(answer="SILENCE"), predicting("scissors"),
                      reply(answer="NO CHANGE"), reply(answer="rock")]


def test_prediction_mode_records_decisions_and_metrics_but_not_rewards():
    plain = FakeInteraction(replies=REPLIES)
    asyncio.run(make_env().run(TASK, fake_agents(interaction=plain)))
    interaction = FakeInteraction(replies=PREDICTING_REPLIES)
    asyncio.run(make_env(prediction_mode="opponent_action_v1").run(TASK, fake_agents(interaction=interaction)))
    trace = interaction.trace
    assert trace.rewards == plain.trace.rewards  # forecasts are never rewarded
    predictions = trace.info["predictions"]
    assert predictions["schema_version"] == 1 and predictions["protocol"] == "opponent_action_v1"
    decisions = predictions["decisions"]
    assert [d["prediction_status"] for d in decisions] == ["valid", "missing"]
    assert [d["decision_id"] for d in decisions] == [0, 1]
    assert decisions[1]["observed_target"] == "scissors"  # round 2: the copycat plays the round-1 move
    assert trace.metrics["prediction_coverage"] == 0.5 and trace.metrics["prediction_decisions"] == 2.0
    assert trace.metrics["prediction_brier_coverage_adjusted"] == (trace.metrics["prediction_brier"] + 2.0) / 2
    assert "predictions" not in plain.trace.info and not any(k.startswith("prediction") for k in plain.trace.metrics)
    assert trace.info["game"]["fingerprint"]["prediction_protocol"] == "opponent_action_v1"


def test_prediction_metrics_on_known_forecasts():
    from self_play_verifiers.env import prediction_metrics
    labels = ["rock", "paper", "scissors"]
    decisions = [
        {"action_source": "model", "prediction_status": "valid", "observed_target": "rock", "prediction_labels": labels,
         "prediction_probs": {"rock": 1.0, "paper": 0.0, "scissors": 0.0}},
        {"action_source": "model", "prediction_status": "valid", "observed_target": "paper", "prediction_labels": labels,
         "prediction_probs": {"rock": 1 / 3, "paper": 1 / 3, "scissors": 1 / 3}},
        {"action_source": "fallback", "prediction_status": "invalid_json", "observed_target": "rock", "prediction_labels": labels,
         "prediction_probs": None},
        {"action_source": "forced", "prediction_status": None, "observed_target": "rock", "prediction_labels": None,
         "prediction_probs": None},
    ]
    metrics = prediction_metrics(decisions=decisions)
    assert metrics["prediction_decisions"] == 3.0  # the forced action is excluded
    assert metrics["prediction_coverage"] == pytest.approx(2 / 3)
    assert metrics["prediction_brier"] == pytest.approx((0 + 2 / 3) / 2)
    assert metrics["prediction_accuracy"] == 0.5  # the uniform forecast's tie goes to "rock", but "paper" was observed
    assert metrics["prediction_brier_coverage_adjusted"] == pytest.approx((0 + 2 / 3 + 2) / 3)


def test_switching_condition_on_a_reused_checkpoint_dir_is_refused(tmp_path):
    asyncio.run(make_env(checkpoint_dir=str(tmp_path)).run(TASK, fake_agents(interaction=FakeInteraction(replies=REPLIES))))
    env = make_env(checkpoint_dir=str(tmp_path), prediction_mode="opponent_action_v1")
    with pytest.raises(ValueError, match="different run condition"):
        asyncio.run(env.run(TASK, fake_agents(interaction=FakeInteraction(replies=PREDICTING_REPLIES))))


def test_a_reloaded_config_keeps_the_runs_sampling_for_every_seat():
    # A resumed eval reloads the saved resolved config, where unset seat sampling fields are
    # written out as null; they must not override the run's sampling.
    import json as _json

    from verifiers.v1.configs.agent import resolve_agent

    import verifiers.v1 as vf
    saved = _json.loads(make_env().config.model_dump_json())
    assert saved["seat0"]["sampling"]["temperature"] is None  # what the saved file holds
    reloaded = SelfPlayEnvConfig.model_validate(saved)
    run_sampling = vf.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=2048, top_k=64)
    for config in (make_env().config, reloaded):
        seat = resolve_agent(config.seat0, model="m", sampling=run_sampling)
        assert seat.sampling.model_dump(exclude_none=True) == {
            "temperature": 0.0, "top_p": 0.95, "max_tokens": 2048, "top_k": 64, "stop": ["[STOP]"],
        }
