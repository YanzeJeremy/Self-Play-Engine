"""The prediction protocol through the Runner: legacy prompts preserved byte for byte when off,
forecasts that never change play, forecast-to-decision association (thinking, retries,
fallbacks, simultaneous play), call ids, and checkpoint fingerprints."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from self_play.chat_model import ChatModel, ModelCallFailed, ScriptedChatModel
from self_play.checkpoint import run_fingerprint
from self_play.game import DialogueMode, DialoguePhase, Game
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.copycat_rps.prompts import answer_request
from self_play.prediction import PredictionSpec
from self_play.prompting import PREDICTION_SYSTEM_PROMPT, SYSTEM_PROMPT
from self_play.runner import Runner

sys.path.insert(0, str(Path(__file__).parent))
from legacy_prompt_game import FIXTURE, REPLIES, play_legacy_game  # noqa: E402

MODE = "opponent_action_v1"
LEGACY = json.loads(Path(FIXTURE).read_text())
ANSWER_REQUEST = answer_request(legal_actions=["rock", "paper", "scissors"])


def with_forecast(reply: str, forecast: str = '{"rock": 0.2, "paper": 0.5, "scissors": 0.3}') -> str:
    """The same reply with a forecast block before its answer."""
    return reply.replace("<answer>", f"<prediction>{forecast}</prediction>\n<answer>", 1)


def predicting_replies(forecast: str = '{"rock": 0.2, "paper": 0.5, "scissors": 0.3}') -> list[str]:
    calls = LEGACY["calls"]
    return [with_forecast(r, forecast) if kind == "answer" and "<answer>" in r else r for r, kind in zip(REPLIES, calls)]


def play(*, replies, path=None, rounds=8, **kwargs):
    model = ScriptedChatModel(replies=replies)
    runner = Runner(
        game=CopycatRPSGame(game_config={"seed": 5, "rounds": rounds}),
        models={"player": model}, max_retries=1, checkpoint_path=path, **kwargs,
    )
    result = asyncio.run(runner.run())
    return runner, model, result


# --- prompts --------------------------------------------------------------------------

def test_off_mode_prompts_are_byte_identical_to_the_pre_protocol_engine():
    _, model = play_legacy_game()
    assert model.calls == LEGACY["messages"]
    _, explicit_off = play_legacy_game(prediction_mode="off")
    assert explicit_off.calls == LEGACY["messages"]


def test_prediction_mode_changes_only_move_calls():
    runner, model, _ = play(replies=predicting_replies(), prediction_mode=MODE)
    assert len(model.calls) == len(LEGACY["messages"])  # same number and schedule of calls
    for kind, sent, legacy in zip(LEGACY["calls"], model.calls, LEGACY["messages"]):
        if kind != "answer":
            assert sent == legacy  # thinking and talk: system prompt and instruction unchanged
            continue
        assert sent[0]["content"] == PREDICTION_SYSTEM_PROMPT and legacy[0]["content"] == SYSTEM_PROMPT
        request = runner.game.render_prediction_request(spec=PredictionSpec("opponent_action", "", ("rock", "paper", "scissors"), 0))
        assert legacy[1]["content"].count(ANSWER_REQUEST) == 1
        assert sent[1]["content"] == legacy[1]["content"].replace(ANSWER_REQUEST, f"{ANSWER_REQUEST}\n\n{request}")


def test_the_request_names_the_labels_but_no_opponent_strategy():
    game = CopycatRPSGame(game_config={"seed": 0, "rounds": 2})
    spec = game.get_prediction_spec(player_id="player", observation=game.get_observation(player_id="player"), target_id="opponent_action")
    text = game.render_prediction_request(spec=spec).lower()
    assert all(f'"{label}"' in text for label in ("rock", "paper", "scissors"))
    for word in ("copy", "copies", "cycle", "previous", "repeat", "mirror", "last move", "0.3", "0.5"):
        assert word not in text


def test_forecasts_never_reach_later_prompts():
    forecast = '{"rock": 0.123, "paper": 0.456, "scissors": 0.421}'
    _, model, _ = play(replies=predicting_replies(forecast), prediction_mode=MODE)
    for messages in model.calls:
        assert "0.123" not in messages[1]["content"] and "<prediction>{" not in messages[1]["content"]


def test_the_first_forecast_cannot_depend_on_the_hidden_opening_move():
    prompts = []
    for opening in ("rock", "paper", "scissors"):
        game = CopycatRPSGame(game_config={"seed": 0, "rounds": 2})
        game.state["opening_move"] = opening
        model = ScriptedChatModel(replies=["<answer>notes</answer>", "<answer>SILENCE</answer>", "<answer>rock</answer>",
                                           "<answer>notes</answer>", "<answer>rock</answer>"])
        asyncio.run(Runner(game=game, models={"player": model}, prediction_mode=MODE).run())
        prompts.append(model.calls[2])
    assert prompts[0] == prompts[1] == prompts[2]


# --- play is unchanged by forecasts ------------------------------------------------------

@pytest.mark.parametrize("forecast", ['{"rock": 0.2, "paper": 0.5, "scissors": 0.3}', "not json", '{"rock": 1}'])
def test_forecast_validity_never_changes_the_game(forecast):
    _, _, legacy = play(replies=REPLIES)
    runner, _, result = play(replies=predicting_replies(forecast), prediction_mode=MODE)
    assert result == legacy
    reference, _ = play_legacy_game()
    assert runner.game.state == reference.game.state and runner.invalid == reference.invalid


# --- forecast-to-decision association ------------------------------------------------------

def test_each_decision_carries_its_final_attempts_forecast():
    replies = predicting_replies()
    # round 2: the illegal "lizard" move carries a forecast of certainty in scissors, its retry the default one
    lizard = LEGACY["calls"].index("answer", LEGACY["calls"].index("answer") + 1)
    replies[lizard] = with_forecast(REPLIES[lizard], '{"rock": 0, "paper": 0, "scissors": 1}')
    runner, _, _ = play(replies=replies, prediction_mode=MODE)
    decisions = runner.prediction_decisions()
    history = runner.game.state["history"]
    assert [d["decision_id"] for d in decisions] == list(range(8))
    assert [d["executed_action"] for d in decisions] == [h["player"] for h in history]
    assert [d["observed_target"] for d in decisions] == [h["opponent"] for h in history]
    assert [d["step_id"] for d in decisions] == [i for i, s in enumerate(runner.game.steps) if s["action"] != "THINK"]
    assert [d["round_reward"] for d in decisions] == [{"player": 1.0, None: 0.5, "opponent": 0.0}[h["winner"]] for h in history]
    # round 2 was retried: the decision is scored on the retry's forecast, and both attempts are listed
    assert len(decisions[1]["attempt_call_ids"]) == 2 and decisions[1]["call_id"] == decisions[1]["attempt_call_ids"][-1]
    assert decisions[1]["prediction_probs"] == {"rock": 0.2, "paper": 0.5, "scissors": 0.3}
    # round 5 fell back after two illegal moves: the last attempt's forecast is kept, the action marked
    assert decisions[4]["action_source"] == "fallback" and decisions[4]["prediction_status"] == "valid"
    assert all(d["prediction_status"] == "valid" and d["schema_version"] == 1 for d in decisions)


def test_call_ids_number_every_call_in_order():
    runner, _, _ = play(replies=predicting_replies(), prediction_mode=MODE)
    ids = [e["call_id"] for e in runner.record if e["kind"] == "call"]
    assert ids == [f"call-{i}" for i in range(len(LEGACY["calls"]))]


def test_off_mode_has_no_decisions_and_no_forecast_fields():
    runner, _ = play_legacy_game()
    assert runner.prediction_decisions() == []
    assert not any("prediction" in e for e in runner.record)


class TwoPlayerRPS(Game):
    """Both players move simultaneously each round; each predicts the other's move."""

    name = "two_player_rps"
    prediction_targets = ("opponent_action",)

    def initial_state(self):
        return {"rounds": [], "current": {}}

    @property
    def player_ids(self):
        return ["a", "b"]

    def _get_dialogue_phase(self):
        played = len(self.state["rounds"])
        if played >= self.game_config["rounds"]:
            return None
        waiting = tuple(p for p in self.player_ids if p not in self.state["current"])
        return DialoguePhase(phase_id=played, phase="round", mode=DialogueMode.SIMULTANEOUS, eligible=waiting)

    def _get_legal_actions(self, *, player_id):
        return [] if player_id in self.state["current"] else ["rock", "paper", "scissors"]

    def get_observation(self, *, player_id):
        return {"round": len(self.state["rounds"]), "history": [dict(r) for r in self.state["rounds"]]}

    def get_outcome(self):
        return {"done": self._get_dialogue_phase() is None, "rewards": {"a": 0.5, "b": 0.5}}

    def get_rules(self, *, observation):
        return "Play rock paper scissors."

    def render_observation(self, *, observation):
        return f"Round {observation['round']}. History: {observation['history']}"

    def render_answer_request(self, *, legal_actions):
        return "Your move: " + ", ".join(legal_actions)

    def parse_answer(self, *, player_id, answer):
        return answer.strip().lower()

    def _apply(self, *, player_id, action):
        self.state["current"][player_id] = action
        if len(self.state["current"]) == 2:
            self.state["rounds"].append(self.state["current"])
            self.state["current"] = {}
        return []

    def get_prediction_spec(self, *, player_id, observation, target_id):
        return PredictionSpec("opponent_action", "the other player's move", ("rock", "paper", "scissors"), observation["round"])

    def resolve_prediction_target(self, *, player_id, step_id, target_id):
        index = sum(1 for s in self.steps[:step_id] if s["action"] != "THINK") // 2  # two moves per round
        other = "b" if player_id == "a" else "a"
        rounds = self.state["rounds"]
        return rounds[index][other] if index < len(rounds) else None


class EchoForecast(ChatModel):
    """Notes for thinking turns; a move with a forecast naming `me` as certain otherwise."""

    def __init__(self, *, me, moves):
        self.me, self.moves, self.calls = me, list(moves), []

    async def complete(self, *, messages):
        self.calls.append(messages)
        if "This is a thinking turn" in messages[1]["content"]:
            return "<answer>notes</answer>"
        forecast = json.dumps({m: float(m == self.me) for m in ("rock", "paper", "scissors")})
        return f"<prediction>{forecast}</prediction><answer>{self.moves.pop(0)}</answer>"


def test_simultaneous_play_pairs_each_players_forecast_with_its_own_decision():
    game = TwoPlayerRPS(game_config={"seed": 0, "rounds": 2})
    a, b = EchoForecast(me="rock", moves=["rock", "paper"]), EchoForecast(me="scissors", moves=["scissors", "scissors"])
    runner = Runner(game=game, models={"a": a, "b": b}, prediction_mode=MODE)
    asyncio.run(runner.run())
    decisions = runner.prediction_decisions()
    assert [(d["player_id"], d["decision_id"], d["executed_action"], d["observed_target"]) for d in decisions] == [
        ("a", 0, "rock", "scissors"), ("b", 0, "scissors", "rock"), ("a", 1, "paper", "scissors"), ("b", 1, "scissors", "paper"),
    ]
    assert decisions[0]["prediction_probs"]["rock"] == 1.0 and decisions[1]["prediction_probs"]["scissors"] == 1.0
    # both answers of a round were asked before either move was applied
    first_round = [e for e in runner.record if e["kind"] in ("call", "action") and e["phase_id"] == 0 and e.get("call") != "think"
                   and e.get("action") != "THINK"]
    assert [e["kind"] for e in first_round] == ["call", "call", "action", "action"]


class NoPredictionGame(TwoPlayerRPS):
    name = "no_prediction"
    prediction_targets = ()


def test_a_game_without_the_target_refuses_the_mode():
    with pytest.raises(ValueError, match="does not provide"):
        Runner(game=NoPredictionGame(game_config={"seed": 0, "rounds": 1}),
               models={"a": ScriptedChatModel(replies=[]), "b": ScriptedChatModel(replies=[])}, prediction_mode=MODE)
    with pytest.raises(ValueError, match="unknown prediction_mode"):
        Runner(game=CopycatRPSGame(game_config={"seed": 0, "rounds": 1}),
               models={"player": ScriptedChatModel(replies=[])}, prediction_mode="opponent_action_v2")


# --- checkpoints ---------------------------------------------------------------------------

class CrashingChatModel(ChatModel):
    def __init__(self, *, replies, fail_at):
        self.inner, self.fail_at = ScriptedChatModel(replies=replies), fail_at

    async def complete(self, *, messages):
        if len(self.inner.calls) == self.fail_at:
            raise ModelCallFailed("simulated crash")
        return await self.inner.complete(messages=messages)


def fingerprint(**changes) -> dict[str, Any]:
    base = {"game": CopycatRPSGame(game_config={"seed": 5, "rounds": 8}), "prediction_mode": MODE,
            "agents": {"seat0": {"model": "m", "sampling": {"temperature": 0.0}}}, "runner_settings": {"max_retries": 1}}
    return run_fingerprint(**{**base, **changes})


@pytest.mark.parametrize("fail_at", [0, 2, 4, 5, 13, 21])
def test_a_resumed_prediction_game_is_identical(tmp_path, fail_at):
    replies = predicting_replies()
    reference, _, expected = play(replies=replies, prediction_mode=MODE)
    path = tmp_path / "game.json"
    with pytest.raises(ModelCallFailed):
        model = CrashingChatModel(replies=replies, fail_at=fail_at)
        asyncio.run(Runner(game=CopycatRPSGame(game_config={"seed": 5, "rounds": 8}), models={"player": model},
                           max_retries=1, checkpoint_path=path, prediction_mode=MODE, fingerprint=fingerprint()).run())
    resumed, model, result = play(replies=replies[fail_at:], path=path, prediction_mode=MODE, fingerprint=fingerprint())
    assert resumed.resumed and len(model.calls) == len(replies) - fail_at
    assert result == expected
    assert resumed.record == json.loads(json.dumps(reference.record))  # call ids and forecasts included
    assert resumed.prediction_decisions() == reference.prediction_decisions()


def test_a_finished_checkpoint_is_reused_only_under_the_same_fingerprint(tmp_path):
    path = tmp_path / "game.json"
    play(replies=predicting_replies(), path=path, prediction_mode=MODE, fingerprint=fingerprint())
    again, _, _ = play(replies=[], path=path, prediction_mode=MODE, fingerprint=fingerprint())  # any call would raise
    assert again.resumed
    for other in (fingerprint(prediction_mode="off"),
                  fingerprint(agents={"seat0": {"model": "other", "sampling": {"temperature": 0.0}}}),
                  fingerprint(agents={"seat0": {"model": "m", "sampling": {"temperature": 1.0}}}),
                  fingerprint(runner_settings={"max_retries": 0})):
        with pytest.raises(ValueError, match="different run condition"):
            play(replies=[], path=path, prediction_mode=MODE, fingerprint=other)


def test_a_legacy_checkpoint_needs_explicit_permission(tmp_path):
    path = tmp_path / "game.json"
    play(replies=REPLIES, path=path)
    snapshot = json.loads(path.read_text())
    del snapshot["fingerprint"]  # as written before fingerprints existed
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="without a run fingerprint"):
        play(replies=[], path=path, fingerprint=fingerprint(prediction_mode="off"))
    again, _, _ = play(replies=[], path=path, fingerprint=fingerprint(prediction_mode="off"), allow_legacy_checkpoint=True)
    assert again.resumed


def test_the_template_hash_follows_the_prompt_texts(monkeypatch):
    import self_play.prompting as prompting
    before = fingerprint()["prompt_template_hash"]
    monkeypatch.setattr(prompting, "PREDICTION_SYSTEM_PROMPT", prompting.PREDICTION_SYSTEM_PROMPT + " ")
    assert fingerprint()["prompt_template_hash"] != before


def test_a_malformed_forecast_never_stops_a_valid_move(monkeypatch):
    replies = predicting_replies('{"rock": ' + "9" * 400 + ', "paper": 0, "scissors": 0}')
    runner, _, result = play(replies=replies, prediction_mode=MODE)
    _, _, legacy = play(replies=REPLIES)
    assert result == legacy  # every move executed as without forecasts
    assert {d["prediction_status"] for d in runner.prediction_decisions()} == {"invalid_values"}

    import self_play.runner as runner_module

    def broken_parser(**kwargs):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(runner_module, "parse_prediction", broken_parser)
    runner, _, result = play(replies=predicting_replies(), prediction_mode=MODE)
    assert result == legacy
    decisions = runner.prediction_decisions()
    assert {d["prediction_status"] for d in decisions} == {"parser_error"}
    call = next(e for e in runner.record if e["kind"] == "call" and e["call"] == "answer")
    assert "parser bug" in call["prediction"]["error"]
