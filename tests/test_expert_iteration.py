"""Expert iteration (variant A): the hidden-rule game, its configs, data selection, the SFT config,
the adapter export, and the end-to-end pipeline script with stand-ins for vLLM, eval.sh and sft."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from self_play.chat_model import ChatModel, ScriptedChatModel
from self_play.games import GAMES
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.hidden_rule_rps import RULES, HiddenRuleRPSGame
from self_play.runner import Runner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}
MOVES = ("rock", "paper", "scissors")


def run(*args, env=None):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})


class MovesModel(ChatModel):
    """Notes for thinking turns, SILENCE for talk, then the given moves in order; a move given
    as None replies without an answer tag."""

    def __init__(self, *, moves):
        self.moves = list(moves)
        self.last_call = {"provider_calls": 1, "prompt_tokens": 10, "cached_input_tokens": 0, "completion_tokens": 5,
                          "finish_reason": "stop", "start": 0.0, "end": 1.0}

    async def complete(self, *, messages):
        user = messages[-1]["content"]
        if "This is a thinking turn" in user:
            return "<thinking>t</thinking><answer>my notes</answer>"
        if "Before you decide" in user:
            return "<answer>SILENCE</answer>"
        move = self.moves.pop(0)
        return "no tags" if move is None else f"<thinking>m</thinking><answer>{move}</answer>"


def play(*, rule, moves, seed=0):
    game = HiddenRuleRPSGame(game_config={"seed": seed, "rounds": len(moves), "rule": rule})
    runner = Runner(game=game, models={"player": MovesModel(moves=moves)})
    asyncio.run(runner.run())
    return runner


# --- the game ------------------------------------------------------------------------------------

def test_every_rule_plays_as_documented():
    mine = ["rock", "paper", "paper", "scissors"]
    for rule, expected in {
        "copy_last": lambda o, m, t: m[-1],
        "counter_last": lambda o, m, t: BEATEN_BY[m[-1]],
        "beaten_by_last": lambda o, m, t: {v: k for k, v in BEATEN_BY.items()}[m[-1]],
        "copy_two_back": lambda o, m, t: m[-2] if len(m) >= 2 else o,
        "repeat_self": lambda o, m, t: o,
        "cycle": lambda o, m, t: BEATEN_BY[t[-1]],
    }.items():
        history = play(rule=rule, moves=mine).game.state["history"]
        opening = history[0]["opponent"]
        for k in range(1, len(history)):
            played = [h["player"] for h in history[:k]]
            theirs = [h["opponent"] for h in history[:k]]
            assert history[k]["opponent"] == expected(opening, played, theirs), (rule, k)
    assert set(RULES) == {"copy_last", "counter_last", "beaten_by_last", "copy_two_back", "repeat_self", "cycle"}
    assert GAMES["hidden_rule_rps"] is HiddenRuleRPSGame
    with pytest.raises(ValueError, match="rule must be one of"):
        HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 2, "rule": "telepathy"})


def test_copy_last_is_copycat_prompt_for_prompt():
    replies = ["<answer>n</answer>", "<answer>SILENCE</answer>", "<answer>rock</answer>",
               "<answer>n</answer>", "<answer>paper</answer>", "<answer>n</answer>", "<answer>SILENCE</answer>", "<answer>scissors</answer>"]
    sent = []
    for game in (CopycatRPSGame(game_config={"seed": 4, "rounds": 3}),
                 HiddenRuleRPSGame(game_config={"seed": 4, "rounds": 3, "rule": "copy_last"})):
        model = ScriptedChatModel(replies=list(replies))
        asyncio.run(Runner(game=game, models={"player": model}).run())
        sent.append((model.calls, game.state["history"]))
    assert sent[0] == sent[1]


def test_the_rotation_only_solves_copycat():
    def rotation(n, first="rock"):
        moves = [first]
        while len(moves) < n:
            moves.append(BEATEN_BY[moves[-1]])
        return moves
    outcome = {rule: [h["winner"] for h in play(rule=rule, moves=rotation(9)).game.state["history"][1:]] for rule in RULES}
    assert set(outcome["copy_last"]) == {"player"}
    assert set(outcome["counter_last"]) == {None}
    assert set(outcome["beaten_by_last"]) == {"opponent"}
    assert outcome["copy_two_back"][1:] == ["opponent"] * 7


# --- configs -------------------------------------------------------------------------------------

def test_configs_hold_out_seeds_and_reuse_the_copycat_game(tmp_path):
    result = run("python", "scripts/expert_iteration/make_configs.py", "--out_dir", str(tmp_path), "--kind", "eval", "--model", "m",
                 "--rules", "copy_last,cycle", "--games", "40", "--copycat_games", "100", "--rounds", "10", "--skip", "50000", "--temperature", "0")
    assert result.returncode == 0, result.stderr
    assert [line.split()[0] for line in result.stdout.splitlines()] == ["copy_last", "cycle"]
    import tomllib
    copycat = tomllib.loads((tmp_path / "eval_copy_last.toml").read_text())
    cycle = tomllib.loads((tmp_path / "eval_cycle.toml").read_text())
    assert copycat["env"]["game"] == "copycat_rps" and copycat["select"] == {"limit": 100, "skip": 0}
    assert cycle["env"]["game"] == "hidden_rule_rps" and cycle["env"]["game_config"]["rule"] == "cycle"
    assert cycle["select"] == {"limit": 40, "skip": 50000} and cycle["env"]["prediction_mode"] == "off"


def test_the_base_copycat_evaluation_is_the_b0_baseline_condition(tmp_path):
    from run_identity import differences, requested_identity
    run("python", "scripts/expert_iteration/make_configs.py", "--out_dir", str(tmp_path), "--kind", "eval", "--model",
        "google/gemma-4-31b-it", "--rules", "copy_last", "--games", "40", "--copycat_games", "100", "--rounds", "10", "--temperature", "0")
    assert differences(a=requested_identity(config_path=tmp_path / "eval_copy_last.toml"),
                       b=requested_identity(config_path=ROOT / "eval_configs/prediction_b0_t0_r10.toml")) == []


# --- selection -----------------------------------------------------------------------------------

def make_run(*, tmp_path, name, rule, games):
    """A complete generation run whose games are given as lists of moves (seed i = games[i])."""
    from fake_runs import write_run
    cfg_dir = tmp_path / "cfg"
    run("python", "scripts/expert_iteration/make_configs.py", "--out_dir", str(cfg_dir), "--kind", "gen", "--model", "google/gemma-4-31b-it",
        "--rules", rule, "--games", str(len(games)), "--rounds", str(len(games[0])))
    rows = []
    for seed, moves in enumerate(games):
        runner = Runner(game=GAMES["hidden_rule_rps"](game_config={"seed": seed, "rounds": len(moves), "rule": rule}),
                        models={"player": MovesModel(moves=moves)})
        result = asyncio.run(runner.run())
        rows.append({"ok": True, "traces": [{"info": {"game": {
            "game": "hidden_rule_rps", "game_config": runner.game.game_config, "seats": {"player": "seat0"}, "resumed": False,
            "outcome": result["outcome"], "steps": runner.game.steps, "step_rewards": result["step_rewards"],
            "invalid": result["invalid"], "record": runner.record}}, "calls": [], "nodes": []}]})
    return write_run(runs_dir=tmp_path / "runs", name=name, config_path=cfg_dir / f"gen_{rule}.toml", games=0, extra_rows=rows)


def select(*runs, out, extra=()):
    args = ["python", "scripts/expert_iteration/select_sft_data.py", "--out_dir", str(out), "--hard_min", "0",
            "--target_min", "1", "--target_max", "1000", "--val_fraction", "0"]
    for r in runs:
        args += ["--runs", str(r)]
    result = run(*args, *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    meta = [json.loads(line) for line in (out / "meta" / "train.jsonl").open()]
    data = [json.loads(line) for line in (out / "dataset" / "train.jsonl").open()]
    return meta, data, json.loads((out / "stats.json").read_text())


def test_selection_keeps_both_calls_of_rounds_that_kept_winning(tmp_path):
    # repeat_self: the opponent always plays its opening move; with seed 0 that is:
    opening = HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 1, "rule": "repeat_self"}).state["opening_move"]
    win, lose = BEATEN_BY[opening], {v: k for k, v in BEATEN_BY.items()}[opening]
    # rounds 1..8: win (round 1, chance: never kept), lose, win x5, lose
    run_dir = make_run(tmp_path=tmp_path, name="gen", rule="repeat_self", games=[[win, lose, win, win, win, win, win, lose]])
    meta, data, stats = select(run_dir, out=tmp_path / "data", extra=("--window", "3", "--min_window", "2", "--thresholds", "0.6"))
    rounds = sorted({m["round"] for m in meta})
    # round k (1-based) qualifies if won and >= 60% of rounds k..k+2 won, with at least 2 rounds left:
    # round 7's window (rounds 7-8: win, loss) is 50%
    assert rounds == [3, 4, 5, 6]
    assert sorted(m["call"] for m in meta if m["round"] == 3) == ["answer", "think"]
    answer = next(d for d, m in zip(data, meta) if m["round"] == 3 and m["call"] == "answer")
    assert answer["completion"][0]["content"].endswith("[STOP]") and f"<answer>{win}</answer>" in answer["completion"][0]["content"]
    assert answer["prompt"][0]["role"] == "system" and "Next is round 3 of 8" in answer["prompt"][1]["content"]
    assert set(data[0]) == {"prompt", "completion"}  # only the columns SFT reads
    assert stats["train_examples"] == len(data) == 8


def test_selection_drops_unusable_calls_and_caps_rules(tmp_path):
    opening = HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 1, "rule": "repeat_self"}).state["opening_move"]
    win = BEATEN_BY[opening]
    # an unreadable move in round 3 falls back: that round's move cannot be an example, its thinking can
    run_dir = make_run(tmp_path=tmp_path, name="gen", rule="repeat_self", games=[[win, win, None, win, win, win]])
    meta, _, stats = select(run_dir, out=tmp_path / "data", extra=("--window", "1", "--min_window", "1", "--thresholds", "1.0"))
    fallback_round = [m["call"] for m in meta if m["round"] == 3]
    assert fallback_round == []  # round 3 was not won (the fallback move lost or drew) or its move is unusable
    assert all(m["round"] >= 2 for m in meta)
    capped = select(run_dir, out=tmp_path / "capped", extra=("--window", "1", "--min_window", "1", "--thresholds", "1.0",
                                                              "--target_max", "4", "--max_rule_share", "0.5"))[2]
    assert capped["train_examples"] == 2  # 50% of a 4-example target for the only rule


# --- export and SFT config -----------------------------------------------------------------------

def write_dcp(*, path, rank, drop=None):
    import torch
    from torch.distributed.checkpoint import save
    model = {}
    for proj, (fan_in, fan_out) in {"self_attn.q_proj": (32, 48), "mlp.down_proj": (64, 32)}.items():
        module = f"model.language_model.layers.0.{proj}"
        model[f"{module}.weight"] = torch.randn(fan_out, fan_in)
        model[f"{module}.lora_A.0"] = torch.randn(rank, fan_in)
        model[f"{module}.lora_B.0"] = torch.randn(fan_out, rank)
    if drop:
        del model[drop]
    save({"app": {"model": model}}, checkpoint_id=str(path))
    return model


def export(ckpt, out, rank):
    return run("python", "scripts/expert_iteration/export_lora_adapter.py", "--ckpt", str(ckpt), "--out_dir", str(out),
               "--base_model", "google/gemma-4-31b-it", "--rank", str(rank), "--alpha", str(2 * rank))


def test_the_adapter_export_writes_prime_rls_adapter_format(tmp_path):
    import torch
    from safetensors.torch import load_file
    write_dcp(path=tmp_path / "run/checkpoints/step_3/trainer", rank=8)
    model = write_dcp(path=tmp_path / "run/checkpoints/step_12/trainer", rank=8)
    result = export(tmp_path / "run", tmp_path / "adapter", 8)
    assert result.returncode == 0, result.stderr
    assert "step_12" in result.stdout  # the latest step
    adapter = load_file(tmp_path / "adapter/adapter_model.safetensors")
    assert sorted(adapter) == sorted(f"model.language_model.layers.0.{p}.lora_{w}.weight"
                                     for p in ("self_attn.q_proj", "mlp.down_proj") for w in "AB")
    key = "model.language_model.layers.0.mlp.down_proj.lora_A"
    assert adapter[f"{key}.weight"].dtype == torch.bfloat16
    assert torch.equal(adapter[f"{key}.weight"], model[f"{key}.0"].bfloat16())
    config = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert config["r"] == 8 and config["lora_alpha"] == 16 and config["target_modules"] == ["down_proj", "q_proj"]
    assert export(tmp_path / "run", tmp_path / "bad_rank", 4).returncode == 1
    write_dcp(path=tmp_path / "half/step_1/trainer", rank=8, drop="model.language_model.layers.0.mlp.down_proj.lora_B.0")
    result = export(tmp_path / "half", tmp_path / "half_adapter", 8)
    assert result.returncode == 1 and "has only ['lora_A']" in result.stderr


@pytest.mark.workflow
def test_the_sft_config_passes_prime_rls_validation(tmp_path):
    opening = HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 1, "rule": "repeat_self"}).state["opening_move"]
    run_dir = make_run(tmp_path=tmp_path, name="gen", rule="repeat_self", games=[[BEATEN_BY[opening]] * 6] * 4)
    select(run_dir, out=tmp_path / "data", extra=("--thresholds", "0.6"))
    result = run("python", "scripts/expert_iteration/make_sft_config.py", "--data_dir", str(tmp_path / "data"),
                 "--out", str(tmp_path / "sft.toml"), "--output_dir", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["max_steps"] >= 1
    result = subprocess.run(["sft", "@", str(tmp_path / "sft.toml"), "--dry-run"], cwd=ROOT / "prime-rl", capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


# --- the pipeline ---------------------------------------------------------------------------------

@pytest.mark.workflow
def test_the_pipeline_runs_resumes_and_guards_its_settings(tmp_path):
    env = {"EI_NO_GPU": "1", "EI_ROOT": str(tmp_path), "EVAL_SCRIPT": "tests/stub_eval.sh", "EI_SFT_CMD": str(ROOT / "tests/stub_sft.sh")}
    first = run("bash", "scripts/run_expert_iteration.sh", "--smoke", "true", "--name", "t", env=env)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-2000:]
    exp = tmp_path / "t-smoke"
    for stage in ("generate", "select", "sft", "export", "evaluate", "summarize"):
        assert (exp / "stages" / f"{stage}.done").exists()
    summary = json.loads((exp / "summary.json").read_text())
    assert [r["rule"] for r in summary["results"]][0] == "copy_last" and not any(r["problems"] for r in summary["results"])
    assert summary["sft_arm"].startswith("sft-")

    again = run("bash", "scripts/run_expert_iteration.sh", "--smoke", "true", "--name", "t", env=env)
    assert again.returncode == 0 and "== stage" not in again.stdout  # nothing redone

    changed = run("bash", "scripts/run_expert_iteration.sh", "--smoke", "true", "--name", "t", "--lr", "3e-4", env=env)
    assert changed.returncode == 1 and "started with other settings" in changed.stdout

    retrained = run("bash", "scripts/run_expert_iteration.sh", "--smoke", "true", "--name", "t", "--redo_from", "sft", env=env)
    assert retrained.returncode == 0, retrained.stdout[-3000:]
    new_arm = json.loads((exp / "summary.json").read_text())["sft_arm"]
    assert new_arm != summary["sft_arm"]  # a new adapter is evaluated afresh, under its own name
