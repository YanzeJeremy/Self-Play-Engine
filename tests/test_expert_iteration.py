"""Expert iteration (variant A): the hidden-rule game, its configs, data selection, the SFT config,
the adapter export, and the end-to-end pipeline script with stand-ins for vLLM, eval.sh and sft."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from collections import Counter

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
        "copy_three_back": lambda o, m, t: m[-3] if len(m) >= 3 else o,
        "repeat_self": lambda o, m, t: o,
        "cycle": lambda o, m, t: BEATEN_BY[t[-1]],
    }.items():
        history = play(rule=rule, moves=mine).game.state["history"]
        opening = history[0]["opponent"]
        for k in range(1, len(history)):
            played = [h["player"] for h in history[:k]]
            theirs = [h["opponent"] for h in history[:k]]
            assert history[k]["opponent"] == expected(opening, played, theirs), (rule, k)
    assert set(RULES) == {"copy_last", "counter_last", "beaten_by_last", "copy_two_back", "copy_three_back", "repeat_self", "cycle"}
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
    meta, data, stats = select(run_dir, out=tmp_path / "data", extra=("--window", "3", "--min_window", "2", "--thresholds", "0.6", "--max_rule_share", "1.0"))
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


def test_selection_drops_unusable_calls(tmp_path):
    opening = HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 1, "rule": "repeat_self"}).state["opening_move"]
    win = BEATEN_BY[opening]
    # an unreadable move in round 3 falls back: that round is not the model's move and never an example
    run_dir = make_run(tmp_path=tmp_path, name="gen", rule="repeat_self", games=[[win, win, None, win, win, win]])
    meta, _, stats = select(run_dir, out=tmp_path / "data",
                            extra=("--window", "1", "--min_window", "1", "--thresholds", "1.0", "--max_rule_share", "1.0"))
    assert sorted({m["round"] for m in meta}) == [2, 4, 5, 6]
    assert all(m["strict"] for m in meta)
    assert stats["train_by_rule"]["repeat_self"] == {"examples": 8, "unique_games": 1, "strict_examples": 8, "share": 1.0}


def winning_games(*, rule, n, rounds=6):
    """n games that win every round against `rule` (the move that beats what the rule plays)."""
    games = []
    for seed in range(n):
        game = HiddenRuleRPSGame(game_config={"seed": seed, "rounds": rounds, "rule": rule})
        moves = []
        for _ in range(rounds):
            move = BEATEN_BY[game.opponent_move()]
            moves.append(move)
            game.state["history"].append({"player": move, "opponent": game.opponent_move(), "winner": "player"})
        games.append(moves)
    return games


def test_the_rule_share_is_enforced_on_the_final_dataset(tmp_path):
    runs = {rule: make_run(tmp_path=tmp_path, name=f"gen-{rule}", rule=rule, games=winning_games(rule=rule, n=3))
            for rule in ("counter_last", "beaten_by_last", "copy_two_back", "repeat_self", "cycle")}
    args = ("--window", "1", "--min_window", "1", "--thresholds", "1.0", "--target_max", "40")
    meta, data, stats = select(*runs.values(), out=tmp_path / "balanced", extra=args)
    counts = Counter(m["rule"] for m in meta)
    assert len(data) == 40 and set(counts.values()) == {8}  # every opponent, evenly; none trimmed away alphabetically
    assert {r: v["examples"] for r, v in stats["train_by_rule"].items()} == dict(counts)  # reported = written
    assert max(v["share"] for v in stats["train_by_rule"].values()) <= 0.35

    alone = run("python", "scripts/expert_iteration/select_sft_data.py", "--runs", str(runs["repeat_self"]), "--out_dir",
                str(tmp_path / "alone"), "--hard_min", "0", "--val_fraction", "0", "--target_min", "1", *args)
    assert alone.returncode == 1 and "no train set satisfies --max_rule_share" in alone.stderr

    lopsided = make_run(tmp_path=tmp_path, name="gen-cycle-big", rule="cycle", games=winning_games(rule="cycle", n=12))
    meta, _, stats = select(runs["repeat_self"], runs["counter_last"], lopsided, out=tmp_path / "lopsided",
                            extra=("--window", "1", "--min_window", "1", "--thresholds", "1.0", "--target_max", "1000"))
    shares = Counter(m["rule"] for m in meta)
    assert max(shares.values()) / sum(shares.values()) <= 0.35 + 1e-9 and stats["diversity_limited"]


def test_random_play_pass_rates():
    from expert_iteration.select_sft_data import random_pass_rate
    assert abs(random_pass_rate(window=5, threshold=0.8) - 1 / 27) < 1e-9
    assert abs(random_pass_rate(window=5, threshold=0.6) - 33 / 243) < 1e-9


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
    select(run_dir, out=tmp_path / "data", extra=("--thresholds", "0.6", "--max_rule_share", "1.0"))
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
    pipeline = lambda *args, **extra: run("bash", "scripts/run_expert_iteration.sh", "--smoke", "true", "--name", "t", *args,
                                          env={**env, **extra})
    exp = tmp_path / "t-smoke"
    calls = lambda arm="sft": (exp / {"sft": "sft", "rand": "sft-random"}[arm] / "outputs/stub_sft_calls.log").read_text().splitlines()

    # SFT crashes before its first checkpoint, then after one: retries restart clean, then resume
    crashed = pipeline(STUB_SFT_CRASH="before_ckpt")
    assert crashed.returncode == 1 and "SFT (sft) failed" in crashed.stdout
    # a started stage keeps its settings: an interrupted SFT cannot be resumed under another lr
    changed = pipeline("--lr", "3e-4")
    assert changed.returncode == 1 and "started stages were run with other settings" in changed.stdout and "sft: lr=" in changed.stdout
    crashed = pipeline(STUB_SFT_CRASH="after_ckpt")
    assert crashed.returncode == 1 and calls() == ["(fresh)", "--clean"]
    first = pipeline()
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-2000:]
    assert calls() == ["(fresh)", "--clean", "--resume"] and calls("rand") == ["(fresh)"]
    for stage in ("generate", "select", "control", "sft", "export", "evaluate", "summarize"):
        assert (exp / "stages" / f"{stage}.done").exists() and (exp / "stages" / f"{stage}.params").exists()
    control = json.loads((exp / "data-random/stats.json").read_text())
    assert control["mode"] == "random" and control["shortfall"] == {}
    summary = json.loads((exp / "summary.json").read_text())
    assert summary["base_arm"] == "base" and summary["sft_arm"].startswith("sft-")
    assert [r["rule"] for r in summary["results"]][0] == "copy_last" and not any(r["problems"] for r in summary["results"])
    assert list(summary["by_horizon"]) == ["4 rounds", "6 rounds"]
    assert [(d["condition"], d["rule"]) for d in summary["diagnostics"]] == [
        ("shuffled", "copy_last"), ("intervene", "copy_last"), ("switch", "copy_last"), ("lag3", "copy_three_back"),
        ("shuffled", "counter_last")]
    assert not any(d["problems"] for d in summary["diagnostics"])
    versus = json.loads((exp / "summary-sft-vs-rand.json").read_text())
    assert versus["base_arm"].startswith("rand-") and (exp / "summary-rand.md").exists()

    again = pipeline()
    assert again.returncode == 0 and "== stage" not in again.stdout  # nothing redone

    # a finished stage keeps its settings, unless it is redone
    assert pipeline("--lr", "5e-5").returncode == 1
    changed = pipeline("--eval_skip", "60000")  # (smoke mode fixes eval_games itself)
    assert changed.returncode == 1 and "evaluate: eval_skip=" in changed.stdout
    # redoing evaluation under a changed setting archives the affected runs and reuses unchanged ones
    reevaluated = pipeline("--eval_skip", "60000", "--redo_from", "evaluate")
    assert reevaluated.returncode == 0, reevaluated.stdout[-3000:]
    archived = sorted(p.name.rsplit(".", 1)[0] for p in (exp / "runs/archive").iterdir())
    # only runs that used --eval_skip changed: the training rules' (copycat starts at seed 0, diagnostics use --diag_skip)
    assert archived and all("copy_last" not in a and "-diag-" not in a for a in archived)
    assert "t-smoke-eval-base-counter_last" in archived and (exp / "runs/t-smoke-eval-base-copy_last").exists()

    retrained = pipeline("--eval_skip", "60000", "--lr", "3e-4", "--redo_from", "sft")
    assert retrained.returncode == 0, retrained.stdout[-3000:]
    assert "previous SFT folder is kept" in retrained.stdout and list(exp.glob("sft.replaced-*")) and list(exp.glob("sft-random.replaced-*"))
    assert calls() == ["(fresh)"]  # a fresh training folder: no resume, no clean
    new_arm = json.loads((exp / "summary.json").read_text())["sft_arm"]
    assert new_arm != summary["sft_arm"]  # a new adapter is evaluated afresh, under its own name


def test_shared_key_value_layers_serve_the_trained_values(tmp_path, monkeypatch):
    """Gemma 4's full-attention layers (5, 11, ...) have no v_proj: values are the k_proj output, so
    training's k_proj LoRA update reaches the values too. vLLM loads k_proj into both the K and V slots
    of its packed qkv_proj but applies LoRA per slot; this checks with vLLM's own loader and packing
    that the exported adapter gives the V slot exactly the trained values."""
    import torch
    from torch.distributed.checkpoint import save
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import PackedLoRALayerWeights
    from vllm.lora.peft_helper import PEFTHelper
    import vllm.lora.lora_model as vllm_lora_model

    monkeypatch.setattr(vllm_lora_model, "PIN_MEMORY", False)  # pinning needs a GPU; the loading logic does not
    rank, alpha, hidden = 8, 16, 32
    torch.manual_seed(0)
    model, shapes = {}, {0: ("q_proj", "k_proj", "v_proj"), 5: ("q_proj", "k_proj")}  # layer 0 sliding, 5 full (k == v)
    for layer, projections in shapes.items():
        for proj in projections:
            module = f"model.language_model.layers.{layer}.self_attn.{proj}"
            model[f"{module}.lora_A.0"] = torch.randn(rank, hidden)
            model[f"{module}.lora_B.0"] = torch.randn(24, rank)
    save({"app": {"model": model}}, checkpoint_id=str(tmp_path / "ckpt"))
    result = export(tmp_path / "ckpt", tmp_path / "adapter", rank)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "adapter/source.json").read_text())["shared_kv_v_proj_copied_from_k_proj"] == [
        "model.language_model.layers.5.self_attn.v_proj"]

    helper = PEFTHelper.from_local_dir(str(tmp_path / "adapter"), max_position_embeddings=None)
    lora_model = LoRAModel.from_local_checkpoint(
        str(tmp_path / "adapter"), expected_lora_modules={"q_proj", "k_proj", "v_proj"}, peft_helper=helper,
        device="cpu", dtype=torch.float32)

    def served_slots(layer):  # what vLLM's model manager packs into this layer's qkv_proj
        prefix = f"model.language_model.layers.{layer}.self_attn"
        return PackedLoRALayerWeights.pack([lora_model.get_lora(f"{prefix}.{p}") for p in ("q_proj", "k_proj", "v_proj")]).optimize()

    x = torch.randn(hidden)
    w_k = torch.randn(24, hidden)  # the base k_proj, which vLLM loads into both the K and V slots
    trained_key = w_k @ x + (alpha / rank) * model["model.language_model.layers.5.self_attn.k_proj.lora_B.0"] @ (
        model["model.language_model.layers.5.self_attn.k_proj.lora_A.0"] @ x)
    packed = served_slots(5)
    served = lambda slot: w_k @ x + (packed.lora_b[slot] @ (packed.lora_a[slot] @ x) if packed.lora_b[slot] is not None else 0)
    # bf16 storage of the adapter is the only difference
    assert torch.allclose(served(1), trained_key, atol=0.5, rtol=0.02)  # keys
    assert torch.allclose(served(2), trained_key, atol=0.5, rtol=0.02)  # values == keys, as in training
    assert served_slots(0).lora_b[2] is not None  # sliding layers keep their own v_proj

    # without the copy (what an export of k_proj alone gives) the served values miss the whole update
    del lora_model.loras["model.language_model.layers.5.self_attn.v_proj"]
    unfixed = served_slots(5)
    assert unfixed.lora_b[2] is None
    assert not torch.allclose(w_k @ x, trained_key, atol=0.5, rtol=0.02)


# --- second review: adapter checks, diagnostics, the random control ------------------------------------------

def test_the_adapter_check_fails_when_it_cannot_run(tmp_path):
    from fake_runs import write_run  # noqa: F401  (a dataset folder is all the check reads before the server)
    data = tmp_path / "data" / "dataset"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text(json.dumps({"prompt": [{"role": "user", "content": "hi"}],
                                                  "completion": [{"role": "assistant", "content": "<answer>rock</answer>"}]}) + "\n")
    args = ["python", "scripts/expert_iteration/check_adapter.py", "--base_url", "http://127.0.0.1:9/v1", "--base_model",
            "google/gemma-4-31b-it", "--adapter_name", "a", "--data_dir", str(tmp_path / "data"), "--out", str(tmp_path / "check.json")]
    failed = run(*args, "--require_improvement")
    assert failed.returncode == 1 and "could not run" in failed.stdout
    assert json.loads((tmp_path / "check.json").read_text()) | {"error": ""} == {"ran": False, "bypassed": False, "error": ""}
    bypassed = run(*args, "--allow_unchecked")
    assert bypassed.returncode == 0 and "UNCHECKED" in bypassed.stdout
    assert json.loads((tmp_path / "check.json").read_text())["bypassed"] is True


def test_the_export_fidelity_check_catches_a_lost_update(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint import save

    import vllm.lora.lora_model as vllm_lora_model
    from expert_iteration.export_lora_adapter import read_adapter, verify_export

    monkeypatch.setattr(vllm_lora_model, "PIN_MEMORY", False)
    model = {}
    for layer, projs in {0: ("q_proj", "k_proj", "v_proj"), 5: ("q_proj", "k_proj")}.items():
        for proj in projs:
            module = f"model.language_model.layers.{layer}.self_attn.{proj}"
            model[f"{module}.lora_A.0"], model[f"{module}.lora_B.0"] = torch.randn(8, 32), torch.randn(24, 8)
    save({"app": {"model": model}}, checkpoint_id=str(tmp_path / "ckpt"))
    assert export(tmp_path / "ckpt", tmp_path / "adapter", 8).returncode == 0
    fidelity = json.loads((tmp_path / "adapter/fidelity.json").read_text())
    assert fidelity["modules_checked"] == 6 and fidelity["max_relative_error"] < 0.03  # 5 modules + the shared V slot
    trained = read_adapter(ckpt=tmp_path / "ckpt", rank=8, dtype=None)
    shared = ["model.language_model.layers.5.self_attn.v_proj"]
    adapter = load_file(tmp_path / "adapter/adapter_model.safetensors")
    for which in ("A", "B"):  # an export without the shared copy
        del adapter[f"model.language_model.layers.5.self_attn.v_proj.lora_{which}.weight"]
    save_file(adapter, str(tmp_path / "adapter/adapter_model.safetensors"))
    lost = verify_export(out_dir=tmp_path / "adapter", trained=trained, shared=shared, rank=8, alpha=16,
                         targets=["k_proj", "q_proj", "v_proj"])
    assert lost["worst"] == shared[0] and lost["max_relative_error"] == float("inf")


def play_diag(*, moves, **config):
    game = HiddenRuleRPSGame(game_config={"seed": 2, "rounds": len(moves), "rule": "copy_last", **config})
    model = MovesModel(moves=[m for i, m in enumerate(moves) if str(i) not in game.state["forced"]])
    runner = Runner(game=game, models={"player": model})
    asyncio.run(runner.run())
    return runner, model


def test_shuffled_dialogue_keeps_the_claims_and_frees_the_phrases():
    runner, _ = play_diag(moves=["rock"] * 12, dialogue="shuffled")
    lines = [h["opponent_says"] for h in runner.game.state["history"]]
    assert all(line.startswith("I will play") for line in lines[3::4])  # claims stay on their rounds
    generic = [line for i, line in enumerate(lines) if i % 4 != 3]
    assert all(not line.startswith("I will play") for line in generic)
    cycled, _ = play_diag(moves=["rock"] * 12)
    assert generic != [line for i, line in enumerate(h["opponent_says"] for h in cycled.game.state["history"]) if i % 4 != 3]


def test_interventions_force_moves_without_asking_and_show_them():
    runner, model = play_diag(moves=["rock"] * 10, interventions=2)
    forced = runner.game.forced_rounds()
    assert len(forced) == 2 and all(1 <= r <= 7 for r in forced)
    history = runner.game.state["history"]
    for r in forced:
        assert history[r]["player"] == runner.game.state["forced"][str(r)]
        assert history[r + 1]["opponent"] == history[r]["player"]  # the copycat copies the forced move
    sources = [e["source"] for e in runner.record if e["kind"] == "action" and e["action"] != "THINK"]
    assert sources.count("forced") == 2 and len(model.moves) == 0  # the model was never asked those rounds


def test_a_rule_switch_and_the_lag_three_rule():
    rotation = ["rock"]
    while len(rotation) < 12:
        rotation.append(BEATEN_BY[rotation[-1]])
    runner, _ = play_diag(moves=rotation, switch_round=6, rule_after="counter_last")
    winners = [h["winner"] for h in runner.game.state["history"]]
    assert set(winners[1:6]) == {"player"} and set(winners[6:]) == {None}  # the rotation stops working at the switch
    history = play(rule="copy_three_back", moves=["rock", "paper", "scissors", "paper", "rock"]).game.state["history"]
    assert [h["opponent"] for h in history[3:]] == ["rock", "paper"]
    with pytest.raises(ValueError, match="go together"):
        HiddenRuleRPSGame(game_config={"seed": 0, "rounds": 4, "rule": "copy_last", "switch_round": 2})


def test_diagnostic_scores_leave_out_forced_rounds():
    from expert_iteration.summarize import diagnostic_metrics
    runner, _ = play_diag(moves=["rock"] * 10, interventions=2, switch_round=5, rule_after="counter_last")
    game = {"record": runner.record, "game_config": runner.game.game_config}
    metrics = diagnostic_metrics(game=game)
    forced = set(runner.game.forced_rounds())
    won = [h["winner"] == "player" for h in runner.game.state["history"]]
    scored = [w for i, w in enumerate(won) if i >= 1 and i not in forced]
    assert metrics["rounds_won"] == sum(scored) / len(scored)
    after = [won[j] for r in forced for j in (r + 1, r + 2) if j < len(won) and j not in forced]
    assert metrics["recovery"] == sum(after) / len(after)
    assert metrics["won_after_switch"] == sum(w for i, w in enumerate(won) if i >= 5 and i not in forced) / len(
        [i for i in range(5, 10) if i not in forced])


def test_the_random_control_matches_counts_from_the_same_games(tmp_path):
    runs = {rule: make_run(tmp_path=tmp_path, name=f"gen-{rule}", rule=rule, games=winning_games(rule=rule, n=3) + [["rock"] * 6] * 3)
            for rule in ("counter_last", "beaten_by_last", "copy_two_back", "repeat_self", "cycle")}
    common = ["--window", "1", "--min_window", "1", "--thresholds", "1.0", "--target_max", "40", "--val_fraction", "0"]
    meta, _, _ = select(*runs.values(), out=tmp_path / "success", extra=common)
    control = run("python", "scripts/expert_iteration/select_sft_data.py", *sum((["--runs", str(r)] for r in runs.values()), []),
                  "--out_dir", str(tmp_path / "control"), "--mode", "random", "--match_dir", str(tmp_path / "success"),
                  "--hard_min", "0", "--val_fraction", "0")
    assert control.returncode == 0, control.stderr
    control_meta = [json.loads(line) for line in (tmp_path / "control/meta/train.jsonl").open()]
    assert Counter((m["rule"], m["call"]) for m in control_meta) == Counter((m["rule"], m["call"]) for m in meta)
    assert any(not m["won"] for m in control_meta)  # drawn whatever the outcome
    stats = json.loads((tmp_path / "control/stats.json").read_text())
    assert stats["mode"] == "random" and stats["shortfall"] == {} and stats["matched_train_tokens"] > 0
