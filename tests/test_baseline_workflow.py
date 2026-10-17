"""The B0/B1 workflow scripts: run identity, completeness, strict pairing, failure propagation,
reuse validation and endpoint moves (review of aaf998e, items 1, 2 and 4). Uses fake run
directories (tests/fake_runs.py) and a stub in place of eval.sh (tests/stub_eval.sh).

These run the real scripts in subprocesses (a few minutes in all); skip them in a quick
iteration with `pytest -m "not workflow"`."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from fake_runs import play_row, write_run  # noqa: E402
from migrate_endpoint import migrate  # noqa: E402
from run_identity import completeness, differences, identity_from_config, requested_identity, run_identity  # noqa: E402

pytestmark = pytest.mark.workflow

SMOKE_B0, SMOKE_B1 = ROOT / "eval_configs/prediction_smoke_b0.toml", ROOT / "eval_configs/prediction_smoke_b1.toml"


def edited(*, tmp_path, source, old, new):
    """A copy of a config with the setting line starting with `old` changed (comments untouched)."""
    lines = source.read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith(old))
    lines[index] = lines[index].replace(old, new, 1)
    path = tmp_path / source.name
    path.write_text("".join(lines))
    return path


def run(*args, env=None, check=False):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})}, check=check)


# --- identity and completeness ------------------------------------------------------------------

def test_a_saved_run_matches_only_its_requested_condition(tmp_path):
    run_dir = write_run(runs_dir=tmp_path / "runs", name="b0", config_path=SMOKE_B0, base_url="http://old-node:8000/v1")
    assert differences(a=run_identity(run_dir=run_dir), b=requested_identity(config_path=SMOKE_B0)) == []
    assert run("python", "scripts/run_identity.py", "check", "--run_dir", str(run_dir), "--config", str(SMOKE_B0)).returncode == 0
    saved = run_identity(run_dir=run_dir)
    for old, new, field in (('model = "google/gemma-4-31b-it"', 'model = "other/model"', "agents.seat0.model"),
                            ("temperature = 0", "temperature = 1.0", "agents.seat0.sampling.temperature"),
                            ("top_k = 64", "top_k = 32", "agents.seat0.sampling.top_k"),
                            ("max_tokens = 2048", "max_tokens = 4096", "agents.seat0.sampling.max_tokens"),
                            ("rounds = 6", "rounds = 7", "env.game_config.rounds"), ("limit = 3", "limit = 4", "select.limit"),
                            ('prediction_mode = "off"', 'prediction_mode = "opponent_action_v1"', "env.prediction_mode")):
        changed = edited(tmp_path=tmp_path, source=SMOKE_B0, old=old, new=new)
        diffs = differences(a=saved, b=requested_identity(config_path=changed))
        assert len(diffs) == 1 and diffs[0].startswith(field), (old, diffs)
    result = run("python", "scripts/run_identity.py", "check", "--run_dir", str(run_dir), "--config", str(changed))
    assert result.returncode == 1 and "is not the condition" in result.stdout and "prediction_mode" in result.stdout
    # the endpoint and concurrency are not part of the condition
    moved = edited(tmp_path=tmp_path, source=SMOKE_B0, old="max_concurrent = 3", new="max_concurrent = 1")
    assert differences(a=saved, b=requested_identity(config_path=moved)) == []


def test_a_run_without_a_manifest_matches_nothing(tmp_path):
    run_dir = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0)
    shutil.rmtree(run_dir / "manifests")
    assert any("prompt_template_hash" in d for d in differences(a=run_identity(run_dir=run_dir), b=requested_identity(config_path=SMOKE_B0)))


@pytest.mark.parametrize("damage", ["missing_seed", "errored_row", "duplicate_seed", "short_game", "unexpected_seed"])
def test_incomplete_runs_are_detected(tmp_path, damage):
    games = {"missing_seed": 2}.get(damage)
    extra = {
        "errored_row": [{"ok": False, "errors": [{"message": "ModelCallFailed"}]}],
        "duplicate_seed": [play_row(seed=0, rounds=6, prediction_mode="off")],
        "short_game": [play_row(seed=2, rounds=5, prediction_mode="off")],
        "unexpected_seed": [play_row(seed=7, rounds=6, prediction_mode="off")],
    }.get(damage, [])
    if damage == "short_game":
        games = 2
    run_dir = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0, games=games, extra_rows=extra)
    status = completeness(run_dir=run_dir)
    assert not status["complete"] and status["problems"], damage
    if damage == "missing_seed":
        assert run("python", "scripts/run_identity.py", "complete", "--run_dir", str(run_dir)).returncode == 1
    assert completeness(run_dir=write_run(runs_dir=tmp_path, name="whole", config_path=SMOKE_B0))["complete"]


# --- strict pairing ------------------------------------------------------------------------------

def analyze_pair(*, a, b, out, exploratory=False):
    args = ["python", "scripts/analyze_copycat_games.py", "--pair", str(a / "traces.jsonl"), str(b / "traces.jsonl"), "--json_out", str(out)]
    return run(*args, *(["--exploratory"] if exploratory else []))


def test_the_paired_comparison_needs_two_complete_matching_runs(tmp_path):
    b0 = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0)
    b1 = write_run(runs_dir=tmp_path, name="b1", config_path=SMOKE_B1)
    result = analyze_pair(a=b0, b=b1, out=tmp_path / "ok.json")
    assert result.returncode == 0, result.stdout
    paired = json.loads((tmp_path / "ok.json").read_text())["paired"]
    assert paired["matched_games"] == 3 and not paired["exploratory"]

    partial = write_run(runs_dir=tmp_path, name="b1_partial", config_path=SMOKE_B1, games=1)
    result = analyze_pair(a=b0, b=partial, out=tmp_path / "partial.json")
    assert result.returncode == 1 and "incomplete" in result.stdout and not (tmp_path / "partial.json").exists()

    other = write_run(runs_dir=tmp_path, name="b1_other_model", config_path=edited(
        tmp_path=tmp_path, source=SMOKE_B1, old='model = "google/gemma-4-31b-it"', new='model = "other/model"'))
    result = analyze_pair(a=b0, b=other, out=tmp_path / "other.json")
    assert result.returncode == 1 and "differ beyond env.prediction_mode" in result.stdout and "model" in result.stdout

    result = analyze_pair(a=b0, b=partial, out=tmp_path / "exploratory.json", exploratory=True)
    assert result.returncode == 0 and "EXPLORATORY" in result.stdout
    assert json.loads((tmp_path / "exploratory.json").read_text())["paired"]["exploratory"] is True


def test_pairing_checks_the_exact_files_analyzed(tmp_path):
    b0 = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0)
    b1 = write_run(runs_dir=tmp_path, name="b1", config_path=SMOKE_B1)
    for run_dir in (b0, b1):  # a one-game subset next to the complete traces.jsonl
        (run_dir / "subset.jsonl").write_text((run_dir / "traces.jsonl").read_text().splitlines(keepends=True)[0])
    args = ["python", "scripts/analyze_copycat_games.py", "--pair", str(b0 / "subset.jsonl"), str(b1 / "subset.jsonl")]
    result = run(*args, "--json_out", str(tmp_path / "subset.json"))
    assert result.returncode == 1 and "not a run's own traces.jsonl" in result.stdout and "incomplete" in result.stdout
    assert not (tmp_path / "subset.json").exists()
    complete_copy = tmp_path / "copy"
    complete_copy.mkdir()
    for run_dir in (b0, b1):  # even a complete copy elsewhere is not a run's own file
        shutil.copy(run_dir / "traces.jsonl", run_dir / "all.jsonl")
    result = run("python", "scripts/analyze_copycat_games.py", "--pair", str(b0 / "all.jsonl"), str(b1 / "all.jsonl"))
    assert result.returncode == 1 and "not a run's own traces.jsonl" in result.stdout and "incomplete" not in result.stdout


def test_pairing_needs_recorded_provenance(tmp_path):
    b0 = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0)
    b1 = write_run(runs_dir=tmp_path, name="b1", config_path=SMOKE_B1)
    for run_dir in (b0, b1):
        shutil.rmtree(run_dir / "manifests")
    result = analyze_pair(a=b0, b=b1, out=tmp_path / "none.json")
    assert result.returncode == 1 and result.stdout.count("no recorded prompt-template hash") == 2
    result = analyze_pair(a=b0, b=b1, out=tmp_path / "exploratory.json", exploratory=True)
    assert result.returncode == 0 and json.loads((tmp_path / "exploratory.json").read_text())["paired"]["exploratory"] is True


def base_config(**changes):
    config = {"model": "m", "client": {"type": "eval"}, "sampling": {"temperature": 1.0, "max_tokens": 64},
              "select": {"limit": 2}, "env": {"game": "copycat_rps", "game_config": {"rounds": 2}, "taskset": {"id": "self-play-verifiers"}}}
    for path, value in changes.items():
        node = config
        *parents, last = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = value
    return identity_from_config(config=config, template_hash="h")


def test_identity_covers_rendering_and_seat_overrides():
    train = {"type": "train", "renderer_model_name": "Qwen/Qwen3-4B"}
    assert base_config(client=train) != base_config(client={**train, "renderer_model_name": "Qwen/Qwen3-8B"})
    assert base_config(client=train) != base_config()  # train vs eval client
    assert base_config(**{"env.seat0": {"client": train}}) != base_config()  # a seat's own client
    assert base_config(**{"env.seat0": {"sampling": {"temperature": 0.0}}}) != base_config()  # a seat's own sampling
    assert base_config(**{"env.seat0": {"model": "pinned"}}) != base_config()
    assert base_config(**{"env.seat3": {"model": "pinned"}}) == base_config()  # copycat has one player: seat3 is unused
    assert base_config(client={"type": "eval", "base_url": "http://elsewhere/v1"}) == base_config()  # endpoints never count


def test_the_identity_of_a_config_is_the_fingerprint_its_run_records():
    import asyncio
    from types import SimpleNamespace
    from contextlib import asynccontextmanager

    import verifiers.v1 as vf
    from verifiers.v1.configs.agent import resolve_agent

    from self_play_verifiers.env import SelfPlayEnv, SelfPlayEnvConfig
    from self_play_verifiers.taskset import SelfPlayData, SelfPlayTask

    config = {"model": "m", "client": {"type": "train", "renderer_model_name": "Qwen/Qwen3-4B"},
              "sampling": {"temperature": 0.7, "max_tokens": 64}, "select": {"limit": 1},
              "env": {"game": "copycat_rps", "game_config": {"rounds": 1}, "taskset": {"id": "self-play-verifiers"},
                      "seat0": {"sampling": {"top_p": 0.9}}}}
    env_config = SelfPlayEnvConfig.model_validate(config["env"])
    import pydantic
    resolved = resolve_agent(env_config.seat0, model="m", client=pydantic.TypeAdapter(vf.ClientConfig).validate_python(config["client"]),
                             sampling=vf.SamplingConfig(**config["sampling"]))
    replies = ["<answer>notes</answer>", "<answer>SILENCE</answer>", "<answer>rock</answer>"]
    trace = SimpleNamespace(rewards={}, metrics={}, info={}, calls=[], record_reward=lambda *a, **k: None,
                            record_metric=lambda *a, **k: None)
    interaction = SimpleNamespace(trace=trace, turn=None)

    async def turn(messages):
        return SimpleNamespace(terminated=False, last_reply=replies.pop(0))
    interaction.turn = turn

    @asynccontextmanager
    async def open_interaction(task):
        yield interaction
    seat = SimpleNamespace(config=resolved, trainable=True, interaction=open_interaction)
    agents = SimpleNamespace(**{f"seat{i}": seat for i in range(6)})
    env = SelfPlayEnv(env_config)
    asyncio.run(env.start())
    asyncio.run(env.run(SelfPlayTask(SelfPlayData(name="g", prompt=None, info={"seed": 0})), agents))
    recorded = trace.info["game"]["fingerprint"]["agents"]
    assert recorded["seat0"]["renderer_model_name"] == "Qwen/Qwen3-4B"
    assert identity_from_config(config=config, template_hash="h")["agents"] == recorded


# --- the stage script ----------------------------------------------------------------------------

def stage(*, tmp_path, configs=None, **env):
    configs = configs or tmp_path / "configs"
    if not configs.exists():
        configs.mkdir()
        for source in (SMOKE_B0, SMOKE_B1):
            shutil.copy(source, configs / source.name)
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    return run("bash", "scripts/run_prediction_baseline.sh", "--stage", "smoke", "--base_url", env.pop("base_url", "http://node-a:8000/v1"),
               "--interrupt_after", "30", env={"EVAL_SCRIPT": "tests/stub_eval.sh", "EVAL_RUNS_DIR": str(runs),
                                               "PREDV1_CONFIG_DIR": str(configs), **env})


def analyses(tmp_path):
    return sorted(p.name for p in (tmp_path / "runs").glob("*-analysis.json"))


def test_a_complete_stage_produces_the_paired_analysis(tmp_path):
    result = stage(tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(analyses(tmp_path)) == 1 and "all conditions complete" in result.stdout


@pytest.mark.parametrize("env", [{"STUB_GAMES": "1"}, {"STUB_EXIT": "1"}])
def test_failed_or_partial_evaluations_stop_before_any_analysis(tmp_path, env):
    result = stage(tmp_path=tmp_path, **env)
    assert result.returncode == 1, result.stdout
    assert "Error:" in result.stdout and analyses(tmp_path) == []


def test_reusing_a_run_under_a_changed_condition_is_refused(tmp_path):
    assert stage(tmp_path=tmp_path).returncode == 0
    config = tmp_path / "configs" / SMOKE_B0.name
    config.write_text(config.read_text().replace("temperature = 0", "temperature = 1.0"))
    before = analyses(tmp_path)
    (tmp_path / "runs" / before[0]).unlink()
    result = stage(tmp_path=tmp_path)
    assert result.returncode == 1 and "is another condition" in result.stdout and "temperature" in result.stdout
    assert analyses(tmp_path) == []  # the old comparison is not regenerated


def test_an_unfinished_run_resumes_against_the_current_server(tmp_path):
    assert stage(tmp_path=tmp_path, STUB_GAMES="1").returncode == 1  # b0 left with 1 of 3 games
    result = stage(tmp_path=tmp_path, base_url="http://node-b:8000/v1")
    assert result.returncode == 0, result.stdout
    calls = (tmp_path / "runs" / "stub_calls.log").read_text().splitlines()
    assert "name=rps-predv1-b0-gemma31b-t0-r6-n3-smoke resume=true base_url=http://node-b:8000/v1" in calls


# --- endpoint moves ------------------------------------------------------------------------------

def test_an_endpoint_moves_only_to_a_server_of_the_runs_model(tmp_path):
    run_dir = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0, base_url="http://node-a:8000/v1")
    saved = run_dir / "configs" / "resolved" / "eval.json"
    before = json.loads(saved.read_text())
    assert migrate(run_dir=run_dir, base_url="http://node-a:8000/v1", served=lambda base_url: 1 / 0) is None  # unchanged: no check
    with pytest.raises(SystemExit, match="not the run's model"):
        migrate(run_dir=run_dir, base_url="http://node-b:8000/v1", served=lambda base_url: ["other/model"])
    with pytest.raises(SystemExit, match="cannot reach"):
        migrate(run_dir=run_dir, base_url="http://node-b:8000/v1", served=lambda base_url: 1 / 0)
    assert json.loads(saved.read_text()) == before
    assert migrate(run_dir=run_dir, base_url="http://node-b:8000/v1", served=lambda base_url: ["google/gemma-4-31b-it"]) == "http://node-a:8000/v1"
    after = json.loads(saved.read_text())
    assert after["client"]["base_url"] == "http://node-b:8000/v1"
    assert {**after, "client": before["client"]} == before  # nothing else changed
    history = json.loads((run_dir / "configs" / "endpoint_history.json").read_text())
    assert [(h["from"], h["to"]) for h in history] == [("http://node-a:8000/v1", "http://node-b:8000/v1")]
    assert differences(a=run_identity(run_dir=run_dir), b=requested_identity(config_path=SMOKE_B0)) == []


def test_eval_sh_resume_refuses_a_server_it_cannot_confirm(tmp_path):
    run_dir = write_run(runs_dir=tmp_path, name="b0", config_path=SMOKE_B0, games=1, base_url="http://node-a:8000/v1")
    saved = (run_dir / "configs" / "resolved" / "eval.json").read_text()
    result = run("bash", "scripts/eval.sh", "--name", "b0", "--resume", "true", "--base_url", "http://127.0.0.1:9/v1",
                 env={"EVAL_RUNS_DIR": str(tmp_path)})
    assert result.returncode == 1 and "cannot reach" in result.stdout + result.stderr
    assert (run_dir / "configs" / "resolved" / "eval.json").read_text() == saved
