"""Fake eval run directories for the B0/B1 workflow tests: the files vf-eval and eval.sh leave
(configs/resolved/eval.json, manifests/, traces.jsonl), with games really played through the
Runner by a scripted policy. Also the CLI behind tests/stub_eval.sh, a stand-in for eval.sh.

    python tests/fake_runs.py --runs_dir D --name N [--config C] [--games all|K]
"""

import asyncio
import json
import sys
import tomllib
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_identity import requested_identity  # noqa: E402
from self_play.chat_model import ChatModel  # noqa: E402
from self_play.games import GAMES  # noqa: E402
from self_play.runner import Runner  # noqa: E402
from self_play_verifiers.env import SelfPlayEnvConfig  # noqa: E402

BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}
CALL_INFO = {"provider_calls": 1, "prompt_tokens": 100, "cached_input_tokens": 0, "completion_tokens": 20,
             "finish_reason": "stop", "start": 0.0, "end": 1.0}


class RotationModel(ChatModel):
    """Plays the rotation, forecasting the copycat correctly whenever a forecast is asked for."""

    def __init__(self):
        self.last_move = None
        self.last_call = dict(CALL_INFO)

    async def complete(self, *, messages):
        user = messages[-1]["content"]
        if "This is a thinking turn" in user:
            return "<answer>notes</answer>"
        if "Before you decide, you may say something" in user:
            return "<answer>SILENCE</answer>"
        move = BEATEN_BY[self.last_move] if self.last_move else "rock"
        forecast = ""
        if "<prediction>" in user:
            expected = self.last_move or "rock"
            forecast = "<prediction>" + json.dumps({m: float(m == expected) for m in BEATEN_BY}) + "</prediction>"
        self.last_move = move
        return f"<thinking>x</thinking>{forecast}<answer>{move}</answer>[STOP]"


def play_row(*, seed: int, rounds: int, prediction_mode: str, game: str = "copycat_rps", game_config: dict | None = None) -> dict:
    runner = Runner(game=GAMES[game](game_config={**(game_config or {}), "seed": seed, "rounds": rounds}),
                    models={"player": RotationModel()}, prediction_mode=prediction_mode)
    result = asyncio.run(runner.run())
    game = {"game": game, "game_config": runner.game.game_config, "seats": {"player": "seat0"}, "resumed": False,
            "outcome": result["outcome"], "steps": runner.game.steps, "step_rewards": result["step_rewards"],
            "invalid": result["invalid"], "record": runner.record}
    info = {"game": game}
    if prediction_mode != "off":
        info["predictions"] = {"schema_version": 1, "protocol": prediction_mode, "decisions": runner.prediction_decisions()}
    return {"ok": True, "traces": [{"info": info, "calls": [], "nodes": []}]}


def resolved_config(*, config_path: Path, run_dir: Path, base_url: str) -> dict:
    """Roughly what vf-eval saves: the requested values plus defaults, unset values as null."""
    toml = tomllib.loads(config_path.read_text())
    env = SelfPlayEnvConfig.model_validate(toml["env"]).model_dump(mode="json")
    env["checkpoint_dir"] = str(run_dir / "checkpoints")
    return {
        "env": env,
        "model": toml["model"],
        "client": {**toml.get("client", {}), "base_url": base_url},
        "sampling": {"temperature": None, "top_p": None, "reasoning_effort": None, "max_tokens": None, **toml.get("sampling", {})},
        "select": {"shuffle": False, "seed": 0, "skip": 0, **toml.get("select", {})},
        "num_rollouts": 1,
        "max_concurrent": toml.get("max_concurrent"),
    }


def write_run(*, runs_dir: Path, name: str, config_path: Path, games: int | None = None,
              base_url: str = "http://localhost:8000/v1", extra_rows: list[dict] = ()) -> Path:
    """Create (or extend) a run with finished games for seeds 0..games-1 (all configured games by default)."""
    run_dir = runs_dir / name
    saved = run_dir / "configs" / "resolved" / "eval.json"
    if not saved.exists():
        saved.parent.mkdir(parents=True)
        saved.write_text(json.dumps(resolved_config(config_path=config_path, run_dir=run_dir, base_url=base_url)))
        (run_dir / "manifests").mkdir()
        (run_dir / "manifests" / "0001.json").write_text(json.dumps(
            {"protocol": {"prompt_template_hash": requested_identity(config_path=config_path)["prompt_template_hash"]}}))
    config = json.loads(saved.read_text())
    limit, rounds, mode = config["select"]["limit"], config["env"]["game_config"]["rounds"], config["env"]["prediction_mode"]
    skip = config["select"].get("skip", 0)
    rows = [play_row(seed=seed, rounds=rounds, prediction_mode=mode, game=config["env"]["game"], game_config=config["env"]["game_config"])
            for seed in range(skip, skip + (limit if games is None else games))]
    (run_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [*rows, *extra_rows]))
    return run_dir


@click.command()
@click.option("--runs_dir", required=True, type=click.Path(path_type=Path))
@click.option("--name", required=True)
@click.option("--config", "config_path", default=None, type=click.Path(path_type=Path))
@click.option("--games", default="all")
@click.option("--base_url", default="none")
def main(runs_dir, name, config_path, games, base_url):
    write_run(runs_dir=runs_dir, name=name, config_path=config_path, games=None if games == "all" else int(games),
              base_url="http://localhost:8000/v1" if base_url == "none" else base_url)


if __name__ == "__main__":
    main()
