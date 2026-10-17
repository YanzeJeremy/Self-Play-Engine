"""What an eval run *is*, and whether it is complete, for the B0/B1 workflow.

A run's identity is the condition that defines its results: model, client type, sampling,
which games (limit, skip, shuffle), the env settings that change play (game, game_config,
prediction mode, retries, reward settings, seats' pinned models/sampling) and the prompt-
template hash. Endpoint URLs, concurrency and checkpoint folders are not part of it.

    python scripts/run_identity.py check    --run_dir <dir> --config <requested eval toml>
    python scripts/run_identity.py complete --run_dir <dir>

`check` exits 1 when the saved run (its configs/resolved/eval.json and first manifest) is not
the requested condition, so a skipped or resumed run is never silently another condition.
`complete` exits 1 unless traces.jsonl holds exactly the configured games (seeds
skip..skip+limit-1, each once), all finished without error with the configured rounds.
"""

import json
import sys
import tomllib
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from self_play.checkpoint import prompt_template_hash  # noqa: E402
from self_play.games import GAMES  # noqa: E402

ENV_FIELDS = ("game", "game_config", "prediction_mode", "max_retries", "forfeit_on_invalid",
              "invalid_move_reward", "reward_mode")
SEATS = tuple(f"seat{i}" for i in range(6))
PROTOCOL_FIELD = ("env", "prediction_mode")
"""The one field a B0/B1 pair is meant to differ in."""


def _no_nulls(value):
    if isinstance(value, dict):
        return {k: _no_nulls(v) for k, v in sorted(value.items()) if v is not None}
    return value


def identity_from_config(*, config: dict, template_hash: str | None) -> dict:
    """The identity of an eval config: a parsed TOML as requested, or a saved resolved eval.json.
    The env is normalized through SelfPlayEnvConfig, so defaults compare equal to explicit values."""
    from self_play_verifiers.env import SelfPlayEnvConfig

    env_config = {k: v for k, v in config.get("env", {}).items() if k not in ("checkpoint_dir", "allow_legacy_checkpoints")}
    env = SelfPlayEnvConfig.model_validate(env_config)
    select = config.get("select", {})
    seats = {}
    for name in SEATS:
        seat = getattr(env, name)
        pinned = {"model": seat.model, "sampling": seat.sampling.model_dump(mode="json") if seat.sampling else None}
        seats[name] = _no_nulls(pinned)
    return {
        "model": config.get("model"),
        "client_type": config.get("client", {}).get("type", "eval"),
        "sampling": _no_nulls(config.get("sampling", {})),
        "select": {"limit": select.get("limit"), "skip": select.get("skip", 0), "shuffle": select.get("shuffle", False)},
        "env": {**{field: json.loads(json.dumps(getattr(env, field))) for field in ENV_FIELDS},
                "taskset": env.taskset.id, "seats": seats},
        "prompt_template_hash": template_hash,
    }


def requested_identity(*, config_path: Path) -> dict:
    config = tomllib.loads(config_path.read_text())
    env = config.get("env", {})
    game = GAMES[env.get("game", "copycat_rps")](game_config={**env.get("game_config", {}), "seed": 0})
    return identity_from_config(config=config, template_hash=prompt_template_hash(game=game))


def run_identity(*, run_dir: Path) -> dict:
    """The identity a run was started with: its saved resolved config and the prompt-template
    hash of its first manifest (None for a run without manifests, which then matches nothing)."""
    saved = run_dir / "configs" / "resolved" / "eval.json"
    if not saved.exists():
        raise FileNotFoundError(f"{run_dir} has no saved config ({saved})")
    manifests = sorted((run_dir / "manifests").glob("*.json"))
    template_hash = json.loads(manifests[0].read_text())["protocol"]["prompt_template_hash"] if manifests else None
    return identity_from_config(config=json.loads(saved.read_text()), template_hash=template_hash)


def differences(*, a: dict, b: dict, ignore: tuple[tuple[str, ...], ...] = (), prefix=()) -> list[str]:
    """Dotted paths where two identities differ, minus `ignore`d paths."""
    out = []
    for key in sorted(set(a) | set(b)):
        path = prefix + (key,)
        if path in ignore:
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += differences(a=va, b=vb, ignore=ignore, prefix=path)
        elif va != vb:
            out.append(f"{'.'.join(path)}: {va!r} != {vb!r}")
    return out


def find_games(*, row):
    stack = [row]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "record" in item and "steps" in item and "outcome" in item:
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def completeness(*, run_dir: Path) -> dict:
    """Whether the run holds exactly its configured games, all finished with the configured rounds."""
    identity = run_identity(run_dir=run_dir)
    limit, skip = identity["select"]["limit"], identity["select"]["skip"]
    rounds = identity["env"]["game_config"].get("rounds")
    expected = set(range(skip, skip + limit)) if limit is not None else None
    problems, seen = [], []
    traces = run_dir / "traces.jsonl"
    rows = [json.loads(line) for line in traces.open()] if traces.exists() else []
    if not traces.exists():
        problems.append("no traces.jsonl")
    for row in rows:
        game = find_games(row=row)
        if not row.get("ok", False) or game is None:
            problems.append(f"an errored or unfinished row ({(row.get('errors') or [{}])[0].get('message', 'no game record')[:120]})")
            continue
        seed = game["game_config"]["seed"]
        seen.append(seed)
        played = sum(1 for e in game["record"] if e["kind"] == "event")
        if not game["outcome"]["done"] or (rounds is not None and played != rounds):
            problems.append(f"seed {seed}: {played} of {rounds} rounds, done={game['outcome']['done']}")
    duplicates = sorted({s for s in seen if seen.count(s) > 1})
    if duplicates:
        problems.append(f"seeds played more than once: {duplicates}")
    if expected is None:
        problems.append("the run has no game limit, so its expected games are unknown")
    else:
        if missing := sorted(expected - set(seen)):
            problems.append(f"{len(missing)} of {limit} seeds have no finished game: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if extra := sorted(set(seen) - expected):
            problems.append(f"unexpected seeds: {extra[:10]}")
    return {"complete": not problems, "expected_games": limit, "finished_games": len(set(seen)), "problems": problems}


@click.group()
def main():
    pass


@main.command()
@click.option("--run_dir", required=True, type=click.Path(path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def check(run_dir, config_path):
    """Exit 1 unless the saved run is the requested condition."""
    diffs = differences(a=run_identity(run_dir=run_dir), b=requested_identity(config_path=config_path))
    if diffs:
        print(f"Error: {run_dir} is not the condition in {config_path}; use a new run name. Differences (saved != requested):")
        for line in diffs:
            print(f"  {line}")
        sys.exit(1)
    print(f"{run_dir.name}: matches {config_path}")


@main.command()
@click.option("--run_dir", required=True, type=click.Path(path_type=Path))
def complete(run_dir):
    """Exit 1 unless the run holds all its games, finished."""
    status = completeness(run_dir=run_dir)
    if not status["complete"]:
        print(f"{run_dir.name}: INCOMPLETE ({status['finished_games']} of {status['expected_games']} games)")
        for problem in status["problems"]:
            print(f"  {problem}")
        sys.exit(1)
    print(f"{run_dir.name}: complete ({status['finished_games']} of {status['expected_games']} games)")


if __name__ == "__main__":
    main()
