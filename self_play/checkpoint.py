"""Checkpoint files: one JSON file per game, named by a hash of everything that defines
the game, written atomically so a crash leaves the old file or the new one, never half.

    <checkpoint_dir>/<hash>.json         the snapshot (see `Runner`)
    <checkpoint_dir>/meta_<hash>.yaml    the game and game_config the hash stands for

The file name fixes the game; `run_fingerprint` fixes the run condition (prompts, model,
sampling, runner settings), which the Runner checks before reusing a snapshot.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from project_utils.hash_handling import hash_meta_dict, write_meta


def game_key(*, game: str, game_config: dict[str, Any]) -> dict[str, str]:
    """What identifies a game. game_config is sorted JSON so its key order never matters."""
    return {"game": game, "game_config": json.dumps(game_config, sort_keys=True)}


def checkpoint_path(*, checkpoint_dir: str | Path, game: str, game_config: dict[str, Any]) -> Path:
    """The game's snapshot path; writes its meta file the first time."""
    key = game_key(game=game, game_config=game_config)
    name = hash_meta_dict(key)
    if not (Path(checkpoint_dir) / f"meta_{name}.yaml").exists():
        write_meta(str(checkpoint_dir), dict(key))  # write_meta adds a timestamp to what it is given
    return Path(checkpoint_dir) / f"{name}.json"


def save_snapshot(*, path: Path, snapshot: dict[str, Any]) -> None:
    """Write via a temporary file and an atomic rename."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot))
    os.replace(tmp, path)


def load_snapshot(*, path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


FINGERPRINT_VERSION = 1


def prompt_template_hash(*, game) -> str:
    """sha256 over every engine prompt text and the game's own prompt templates."""
    from self_play.prompting import prompt_templates  # imported here: prompting must not depend on checkpoint

    texts = prompt_templates() + game.prompt_templates()
    return hashlib.sha256(json.dumps(texts).encode()).hexdigest()


def run_fingerprint(
    *, game, prediction_mode: str, agents: dict[str, Any], runner_settings: dict[str, Any]
) -> dict[str, Any]:
    """What must match for a checkpoint to be reused: the prediction protocol, the prompt
    templates, each seat's model/client/renderer/sampling (`agents`, JSON values), and the
    runner settings that change play (retries, talk window). Endpoint URLs are left out, so a
    run can resume against the same model served elsewhere."""
    return json.loads(json.dumps({
        "fingerprint_version": FINGERPRINT_VERSION,
        "prediction_protocol": prediction_mode,
        "prompt_template_hash": prompt_template_hash(game=game),
        "agents": agents,
        "runner": runner_settings,
    }, sort_keys=True))
