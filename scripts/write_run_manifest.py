"""Write a run manifest next to an eval run: everything needed to tell two runs' conditions
apart, beyond what vf-eval saves in <run_dir>/configs/. Called by scripts/eval.sh before each
vf-eval invocation (a resumed run gets a new manifest, so a changed environment is visible).

    python scripts/write_run_manifest.py --run_dir <dir> --config <eval toml or saved eval.json> [--base_url URL]

Records: the root/prime-rl/verifiers commits and whether the trees are dirty; the eval config
(raw and parsed); the prediction protocol constants and the game's prompt-template hash; the
served model (vLLM /v1/models and /version) and, from the local Hugging Face cache, the model
revision, its generation_config.json (the server's default sampling) and a chat-template hash.
Anything unavailable is recorded as null with the reason, never guessed.
"""

import datetime
import hashlib
import json
import subprocess
import tomllib
import urllib.request
from pathlib import Path

import click

from self_play.checkpoint import prompt_template_hash
from self_play.games import GAMES
from self_play.prediction import LOG_LOSS_FLOOR, SCHEMA_VERSION, SUM_TOLERANCE

ROOT = Path(__file__).resolve().parent.parent


def git_state(*, path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return {"error": str(error)}


def load_config(*, path: Path) -> dict:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    return json.loads(path.read_text())


def fetch_json(*, url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {"url": url, "response": json.loads(response.read())}
    except Exception as error:  # the manifest must never stop a run
        return {"url": url, "error": repr(error)}


def hf_cache_info(*, model: str) -> dict:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        return {"error": repr(error)}
    info: dict = {}
    for name in ("generation_config.json", "chat_template.jinja", "tokenizer_config.json"):
        path = try_to_load_from_cache(model, name)
        if not isinstance(path, str):
            info[name] = None
            continue
        text = Path(path).read_text()
        info["revision"] = Path(path).parent.name
        if name == "generation_config.json":
            info[name] = json.loads(text)
        else:
            info[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest()}
    return info


@click.command()
@click.option("--run_dir", required=True, type=click.Path(path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--base_url", default="none", help="Overrides the config's [client] base_url, as in eval.sh.")
def main(run_dir, config_path, base_url):
    config = load_config(path=config_path)
    env = config.get("env", {})
    game_name = env.get("game", "copycat_rps")
    game = GAMES[game_name](game_config={**env.get("game_config", {}), "seed": 0})
    if base_url == "none":
        base_url = config.get("client", {}).get("base_url")
    server_root = base_url.removesuffix("/").removesuffix("/v1") if base_url else None
    model = config.get("model")
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": {"root": git_state(path=ROOT), "prime-rl": git_state(path=ROOT / "prime-rl"),
                "verifiers": git_state(path=ROOT / "prime-rl" / "deps" / "verifiers")},
        "eval_config": {"path": str(config_path), "raw": config_path.read_text(), "parsed": config},
        "protocol": {
            "prediction_mode": env.get("prediction_mode", "off"),
            "decision_schema_version": SCHEMA_VERSION,
            "sum_tolerance": SUM_TOLERANCE,
            "log_loss_floor": LOG_LOSS_FLOOR,
            "game": game_name,
            "prompt_template_hash": prompt_template_hash(game=game),
            "max_retries": env.get("max_retries", 0),
            "reward_mode": env.get("reward_mode", "game"),
        },
        "sampling_requested": config.get("sampling", {}),
        "server": {
            "base_url": base_url,
            "models": fetch_json(url=f"{base_url.removesuffix('/')}/models") if base_url else None,
            "version": fetch_json(url=f"{server_root}/version") if server_root else None,
        },
        "model_files": hf_cache_info(model=model) if model else None,
    }
    out = run_dir / "manifests" / f"{manifest['created_utc'].replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"manifest written: {out}")


if __name__ == "__main__":
    main()
