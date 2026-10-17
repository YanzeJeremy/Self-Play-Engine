"""Point an existing eval run at a new server address before resuming it.

    python scripts/migrate_endpoint.py --run_dir <dir> --base_url http://<new node>:8000/v1

vf-eval resumes a run only under the exact saved config (`configs/resolved/eval.json`),
endpoint included, so a server that moved to another node cannot be passed as an override.
The endpoint is not part of the run's condition (scripts/run_identity.py, the checkpoint
fingerprint), so this rewrites `client.base_url` in the saved config, and nothing else, once
the new server is confirmed to serve the run's model. Every change is appended to
`configs/endpoint_history.json`. A no-op when the address is unchanged.
"""

import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

import click


def served_models(*, base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url.removesuffix('/')}/models", timeout=10) as response:
        return [model["id"] for model in json.loads(response.read())["data"]]


def migrate(*, run_dir: Path, base_url: str, served=served_models) -> str | None:
    """Rewrite the saved endpoint; returns the old address, or None when nothing changed."""
    saved_path = run_dir / "configs" / "resolved" / "eval.json"
    config = json.loads(saved_path.read_text())
    old = config["client"]["base_url"]
    if old == base_url:
        return None
    try:
        models = served(base_url=base_url)
    except Exception as error:
        raise SystemExit(f"Error: cannot reach {base_url} to confirm it serves {config['model']}: {error!r}")
    if config["model"] not in models:
        raise SystemExit(f"Error: {base_url} serves {models}, not the run's model {config['model']}; refusing to move the run")
    history_path = run_dir / "configs" / "endpoint_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "from": old, "to": base_url,
                    "served_models": models})
    history_path.write_text(json.dumps(history, indent=1))
    config["client"]["base_url"] = base_url
    tmp = saved_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config))
    os.replace(tmp, saved_path)
    return old


@click.command()
@click.option("--run_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--base_url", required=True)
def main(run_dir, base_url):
    old = migrate(run_dir=run_dir, base_url=base_url)
    if old is None:
        print(f"{run_dir.name}: endpoint unchanged ({base_url})")
    else:
        print(f"{run_dir.name}: endpoint moved {old} -> {base_url} (recorded in configs/endpoint_history.json)")


if __name__ == "__main__":
    sys.exit(main())
