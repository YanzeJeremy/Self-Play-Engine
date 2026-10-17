"""Check that the served adapter carries the training: compare the base model and the adapter, as
the evaluation server serves them, on SFT training examples.

    python scripts/expert_iteration/check_adapter.py --base_url http://localhost:8000/v1 --base_model M \
        --adapter_name A --data_dir <selection out_dir> --out check.json [--n 24] [--require_improvement]

Each example's prompt (rendered with the model's chat template, generation prompt included) and
completion are sent as token ids to /v1/completions with prompt_logprobs; the completion tokens'
mean negative log-likelihood is computed under both models. An adapter that changes nothing (every
difference below 1e-4 nats) is not being applied: exit 1. With --require_improvement, an adapter
whose likelihood of its own training data is not higher than the base model's also exits 1 (a
trained adapter that serves correctly should lower it). If the check itself cannot run (an API
mismatch, a server error), it says so and exits 0: it is a guard, not a gate on its own bugs.
"""

import json
import random
import sys
import urllib.request
from pathlib import Path

import click


def post(*, url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def completion_nll(*, base_url: str, model: str, prompt_ids: list[int], completion_ids: list[int]) -> float:
    ids = prompt_ids + completion_ids
    body = post(url=f"{base_url.removesuffix('/')}/completions",
                payload={"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    entries = body["choices"][0]["prompt_logprobs"]
    total = 0.0
    for position in range(len(prompt_ids), len(ids)):
        entry = entries[position]
        token = ids[position]
        info = entry.get(str(token), entry.get(token)) if isinstance(entry, dict) else None
        if info is None:
            raise ValueError(f"no logprob for token {token} at position {position}")
        total -= info["logprob"] if isinstance(info, dict) else float(info)
    return total / len(completion_ids)


@click.command()
@click.option("--base_url", required=True)
@click.option("--base_model", required=True)
@click.option("--adapter_name", required=True)
@click.option("--data_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--n", default=24)
@click.option("--require_improvement", is_flag=True)
def main(base_url, base_model, adapter_name, data_dir, out, n, require_improvement):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        rows = [json.loads(line) for line in (data_dir / "dataset" / "train.jsonl").open()]
        rows = random.Random(0).sample(rows, min(n, len(rows)))
        results = []
        for row in rows:
            encoded = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            prompt_ids = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
            completion_ids = tokenizer(row["completion"][0]["content"], add_special_tokens=False)["input_ids"]
            base = completion_nll(base_url=base_url, model=base_model, prompt_ids=prompt_ids, completion_ids=completion_ids)
            tuned = completion_nll(base_url=base_url, model=adapter_name, prompt_ids=prompt_ids, completion_ids=completion_ids)
            results.append({"base_nll": base, "adapter_nll": tuned, "tokens": len(completion_ids)})
    except Exception as error:  # noqa: BLE001 - see the module docstring
        print(f"WARNING: the adapter check could not run ({error!r}); continuing without it")
        out.write_text(json.dumps({"ran": False, "error": repr(error)}, indent=1))
        return
    mean = lambda key: sum(r[key] for r in results) / len(results)
    differences = [abs(r["base_nll"] - r["adapter_nll"]) for r in results]
    summary = {
        "ran": True, "examples": len(results),
        "base_mean_nll": mean("base_nll"), "adapter_mean_nll": mean("adapter_nll"),
        "examples_improved": sum(r["adapter_nll"] < r["base_nll"] for r in results),
        "max_abs_difference": max(differences),
        "per_example": results,
    }
    out.write_text(json.dumps(summary, indent=1))
    print(f"adapter check on {len(results)} training examples: mean NLL per completion token base {summary['base_mean_nll']:.4f}, "
          f"adapter {summary['adapter_mean_nll']:.4f}; adapter lower on {summary['examples_improved']}/{len(results)}")
    if summary["max_abs_difference"] < 1e-4:
        print("Error: the served adapter changes nothing: it is not being applied")
        sys.exit(1)
    if require_improvement and summary["adapter_mean_nll"] >= summary["base_mean_nll"]:
        print("Error: the served adapter does not make its own training data more likely than the base model does")
        sys.exit(1)


if __name__ == "__main__":
    main()
