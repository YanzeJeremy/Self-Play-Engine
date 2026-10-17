"""Select SFT examples from self-play games (expert iteration, variant A).

    python scripts/expert_iteration/select_sft_data.py --runs <run dir> [--runs ...] --out_dir D \
        [--target_min 5000] [--target_max 10000] [--window 5] [--min_window 3] [--thresholds 0.8,0.6] ...

The examples are the model's own replies, never a description of the rule. A round k (from the
second round on) qualifies when the player won it and won at least `threshold` of the rounds
k..k+window-1 (at least `min_window` of them played): a single win may be luck, a run of wins
after this decision means the reasoning behind it worked. From a qualifying round both calls
that decided it become examples: the thinking turn before the move (its notes are what the
move reads) and the move itself (its final, accepted attempt), each as the exact prompt the
model saw and its reply ending with the stop string. Talk turns are not trained.

Dropped: round 1 (chance), calls with no readable answer, moves that were not accepted, replies
cut at the token limit, examples longer than --max_tokens_per_example, exact duplicates.
Limits: at most --per_game_cap examples per game and --max_rule_share of the total per rule
(so easy rules cannot crowd out the others); within a rule, higher reward-to-go first, then
earlier rounds (where a discovery shows). The strictest threshold that yields at least
--target_min examples is used; if none does, the loosest threshold's examples are taken and the
shortfall is reported. Games are split into train/validation before examples are taken, so no
game contributes to both.

Writes D/dataset/{train,validation}.jsonl (only `prompt` and `completion` message lists, the
columns prime-rl's SFT reads; D/dataset is the SFT data folder), D/meta/{train,validation}.jsonl (each example's run, rule, seed,
round, call kind, window win fraction and tokens, line by line), and D/stats.json (yields per
rule and threshold, token counts, what was dropped).
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_identity import completeness, find_games  # noqa: E402
from self_play.chat_model import STOP_STRING  # noqa: E402

VALUE = {"W": 1.0, "D": 0.5, "L": 0.0}


def round_results(*, record):
    events = [e["content"] for e in record if e["kind"] == "event"]
    return ["W" if "the player wins" in t else "L" if "the opponent wins" in t else "D" for t in events]


def decisive_calls(*, record):
    """Per round index: (thinking call before its move or None, the accepted move call or None).
    A call belongs to round k when it is made after k rounds were played."""
    rounds: dict[int, dict] = defaultdict(dict)
    played = 0
    last_answer = None
    for index, entry in enumerate(record):
        if entry["kind"] == "event":
            played += 1
        elif entry["kind"] == "call" and entry["call"] == "think":
            rounds[played]["think"] = entry
        elif entry["kind"] == "call" and entry["call"] == "answer":
            last_answer = entry
        elif entry["kind"] == "action" and entry["action"] != "THINK" and entry["accepted"]:
            if entry["source"] == "model" and last_answer is not None:
                rounds[played]["answer"] = last_answer
            last_answer = None
    return rounds


def usable(*, call) -> str | None:
    """Why a call cannot be an example, or None."""
    if call is None:
        return "missing"
    if call.get("answer") is None or (call["call"] == "think" and call.get("invalid")):
        return "no_answer"
    if (call.get("call_info") or {}).get("finish_reason") == "length":
        return "truncated"
    return None


def example(*, call, meta):
    reply = call["reply"].rstrip()
    if STOP_STRING not in reply:
        reply = f"{reply}\n{STOP_STRING}"
    return {
        "prompt": call["messages"],
        "completion": [{"role": "assistant", "content": reply}],
        "meta": meta,
    }


def token_counter(*, model):
    """Tokens of prompt + completion as the model's chat template renders them, or a character
    estimate (3 characters per token) when the tokenizer cannot be loaded offline."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

        def count(messages):
            encoded = tokenizer.apply_chat_template(messages, tokenize=True)
            return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)  # newer transformers return a BatchEncoding
        count(messages=[{"role": "user", "content": "hi"}])
        return count, "chat_template"
    except Exception as error:  # noqa: BLE001 - any failure falls back to the estimate
        print(f"note: no offline tokenizer for {model} ({error!r}); estimating 3 characters per token")
        return (lambda messages: sum(len(m["content"]) for m in messages) // 3 + 16), "estimate"


def in_validation(*, rule, seed, run, fraction):
    digest = hashlib.sha256(f"{run}:{rule}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < fraction


@click.command()
@click.option("--runs", "run_dirs", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out_dir", required=True, type=click.Path(path_type=Path))
@click.option("--model", default="google/gemma-4-31b-it", help="For token counting (offline tokenizer).")
@click.option("--target_min", default=5000, show_default=True)
@click.option("--target_max", default=10000, show_default=True)
@click.option("--window", default=5, show_default=True, help="Rounds from the decision on that are judged.")
@click.option("--min_window", default=3, show_default=True, help="Fewest rounds left in the window for a decision to count.")
@click.option("--thresholds", default="0.8,0.6", show_default=True, help="Win fractions to try, strictest first.")
@click.option("--per_game_cap", default=16, show_default=True)
@click.option("--max_rule_share", default=0.35, show_default=True)
@click.option("--val_fraction", default=0.05, show_default=True, help="Share of games held out for validation.")
@click.option("--max_tokens_per_example", default=7936, show_default=True, help="Longer examples are dropped (SFT seq_len minus margin).")
@click.option("--hard_min", default=500, show_default=True, help="Fail when fewer train examples than this remain.")
def main(run_dirs, out_dir, model, target_min, target_max, window, min_window, thresholds, per_game_cap,
         max_rule_share, val_fraction, max_tokens_per_example, hard_min):
    thresholds = [float(t) for t in thresholds.split(",")]
    dropped = Counter()
    candidates = []  # (rule, run, seed, round, reward_to_go, window_win_fraction, [examples])
    games_per_rule = Counter()
    for run_dir in run_dirs:
        status = completeness(run_dir=run_dir)
        if not status["complete"]:
            raise SystemExit(f"Error: {run_dir} is incomplete; generation must finish first: {status['problems']}")
        for line in (run_dir / "traces.jsonl").open():
            game = find_games(row=json.loads(line))
            rule = game["game_config"].get("rule", "copy_last")
            seed = game["game_config"]["seed"]
            games_per_rule[rule] += 1
            results = round_results(record=game["record"])
            calls = decisive_calls(record=game["record"])
            for k in range(1, len(results)):
                span = results[k:k + window]
                if len(span) < min_window or results[k] != "W":
                    continue
                win_fraction = span.count("W") / len(span)
                pair = []
                for kind in ("think", "answer"):
                    call = calls[k].get(kind)
                    reason = usable(call=call)
                    if reason:
                        dropped[f"{kind}:{reason}"] += 1
                        continue
                    pair.append(example(call=call, meta={"run": run_dir.name, "rule": rule, "seed": seed, "round": k + 1,
                                                         "call": kind, "window_win_fraction": win_fraction}))
                if pair:
                    reward_to_go = sum(VALUE[r] for r in span) / len(span)
                    candidates.append((rule, run_dir.name, seed, k, reward_to_go, win_fraction, pair))

    count_tokens, token_method = token_counter(model=model)
    chosen_threshold, train, validation, per_rule = None, [], [], {}
    for threshold in thresholds:
        train, validation, seen, per_game = [], [], set(), Counter()
        qualifying = [c for c in candidates if c[5] >= threshold]
        qualifying.sort(key=lambda c: (c[0], -c[4], c[3], c[2]))
        by_rule = defaultdict(list)
        for c in qualifying:
            by_rule[c[0]].append(c)
        rule_cap = max(1, int(max_rule_share * target_max))
        per_rule = {}
        for rule, items in sorted(by_rule.items()):
            kept = 0
            for rule_, run, seed, k, _, _, pair in items:
                if kept >= rule_cap:
                    break
                for ex in pair:
                    key = hashlib.sha256(json.dumps([ex["prompt"], ex["completion"]]).encode()).hexdigest()
                    if key in seen or per_game[(run, rule, seed)] >= per_game_cap or kept >= rule_cap:
                        continue
                    tokens = count_tokens(ex["prompt"] + ex["completion"])
                    if tokens > max_tokens_per_example:
                        dropped["too_long"] += 1
                        continue
                    seen.add(key)
                    ex["meta"]["tokens"] = tokens
                    per_game[(run, rule, seed)] += 1
                    (validation if in_validation(rule=rule, seed=seed, run=run, fraction=val_fraction) else train).append(ex)
                    kept += 1
            per_rule[rule] = {"kept": kept, "qualifying_rounds": len(items), "cap": rule_cap}
        if len(train) > target_max:
            train = train[:target_max]
        chosen_threshold = threshold
        if len(train) >= target_min:
            break

    # D/dataset holds only the files and columns prime-rl's SFT reads (load_dataset on a folder
    # takes every train*/validation* file below it, so metadata lives outside it, in D/meta)
    for sub in ("dataset", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (out_dir / "dataset" / f"{name}.jsonl").open("w") as data, (out_dir / "meta" / f"{name}.jsonl").open("w") as meta:
            for row in rows:
                data.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]}) + "\n")
                meta.write(json.dumps(row["meta"]) + "\n")
    stats = {
        "threshold": chosen_threshold,
        "thresholds_tried": thresholds[: thresholds.index(chosen_threshold) + 1],
        "target_min": target_min, "target_max": target_max,
        "met_target_min": len(train) >= target_min,
        "train_examples": len(train), "validation_examples": len(validation),
        "train_tokens": sum(ex["meta"]["tokens"] for ex in train),
        "max_example_tokens": max((ex["meta"]["tokens"] for ex in train), default=0),
        "token_method": token_method,
        "examples_by_rule": dict(Counter(ex["meta"]["rule"] for ex in train)),
        "examples_by_call": dict(Counter(ex["meta"]["call"] for ex in train)),
        "per_rule": per_rule,
        "games_by_rule": dict(games_per_rule),
        "candidate_rounds": len(candidates),
        "dropped": dict(dropped),
        "settings": {"window": window, "min_window": min_window, "per_game_cap": per_game_cap,
                     "max_rule_share": max_rule_share, "val_fraction": val_fraction,
                     "max_tokens_per_example": max_tokens_per_example, "runs": [str(r) for r in run_dirs]},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("threshold", "train_examples", "validation_examples", "train_tokens",
                                            "examples_by_rule", "met_target_min")}, indent=1))
    if not stats["met_target_min"]:
        print(f"WARNING: only {len(train)} train examples (target {target_min}-{target_max}) at the loosest threshold {chosen_threshold}")
    if len(train) < hard_min:
        raise SystemExit(f"Error: {len(train)} train examples is below --hard_min {hard_min}; not enough successful play to train on")


if __name__ == "__main__":
    main()
