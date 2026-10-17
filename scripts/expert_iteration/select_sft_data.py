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

Budget and balance. Games are split into train and validation first (by a hash of run, rule
and seed), so no game contributes to both. Within each rule the eligible train examples are
ordered (higher reward-to-go first, then earlier rounds, where a discovery shows; at most
--per_game_cap per game). The budget N is then allocated across rules *before* any example is
taken: the largest N <= --target_max for which every rule can be given at most
--max_rule_share * N examples (the share of the final dataset, not of the target) and the
allocations add up to N; spare room is shared round-robin. When the rules' yields make the cap
bind (too few rules with data), N shrinks, and stats.json says so ("diversity_limited"); a cap
that cannot be met at all (no N > 0) fails. Validation takes the same proportions from the
validation games. All reported counts are computed from the written files.

Thresholds. --thresholds lists win fractions to try, strictest first; the first that reaches
--target_min is used, else the one that yields the most. The default is strict only: a
uniformly random player passes the 0.8 filter about 3.7% of the time per round, the 0.6
filter about 13.6% (stats.json reports the rate for the settings used next to the observed
rates), so relaxing admits much more luck. Every example records whether it passed the
strictest threshold.

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


def random_pass_rate(*, window: int, threshold: float) -> float:
    """Chance that a uniformly random player wins a round and >= threshold of a full window from it."""
    from math import ceil, comb
    need = max(ceil(threshold * window - 1e-9) - 1, 0)  # wins needed in the window's other rounds
    rest = window - 1
    return (1 / 3) * sum(comb(rest, j) * (1 / 3) ** j * (2 / 3) ** (rest - j) for j in range(need, rest + 1))


def allocate(*, available: dict[str, int], target: int, share: float) -> tuple[int, dict[str, int]]:
    """The largest N <= target with an allocation a_r <= available_r, a_r <= floor(share * N),
    sum a_r = N; returns (N, a). Spare room goes round-robin, largest pools first."""
    n = min(target, sum(available.values()))
    while n > 0:
        cap = int(share * n)
        alloc = {rule: min(count, cap) for rule, count in available.items()}
        total = sum(alloc.values())
        if total >= n:
            order = sorted(alloc, key=lambda r: (-alloc[r], r))
            while total > n:  # trim evenly from the largest allocations
                for rule in order:
                    if total > n and alloc[rule] > 0 and alloc[rule] == max(alloc.values()):
                        alloc[rule] -= 1
                        total -= 1
            return n, alloc
        n = total
    return 0, {rule: 0 for rule in available}


@click.command()
@click.option("--runs", "run_dirs", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out_dir", required=True, type=click.Path(path_type=Path))
@click.option("--model", default="google/gemma-4-31b-it", help="For token counting (offline tokenizer).")
@click.option("--target_min", default=5000, show_default=True)
@click.option("--target_max", default=10000, show_default=True)
@click.option("--window", default=5, show_default=True, help="Rounds from the decision on that are judged.")
@click.option("--min_window", default=3, show_default=True, help="Fewest rounds left in the window for a decision to count.")
@click.option("--thresholds", default="0.8", show_default=True, help="Win fractions to try, strictest first (e.g. 0.8,0.6).")
@click.option("--per_game_cap", default=16, show_default=True)
@click.option("--max_rule_share", default=0.35, show_default=True, help="Most any rule may make up of the final train set.")
@click.option("--val_fraction", default=0.05, show_default=True, help="Share of games held out for validation.")
@click.option("--max_tokens_per_example", default=7936, show_default=True, help="Longer examples are dropped (SFT seq_len minus margin).")
@click.option("--hard_min", default=500, show_default=True, help="Fail when fewer train examples than this remain.")
def main(run_dirs, out_dir, model, target_min, target_max, window, min_window, thresholds, per_game_cap,
         max_rule_share, val_fraction, max_tokens_per_example, hard_min):
    thresholds = [float(t) for t in thresholds.split(",")]
    strict = thresholds[0]
    dropped = Counter()
    rounds_by_rule = defaultdict(list)  # rule -> [(reward_to_go, window_win_fraction, round, run, seed, [examples])]
    games_per_rule, eligible_rounds = Counter(), Counter()
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
                if len(span) < min_window:
                    continue
                eligible_rounds[rule] += 1
                if results[k] != "W":
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
                                                         "call": kind, "window_win_fraction": win_fraction,
                                                         "strict": win_fraction >= strict}))
                if pair:
                    reward_to_go = sum(VALUE[r] for r in span) / len(span)
                    rounds_by_rule[rule].append((reward_to_go, win_fraction, k, run_dir.name, seed, pair))

    count_tokens, token_method = token_counter(model=model)
    token_cache = {}

    def pools(threshold):
        """Per rule: ordered eligible examples from train games and from validation games."""
        out = {"train": defaultdict(list), "validation": defaultdict(list)}
        for rule, rounds in rounds_by_rule.items():
            seen, per_game = set(), Counter()
            for reward_to_go, win_fraction, k, run, seed, pair in sorted(rounds, key=lambda c: (-c[0], c[2], c[4])):
                if win_fraction < threshold:
                    continue
                split = "validation" if in_validation(rule=rule, seed=seed, run=run, fraction=val_fraction) else "train"
                for ex in pair:
                    key = hashlib.sha256(json.dumps([ex["prompt"], ex["completion"]]).encode()).hexdigest()
                    if key in seen or per_game[(run, seed)] >= per_game_cap:
                        continue
                    if key not in token_cache:
                        token_cache[key] = count_tokens(ex["prompt"] + ex["completion"])
                    if token_cache[key] > max_tokens_per_example:
                        continue
                    seen.add(key)
                    per_game[(run, seed)] += 1
                    out[split][rule].append({**ex, "meta": {**ex["meta"], "tokens": token_cache[key]}})
        return out

    best = None
    for threshold in thresholds:
        pool = pools(threshold)
        available = {rule: len(pool["train"].get(rule, [])) for rule in sorted(rounds_by_rule)}
        n, alloc = allocate(available=available, target=target_max, share=max_rule_share)
        unconstrained = min(target_max, sum(available.values()))
        attempt = {"threshold": threshold, "pool": pool, "available": available, "n": n, "alloc": alloc,
                   "diversity_limited": n < unconstrained}
        if best is None or n > best["n"]:
            best = attempt
        if n >= target_min:
            best = attempt
            break

    if best["n"] == 0:
        raise SystemExit(
            f"Error: no train set satisfies --max_rule_share {max_rule_share}: eligible examples per rule {best['available']} "
            f"(at least {int(1 / max_rule_share) + 1} rules with data are needed); raise --max_rule_share to accept the imbalance")
    train, validation = [], []
    ratio = val_fraction / (1 - val_fraction) if val_fraction < 1 else 0
    for rule, count in best["alloc"].items():
        train += best["pool"]["train"].get(rule, [])[:count]
        validation += best["pool"]["validation"].get(rule, [])[:max(0, round(count * ratio))]

    # D/dataset holds only the files and columns prime-rl's SFT reads (load_dataset on a folder
    # takes every train*/validation* file below it, so metadata lives outside it, in D/meta)
    for sub in ("dataset", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (out_dir / "dataset" / f"{name}.jsonl").open("w") as data, (out_dir / "meta" / f"{name}.jsonl").open("w") as meta:
            for row in rows:
                data.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]}) + "\n")
                meta.write(json.dumps(row["meta"]) + "\n")

    def by_rule(rows):
        out = {}
        for rule in sorted({r["meta"]["rule"] for r in rows}):
            mine = [r["meta"] for r in rows if r["meta"]["rule"] == rule]
            out[rule] = {"examples": len(mine), "unique_games": len({(m["run"], m["seed"]) for m in mine}),
                         "strict_examples": sum(m["strict"] for m in mine), "share": len(mine) / len(rows)}
        return out

    qualifying = Counter(rule for rule, rounds in rounds_by_rule.items() for c in rounds if c[1] >= best["threshold"])
    stats = {
        "threshold": best["threshold"],
        "strict_threshold": strict,
        "thresholds_tried": thresholds,
        "target_min": target_min, "target_max": target_max,
        "met_target_min": len(train) >= target_min,
        "diversity_limited": best["diversity_limited"],
        "train_examples": len(train), "validation_examples": len(validation),
        "train_unique_games": len({(r["meta"]["run"], r["meta"]["seed"]) for r in train}),
        "train_tokens": sum(r["meta"]["tokens"] for r in train),
        "max_example_tokens": max((r["meta"]["tokens"] for r in train), default=0),
        "token_method": token_method,
        "train_by_rule": by_rule(train),
        "validation_by_rule": by_rule(validation),
        "examples_by_call": dict(Counter(r["meta"]["call"] for r in train)),
        "eligible_train_examples_by_rule": best["available"],
        "allocation": best["alloc"],
        "max_rule_share": max_rule_share,
        "qualification": {
            rule: {"eligible_rounds": eligible_rounds[rule], "qualifying_rounds": qualifying[rule],
                   "rate": qualifying[rule] / eligible_rounds[rule] if eligible_rounds[rule] else None}
            for rule in sorted(games_per_rule)
        },
        "random_player_pass_rate": random_pass_rate(window=window, threshold=best["threshold"]),
        "games_by_rule": dict(games_per_rule),
        "dropped": dict(dropped),
        "settings": {"window": window, "min_window": min_window, "thresholds": thresholds, "per_game_cap": per_game_cap,
                     "val_fraction": val_fraction, "max_tokens_per_example": max_tokens_per_example,
                     "runs": [str(r) for r in run_dirs]},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("threshold", "train_examples", "train_unique_games", "validation_examples",
                                            "train_tokens", "train_by_rule", "diversity_limited", "met_target_min",
                                            "qualification", "random_player_pass_rate")}, indent=1))
    if not stats["met_target_min"]:
        reason = "the per-rule share cap" if best["diversity_limited"] else "the yield of qualifying play"
        print(f"WARNING: {len(train)} train examples, below --target_min {target_min}, limited by {reason}")
    if len(train) < hard_min:
        raise SystemExit(f"Error: {len(train)} train examples is below --hard_min {hard_min}; not enough successful play to train on")


if __name__ == "__main__":
    main()
