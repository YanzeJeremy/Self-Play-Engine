"""Summarize copycat rock paper scissors games from vf-eval traces.jsonl files (read-only).

    python scripts/analyze_copycat_games.py --traces <run_dir>/traces.jsonl [--traces ...] [--show 2]
        [--json_out metrics.json] [--pair <B0 traces.jsonl> <B1 traces.jsonl>]

Per run: match W/T/L; round W/D/L for all rounds and rounds 2+; mean round payoff (win 1,
draw 0.5, loss 0; not a win rate); adaptation windows (rounds 1-5, 6-10, then 10-round
windows); moves that beat the player's own previous move; invalid game actions and
fallbacks; output costs (calls, tokens, reply length, truncation, wall time); calls that
mention copying/mirroring (for qualitative reading only). Runs with prediction protocol v1
(`trace.info["predictions"]`) also get forecast coverage with every status, accuracy (argmax,
ties to the earliest label), Brier, log loss, a coverage-adjusted Brier (invalid = 2), and
action consistency (expected-payoff gap between the chosen move and the best move under the
forecast). Runs without predictions report those as unavailable.

--pair compares two runs game by game on matching seeds (paired differences, 95% bootstrap
intervals over games). --json_out writes every summary and per-game row as JSON.
"""

import json
import math
import random
import re
from collections import Counter

import click

LABELS = ("rock", "paper", "scissors")
BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}
"""Each move -> the move that beats it."""
PAYOFF = {"W": 1.0, "D": 0.5, "L": 0.0}
LOG_LOSS_FLOOR = 1e-8
WORST_BRIER = 2.0
BOOTSTRAP_RESAMPLES = 10_000

COPY_MENTION = re.compile(
    r"\b(copy|copies|copying|copied|copycat|mirror\w*|mimic\w*|same (move )?as (my|what I)"
    r"|plays? what I (just )?played|repeats? my (previous|last))\b",
    re.I,
)
"""Wording that would show the player considering the real rule. Matches need reading: most are
about phrases or dialogue, not the opponent copying the player's move."""


# --- loading ---------------------------------------------------------------------------------

def find(*, obj, test):
    """The first dict inside `obj` (depth-first) for which `test` holds, else None."""
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if test(item):
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(reversed(item))
    return None


def load_episodes(*, path):
    """One dict per finished game: its game record, prediction records (or None) and model calls."""
    episodes, unfinished = [], 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            game = find(obj=row, test=lambda d: "record" in d and "steps" in d and "outcome" in d)
            if game is None:
                unfinished += 1
                continue
            predictions = find(obj=row, test=lambda d: "decisions" in d and "schema_version" in d)
            trace = find(obj=row, test=lambda d: "calls" in d and "nodes" in d) or {}
            episodes.append({"game": game, "predictions": predictions, "calls": trace.get("calls") or []})
    return sorted(episodes, key=lambda e: e["game"]["game_config"]["seed"]), unfinished


# --- per game ----------------------------------------------------------------------------------

def round_results(*, game):
    events = [e["content"] for e in game["record"] if e["kind"] == "event"]
    return ["W" if "the player wins" in t else "L" if "the opponent wins" in t else "D" for t in events]


def moves_of(*, game):
    return [e["action"] for e in game["record"] if e["kind"] == "action" and e["accepted"] and e["action"] != "THINK"]


def windows(*, n_rounds):
    """(first, last) round numbers: 1-5, 6-10, then 10-round windows."""
    spans = [(1, min(5, n_rounds))]
    if n_rounds > 5:
        spans.append((6, min(10, n_rounds)))
    start = 11
    while start <= n_rounds:
        spans.append((start, min(start + 9, n_rounds)))
        start += 10
    return spans


def expected_payoff(*, move, probs):
    """Expected round payoff of `move` against an opponent distributed as `probs`."""
    return sum(p * (1.0 if BEATEN_BY[other] == move else 0.5 if other == move else 0.0) for other, p in probs.items())


def forecast_rows(*, predictions):
    """Per scored decision (forced actions excluded): status and, when valid, the scores."""
    rows = []
    for d in predictions["decisions"]:
        if d["action_source"] == "forced":
            continue
        row = {"status": d["prediction_status"], "action_source": d["action_source"]}
        probs, observed = d["prediction_probs"], d["observed_target"]
        if d["prediction_status"] == "valid" and observed is not None:
            labels = tuple(d["prediction_labels"])
            best = max(probs[label] for label in labels)
            row |= {
                "brier": sum((probs[label] - (label == observed)) ** 2 for label in labels),
                "log_loss": -math.log(max(probs[observed], LOG_LOSS_FLOOR)),
                "zero_prob_observed": probs[observed] == 0.0,
                "correct": next(label for label in labels if probs[label] == best) == observed,
            }
            if d["action_source"] == "model":  # a fallback move was not the player's choice
                values = {move: expected_payoff(move=move, probs=probs) for move in labels}
                row["consistency_gap"] = max(values.values()) - values[d["executed_action"]]
        rows.append(row)
    return rows


def episode_metrics(*, episode):
    game, calls = episode["game"], episode["calls"]
    results, moves = round_results(game=game), moves_of(game=game)
    n = len(results)
    later = results[1:]
    rotation = [b == BEATEN_BY[a] for a, b in zip(moves, moves[1:])]
    sources = Counter(e["source"] for e in game["record"] if e["kind"] == "action" and e["accepted"])
    usage = [c.get("usage") or {} for c in calls]
    times = [c.get("time") or {} for c in calls]
    starts = [t["start"] for t in times if t.get("start")]
    ends = [t["end"] for t in times if t.get("end")]
    row = {
        "seed": game["game_config"]["seed"],
        "rounds": n,
        "match_reward": game["outcome"]["rewards"]["player"],
        "rounds_won": results.count("W"), "rounds_drawn": results.count("D"), "rounds_lost": results.count("L"),
        "mean_round_payoff": sum(PAYOFF[r] for r in results) / n,
        "mean_round_payoff_2plus": sum(PAYOFF[r] for r in later) / len(later) if later else None,
        "round_win_frac_2plus": later.count("W") / len(later) if later else None,
        "beat_own_previous_frac": sum(rotation) / len(rotation) if rotation else None,
        "windows": {f"{a}-{b}": sum(PAYOFF[r] for r in results[a - 1:b]) / (b - a + 1) for a, b in windows(n_rounds=n)},
        "invalid": game["invalid"]["player"],
        "fallback_moves": sources.get("fallback", 0),
        "calls": sum(1 for e in game["record"] if e["kind"] == "call"),
        "reply_chars": sum(len(e["reply"] or "") for e in game["record"] if e["kind"] == "call"),
        "input_tokens": sum(u.get("prompt_tokens", 0) + (u.get("cached_input_tokens") or 0) for u in usage) if usage else None,
        "output_tokens": sum(u.get("completion_tokens", 0) for u in usage) if usage else None,
        "truncated_calls": sum(1 for c in calls if c.get("finish_reason") == "length"),
        "wall_seconds": max(ends) - min(starts) if starts and ends else None,
        "copy_mentions": sum(
            1 for e in game["record"] if e["kind"] == "call"
            and COPY_MENTION.search((e.get("thinking") or "") + " " + (e.get("answer") or ""))
        ),
    }
    if episode["predictions"] is not None:
        rows = forecast_rows(predictions=episode["predictions"])
        valid = [r for r in rows if "brier" in r]
        row["forecast"] = {
            "decisions": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "coverage": len(valid) / len(rows) if rows else None,
            "brier": mean(r["brier"] for r in valid),
            "log_loss": mean(r["log_loss"] for r in valid),
            "accuracy": mean(float(r["correct"]) for r in valid),
            "zero_prob_observed": sum(r["zero_prob_observed"] for r in valid),
            "brier_coverage_adjusted": (sum(r["brier"] for r in valid) + WORST_BRIER * (len(rows) - len(valid))) / len(rows) if rows else None,
            "consistency_gap": mean(r["consistency_gap"] for r in valid if "consistency_gap" in r),
            "inconsistent_frac": mean(float(r["consistency_gap"] > 1e-9) for r in valid if "consistency_gap" in r),
        }
    return row


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --- per run -------------------------------------------------------------------------------------

def summarize(*, rows):
    """Run-level summary from per-game rows: rounds pooled over games, forecasts pooled over decisions."""
    total = lambda key: sum(r[key] for r in rows)
    rounds = total("rounds")
    later_rounds = rounds - len(rows)
    match = Counter({1.0: "W", 0.5: "T", 0.0: "L"}[r["match_reward"]] for r in rows)
    window_keys = list(rows[0]["windows"]) if rows else []
    summary = {
        "games": len(rows),
        "match_W/T/L": f"{match['W']}/{match['T']}/{match['L']}",
        "rounds_W/D/L": f"{total('rounds_won')}/{total('rounds_drawn')}/{total('rounds_lost')}",
        "mean_round_payoff": mean(r["mean_round_payoff"] for r in rows),
        "mean_round_payoff_2plus": mean(r["mean_round_payoff_2plus"] for r in rows),
        "round_win_frac_2plus": mean(r["round_win_frac_2plus"] for r in rows),
        "beat_own_previous_frac": mean(r["beat_own_previous_frac"] for r in rows),
        "windows_mean_payoff": {k: mean(r["windows"].get(k) for r in rows) for k in window_keys},
        "invalid_actions": total("invalid"),
        "fallback_moves": total("fallback_moves"),
        "calls": total("calls"),
        "mean_reply_chars_per_call": total("reply_chars") / max(total("calls"), 1),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "truncated_calls": total("truncated_calls"),
        "mean_wall_seconds_per_game": mean(r["wall_seconds"] for r in rows),
        "copy_mentions": total("copy_mentions"),
        "later_rounds": later_rounds,
    }
    forecasts = [r["forecast"] for r in rows if "forecast" in r]
    if not forecasts:
        summary["forecast"] = "unavailable (no prediction records: prediction_mode off or an older run)"
        return summary
    decisions = sum(f["decisions"] for f in forecasts)
    statuses = Counter()
    for f in forecasts:
        statuses.update(f["statuses"])
    weighted = lambda key, weight: (
        sum(f[key] * f[weight] for f in forecasts if f[key] is not None) / max(sum(f[weight] for f in forecasts if f[key] is not None), 1e-12)
        if any(f[key] is not None for f in forecasts) else None
    )
    for f in forecasts:
        f["_valid"] = round((f["coverage"] or 0) * f["decisions"])
    summary["forecast"] = {
        "decisions": decisions,
        "statuses": dict(statuses),
        "coverage": statuses.get("valid", 0) / decisions if decisions else None,
        "brier": weighted("brier", "_valid"),
        "log_loss": weighted("log_loss", "_valid"),
        "accuracy": weighted("accuracy", "_valid"),
        "zero_prob_observed": sum(f["zero_prob_observed"] for f in forecasts),
        "brier_coverage_adjusted": weighted("brier_coverage_adjusted", "decisions"),
        "consistency_gap": mean(f["consistency_gap"] for f in forecasts),
        "inconsistent_frac": mean(f["inconsistent_frac"] for f in forecasts),
        "note": "brier/log_loss/accuracy over valid forecasts only; coverage-adjusted Brier charges invalid ones 2",
    }
    for f in forecasts:
        del f["_valid"]
    return summary


def print_summary(*, path, summary, unfinished):
    print(f"\n=== {path}\n{summary['games']} finished games, {unfinished} rows without a game record")
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    for key, value in summary.items():
        if key in ("games", "forecast", "windows_mean_payoff"):
            continue
        print(f"  {key}: {fmt(value)}")
    print("  windows (mean round payoff):", "  ".join(f"r{k}: {fmt(v)}" for k, v in summary["windows_mean_payoff"].items()))
    forecast = summary["forecast"]
    if isinstance(forecast, str):
        print(f"  forecast: {forecast}")
    else:
        print("  forecast:", ", ".join(f"{k}={fmt(v)}" for k, v in forecast.items() if k != "note"))
        print(f"    ({forecast['note']})")


# --- paired comparison ----------------------------------------------------------------------------

PAIRED_METRICS = ("mean_round_payoff", "mean_round_payoff_2plus", "round_win_frac_2plus", "match_reward", "beat_own_previous_frac")


def bootstrap_interval(*, diffs, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def paired(*, rows_a, rows_b):
    """B minus A per metric, over seeds present in both runs; games are the resampling unit."""
    by_seed_a, by_seed_b = {r["seed"]: r for r in rows_a}, {r["seed"]: r for r in rows_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    out = {"matched_games": len(seeds), "only_in_a": sorted(set(by_seed_a) - set(seeds)), "only_in_b": sorted(set(by_seed_b) - set(seeds))}
    for metric in PAIRED_METRICS:
        diffs = [by_seed_b[s][metric] - by_seed_a[s][metric] for s in seeds
                 if by_seed_a[s][metric] is not None and by_seed_b[s][metric] is not None]
        if not diffs:
            continue
        low, high = bootstrap_interval(diffs=diffs)
        out[metric] = {"mean_diff": sum(diffs) / len(diffs), "ci95": [low, high], "games": len(diffs)}
    return out


# --- CLI -------------------------------------------------------------------------------------------

@click.command()
@click.option("--traces", "paths", multiple=True, help="A vf-eval traces.jsonl; repeat for several runs.")
@click.option("--pair", "pair", nargs=2, default=None, help="Two traces.jsonl (A then B): paired B - A by seed.")
@click.option("--show", default=0, show_default=True, help="Games to print in detail per run.")
@click.option("--json_out", default=None, help="Write all summaries and per-game rows here as JSON.")
def main(paths, pair, show, json_out):
    output = {"runs": {}, "paired": None}
    for path in list(paths) + [p for p in (pair or ()) if p not in paths]:
        episodes, unfinished = load_episodes(path=path)
        rows = [episode_metrics(episode=e) for e in episodes]
        if not rows:
            print(f"\n=== {path}\nno finished games ({unfinished} rows without a game record)")
            continue
        summary = summarize(rows=rows)
        output["runs"][path] = {"summary": summary, "games": rows, "unfinished_rows": unfinished}
        print_summary(path=path, summary=summary, unfinished=unfinished)
        for episode in episodes[:show]:
            game = episode["game"]
            score = game["outcome"]["score"]
            print(f"\n  --- seed {game['game_config']['seed']}: {score['wins']}W/{score['draws']}D/{score['losses']}L")
            print("    moves:", " ".join(moves_of(game=game)))
            for d in (episode["predictions"] or {}).get("decisions", [])[:5]:
                print(f"    round {d['decision_id']}: forecast {d['prediction_probs']} ({d['prediction_status']}) "
                      f"-> played {d['executed_action']}, opponent {d['observed_target']}")
    if pair:
        a, b = pair
        if a in output["runs"] and b in output["runs"]:
            output["paired"] = {"a": a, "b": b, **paired(rows_a=output["runs"][a]["games"], rows_b=output["runs"][b]["games"])}
            print(f"\n=== paired: B - A by seed\n  A = {a}\n  B = {b}")
            for key, value in output["paired"].items():
                if isinstance(value, dict):
                    print(f"  {key}: {value['mean_diff']:+.3f}  95% CI [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]  ({value['games']} games)")
                elif key not in ("a", "b"):
                    print(f"  {key}: {value}")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(output, f, indent=1)
        print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
