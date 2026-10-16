"""Summarize copycat rock paper scissors games from vf-eval traces.jsonl files (read-only).

    python scripts/analyze_copycat_games.py --traces <run_dir>/traces.jsonl [--traces ...] [--window 10] [--show 2]

Reports, per run: games won/tied/lost, rounds won, the share of rounds 2+ where the move beats
the player's own previous move (the optimal policy against the copycat), rounds won per window
of rounds, calls that mention copying/mirroring, invalid answers and reply lengths. --show N
prints the moves and reasoning of N games.
"""

import json
import re
from collections import Counter

import click

BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}
"""Each move -> the move that beats it."""

COPY_MENTION = re.compile(
    r"\b(copy|copies|copying|copied|copycat|mirror\w*|mimic\w*|same (move )?as (my|what I)"
    r"|plays? what I (just )?played|repeats? my (previous|last))\b",
    re.I,
)
"""Wording that would show the player considering the real rule. Matches need reading: most are
about phrases or dialogue, not the opponent copying the player's move."""


def find_game(*, obj):
    """The SelfPlayEnv game record (trace.info["game"]) wherever it sits in one jsonl row."""
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "record" in item and "steps" in item and "outcome" in item:
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def load_games(*, path):
    games, unfinished = [], 0
    with open(path) as f:
        for line in f:
            game = find_game(obj=json.loads(line))
            if game is None:
                unfinished += 1
            else:
                games.append(game)
    return sorted(games, key=lambda g: g["game_config"]["seed"]), unfinished


def moves_of(*, game):
    return [e["action"] for e in game["record"] if e["kind"] == "action" and e["accepted"] and e["action"] != "THINK"]


def round_results(*, game):
    """Per round: "W", "D" or "L" for the player, from the game's public events."""
    events = [e["content"] for e in game["record"] if e["kind"] == "event"]
    return ["W" if "the player wins" in t else "L" if "the opponent wins" in t else "D" for t in events]


def copy_mentions(*, game):
    """(call kind, round) of every call whose reasoning or answer matches COPY_MENTION."""
    found, round_no = [], 1
    for entry in game["record"]:
        if entry["kind"] == "event":
            round_no += 1
        elif entry["kind"] == "call":
            text = (entry.get("thinking") or "") + " " + (entry.get("answer") or "")
            if COPY_MENTION.search(text):
                found.append((entry["call"], round_no))
    return found


def summarize(*, path, window, show):
    games, unfinished = load_games(path=path)
    print(f"\n=== {path}\n{len(games)} finished games, {unfinished} rows without a game record")
    if not games:
        return
    outcomes = Counter({1.0: "win", 0.5: "tie", 0.0: "loss"}[g["outcome"]["rewards"]["player"]] for g in games)
    all_rounds, later_rounds, rotation, rotation_total = Counter(), Counter(), 0, 0
    n_rounds = max(len(round_results(game=g)) for g in games)
    window_wins = [[] for _ in range(0, n_rounds, window)]
    calls, reply_chars, mentions = Counter(), {}, 0
    for game in games:
        results, moves = round_results(game=game), moves_of(game=game)
        all_rounds.update(results)
        later_rounds.update(results[1:])
        for prev, move in zip(moves, moves[1:]):
            rotation_total += 1
            rotation += move == BEATEN_BY[prev]
        for k, start in enumerate(range(0, n_rounds, window)):
            chunk = results[start:start + window]
            window_wins[k].append(chunk.count("W") / len(chunk))
        for entry in game["record"]:
            if entry["kind"] == "call":
                calls[entry["call"]] += 1
                reply_chars.setdefault(entry["call"], []).append(len(entry["reply"] or ""))
        mentions += len(copy_mentions(game=game))
    n = len(games)
    pct = lambda counter: f"{counter['W']}/{counter['D']}/{counter['L']} ({counter['W'] / sum(counter.values()):.1%} won)"
    print(f"games won/tied/lost: {outcomes['win']}/{outcomes['tie']}/{outcomes['loss']}")
    print(f"rounds W/D/L, all rounds: {pct(all_rounds)}; rounds 2+: {pct(later_rounds)}  (random 33% won, optimal 100%)")
    print(f"moves that beat the player's own previous move, rounds 2+: {rotation}/{rotation_total} = {rotation / rotation_total:.1%}")
    print(f"rounds won per {window}-round window:", " ".join(f"{sum(w) / len(w):.0%}" for w in window_wins))
    print(f"calls mentioning copying/mirroring: {mentions} of {sum(calls.values())}")
    print(f"invalid answers: {sum(g['invalid']['player'] for g in games)}")
    print("calls per game:", {k: round(v / n, 1) for k, v in calls.items()},
          "| mean reply chars:", {k: round(sum(v) / len(v)) for k, v in reply_chars.items()})
    for game in games[:show]:
        score = game["outcome"]["score"]
        print(f"\n--- seed {game['game_config']['seed']}: {score['wins']}W/{score['draws']}D/{score['losses']}L")
        print("  moves:", " ".join(moves_of(game=game)))
        print("  copy mentions (call, round):", copy_mentions(game=game))
        answers = [e for e in game["record"] if e["kind"] == "call" and e["call"] == "answer"]
        for entry in answers[-2:]:
            print(f"  late answer thinking: {(entry.get('thinking') or '')[:500]!r} -> {entry.get('answer')!r}")


@click.command()
@click.option("--traces", "paths", multiple=True, required=True, help="A vf-eval traces.jsonl; repeat for several runs.")
@click.option("--window", default=10, show_default=True, help="Rounds per window for the win-rate trend.")
@click.option("--show", default=0, show_default=True, help="Games to print in detail per run.")
def main(paths, window, show):
    for path in paths:
        summarize(path=path, window=window, show=show)


if __name__ == "__main__":
    main()
