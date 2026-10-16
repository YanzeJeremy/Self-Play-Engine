"""Reference scores for copycat rock paper scissors: fixed policies played through the real Runner
and CopycatRPSGame (no LLM, no GPU). The policies read the same prompts a model gets.

    python scripts/scripted_copycat_baseline.py [--seeds 100] [--rounds 10]
"""

import asyncio
import random
import re
from collections import defaultdict

import click

from self_play.chat_model import ChatModel
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.runner import Runner

BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}
"""Each move -> the move that beats it."""
MOVES = list(BEATEN_BY)
HISTORY = re.compile(r"you played (\w+), the opponent played (\w+)")
ROUND_VALUE = {"player": 1.0, None: 0.5, "opponent": 0.0}


def reply(*, answer):
    return f"<thinking>.</thinking><answer>{answer}</answer>[STOP]"


class PolicyModel(ChatModel):
    """Think turns write fixed notes, talk turns say SILENCE, and moves come from
    `policy(history=..., rng=...)`, where history is [(my_move, opponent_move), ...] read from the prompt."""

    def __init__(self, *, policy, rng):
        self._policy, self._rng = policy, rng

    async def complete(self, *, messages):
        user = messages[-1]["content"]
        if "This is a thinking turn" in user:
            return reply(answer="notes")
        if "Before you decide, you may say something" in user:
            return reply(answer="SILENCE")
        return reply(answer=self._policy(history=HISTORY.findall(user), rng=self._rng))


def optimal(*, history, rng):
    """Beat your own last move: the opponent plays it."""
    return BEATEN_BY[history[-1][0]] if history else rng.choice(MOVES)


def always_rock(*, history, rng):
    """What the runner's fallback plays for an invalid answer."""
    return "rock"


def uniform_random(*, history, rng):
    return rng.choice(MOVES)


def beat_opponents_last(*, history, rng):
    """A common heuristic, which is wrong against this opponent."""
    return BEATEN_BY[history[-1][1]] if history else rng.choice(MOVES)


POLICIES = {
    "optimal (beat own last)": optimal,
    "always rock": always_rock,
    "uniform random": uniform_random,
    "beat opponent's last": beat_opponents_last,
}


async def play(*, policy, seed, rounds):
    runner = Runner(
        game=CopycatRPSGame(game_config={"seed": seed, "rounds": rounds}),
        models={"player": PolicyModel(policy=policy, rng=random.Random(seed))},
    )
    result = await runner.run()
    calls = sum(1 for entry in runner.record if entry["kind"] == "call")
    return runner.game.state["history"], result, calls


@click.command()
@click.option("--seeds", default=100, show_default=True, help="Games per policy (seeds 0..N-1).")
@click.option("--rounds", default=10, show_default=True, help="Rounds per game.")
def main(seeds, rounds):
    print(f"{seeds} seeds x {rounds} rounds. Round result: 1 win, 0.5 draw, 0 loss.\n")
    mean = lambda xs: sum(xs) / len(xs)
    for name, policy in POLICIES.items():
        per_round, game_rewards, calls_per_game = defaultdict(list), [], []
        for seed in range(seeds):
            history, result, calls = asyncio.run(play(policy=policy, seed=seed, rounds=rounds))
            for i, entry in enumerate(history):
                per_round[i].append(ROUND_VALUE[entry["winner"]])
            game_rewards.append(result["outcome"]["rewards"]["player"])
            calls_per_game.append(calls)
        later = [v for i in range(1, rounds) for v in per_round[i]]
        print(f"{name:24s} game reward {mean(game_rewards):.2f} | mean round result, rounds 2+ {mean(later):.2f} "
              f"| calls/game {mean(calls_per_game):.0f}")
        print(f"{'':24s} per round: {' '.join(f'{mean(per_round[i]):.2f}' for i in range(rounds))}")


if __name__ == "__main__":
    main()
