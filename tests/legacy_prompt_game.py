"""A scripted copycat game whose prompts were recorded from the engine before the prediction
protocol existed (commit 54cf5ed) in tests/fixtures/legacy_prompts_copycat.json. With
prediction_mode "off" the engine must still build exactly these prompts, byte for byte.

8 rounds cover all four dialogue modes twice, with talk, a spoken message, an invalid
thinking turn, an illegal answer that is retried, and one that falls back."""

import asyncio

from self_play.chat_model import ScriptedChatModel
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.runner import Runner

FIXTURE = "tests/fixtures/legacy_prompts_copycat.json"


def reply(*, answer, thinking="scratch"):
    return f"<thinking>{thinking}</thinking>\n<answer>{answer}</answer>\n[STOP]"


REPLIES = [
    reply(answer="first notes"), reply(answer="I always win."), reply(answer="rock"),            # round 1: think, talk, move
    reply(answer="NO CHANGE"), reply(answer="lizard"), reply(answer="paper"),                     # round 2: think, illegal move, retry
    "no tags at all", reply(answer="SILENCE"), reply(answer="scissors"),                         # round 3: invalid think, talk, move
    reply(answer="second notes"), reply(answer="rock"),                                          # round 4 (simultaneous)
    reply(answer="NO CHANGE"), reply(answer="SILENCE"), reply(answer="spock"), reply(answer="x"),  # round 5: two bad answers -> fallback
    reply(answer="NO CHANGE"), reply(answer="paper"),                                            # round 6
    reply(answer="third notes"), reply(answer="We'll see."), reply(answer="scissors"),           # round 7
    reply(answer="NO CHANGE"), reply(answer="rock"),                                             # round 8
]


def play_legacy_game(**runner_kwargs):
    """Play the scripted game; returns (runner, model). Extra kwargs go to the Runner."""
    model = ScriptedChatModel(replies=REPLIES)
    runner = Runner(
        game=CopycatRPSGame(game_config={"seed": 5, "rounds": 8}),
        models={"player": model},
        max_retries=1,
        **runner_kwargs,
    )
    asyncio.run(runner.run())
    return runner, model
