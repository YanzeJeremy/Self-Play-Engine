"""Rock paper scissors against a fixed opponent that follows one hidden rule.

The same game as copycat RPS (rules text, phases, the opponent's lines and its true-or-false
claims, thinking and talk turns, rewards, prompts) except for how the opponent picks its move,
which `game_config["rule"]` names. The player is never told the rule; working it out is the
game. Every rule plays the seeded opening move in round 1.

    copy_last        the player's previous move (the copycat; RPS "copycat_rps" is the same game)
    counter_last     the move that beats the player's previous move
    beaten_by_last   the move the player's previous move beats
    copy_two_back    the player's move from two rounds ago (the opening again in round 2)
    repeat_self      its own opening move, every round
    cycle            the move that beats its own previous move (rock, paper, scissors, ...)

Always playing the move that beats your own previous move (the rotation that is optimal
against copy_last) wins every round after the first against copy_last, draws against
counter_last, and loses against beaten_by_last and copy_two_back, so a policy that memorized
the rotation is told apart from one that infers the rule.
"""

import inspect
from collections.abc import Callable
from typing import Any

from self_play.games.copycat_rps import game as copycat_game
from self_play.games.copycat_rps.game import BEATS, CopycatRPSGame

BEATEN_BY = {loser: winner for winner, loser in BEATS.items()}
"""Each move -> the move that beats it."""

Rule = Callable[[list[str], list[str], str], str]
"""(player's moves so far, opponent's moves so far, opening move) -> the opponent's move."""

RULES: dict[str, Rule] = {
    "copy_last": lambda mine, theirs, opening: mine[-1] if mine else opening,
    "counter_last": lambda mine, theirs, opening: BEATEN_BY[mine[-1]] if mine else opening,
    "beaten_by_last": lambda mine, theirs, opening: BEATS[mine[-1]] if mine else opening,
    "copy_two_back": lambda mine, theirs, opening: mine[-2] if len(mine) >= 2 else opening,
    "repeat_self": lambda mine, theirs, opening: opening,
    "cycle": lambda mine, theirs, opening: BEATEN_BY[theirs[-1]] if theirs else opening,
}


class HiddenRuleRPSGame(CopycatRPSGame):
    """game_config: {"seed": int, "rounds": int >= 1, "rule": one of RULES}."""

    name = "hidden_rule_rps"

    def initial_state(self) -> dict[str, Any]:
        rule = self.game_config.get("rule")
        if rule not in RULES:
            raise ValueError(f"rule must be one of {sorted(RULES)}, not {rule!r}")
        return super().initial_state()

    def opponent_move(self) -> str:
        history = self.state["history"]
        mine = [entry["player"] for entry in history]
        theirs = [entry["opponent"] for entry in history]
        return RULES[self.game_config["rule"]](mine, theirs, self.state["opening_move"])

    def prompt_templates(self) -> list[str]:
        # this module, copycat's prompts (via super) and copycat's game module, whose phases,
        # lines and claims this game inherits
        return super().prompt_templates() + [inspect.getsource(copycat_game)]
