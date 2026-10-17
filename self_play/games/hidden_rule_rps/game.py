"""Rock paper scissors against a fixed opponent that follows one hidden rule.

The same game as copycat RPS (rules text, phases, the opponent's lines and its true-or-false
claims, thinking and talk turns, rewards, prompts) except for how the opponent picks its move,
which `game_config["rule"]` names. The player is never told the rule; working it out is the
game. Every rule plays the seeded opening move in round 1.

    copy_last        the player's previous move (the copycat; RPS "copycat_rps" is the same game)
    counter_last     the move that beats the player's previous move
    beaten_by_last   the move the player's previous move beats
    copy_two_back    the player's move from two rounds ago (the opening again in round 2)
    copy_three_back  the player's move from three rounds ago (the opening until round 4)
    repeat_self      its own opening move, every round
    cycle            the move that beats its own previous move (rock, paper, scissors, ...)

Diagnostic options (game_config; all default off, and then every prompt is as before):

    dialogue        "cycle" (default): the opponent's generic lines follow the 4-round phase cycle.
                    "shuffled": each round's generic line is drawn (from the seed) among the three,
                    independently of the round; the move claims stay on their rounds and stay
                    informative. Breaks superstitions tied to which phrase precedes which move.
    interventions   N >= 0 rounds (drawn from the seed, never the first or the last two) on which
                    the player's move is forced to a move drawn from the seed: the only legal
                    action, so the runner plays it without asking the model. The history shows the
                    move that was played. Scores should leave these rounds out; the rounds after
                    them measure recovery.
    switch_round    with rule_after: from this round index (0-based) on, the opponent follows
                    rule_after instead of rule, unannounced.

Always playing the move that beats your own previous move (the rotation that is optimal
against copy_last) wins every round after the first against copy_last, draws against
counter_last, and loses against beaten_by_last and copy_two_back, so a policy that memorized
the rotation is told apart from one that infers the rule.
"""

import inspect
from collections.abc import Callable
from typing import Any

from self_play.games.copycat_rps import game as copycat_game
from self_play.games.copycat_rps.game import BEATS, CHOICES, PHASES, PLAYER, CopycatRPSGame

BEATEN_BY = {loser: winner for winner, loser in BEATS.items()}
"""Each move -> the move that beats it."""

Rule = Callable[[list[str], list[str], str], str]
"""(player's moves so far, opponent's moves so far, opening move) -> the opponent's move."""

RULES: dict[str, Rule] = {
    "copy_last": lambda mine, theirs, opening: mine[-1] if mine else opening,
    "counter_last": lambda mine, theirs, opening: BEATEN_BY[mine[-1]] if mine else opening,
    "beaten_by_last": lambda mine, theirs, opening: BEATS[mine[-1]] if mine else opening,
    "copy_two_back": lambda mine, theirs, opening: mine[-2] if len(mine) >= 2 else opening,
    "copy_three_back": lambda mine, theirs, opening: mine[-3] if len(mine) >= 3 else opening,
    "repeat_self": lambda mine, theirs, opening: opening,
    "cycle": lambda mine, theirs, opening: BEATEN_BY[theirs[-1]] if theirs else opening,
}


class HiddenRuleRPSGame(CopycatRPSGame):
    """game_config: {"seed": int, "rounds": int >= 1, "rule": one of RULES}."""

    name = "hidden_rule_rps"

    def initial_state(self) -> dict[str, Any]:
        config = self.game_config
        for key in ("rule", "rule_after"):
            if key in config and config[key] not in RULES:
                raise ValueError(f"{key} must be one of {sorted(RULES)}, not {config[key]!r}")
        if "rule" not in config:
            raise ValueError(f"rule must be one of {sorted(RULES)}, not None")
        if ("switch_round" in config) != ("rule_after" in config):
            raise ValueError("switch_round and rule_after go together")
        if config.get("dialogue", "cycle") not in ("cycle", "shuffled"):
            raise ValueError(f"dialogue must be 'cycle' or 'shuffled', not {config['dialogue']!r}")
        state = super().initial_state()
        rounds, n = config["rounds"], config.get("interventions", 0)
        candidates = list(range(1, max(rounds - 2, 1)))
        if n > len(candidates):
            raise ValueError(f"{n} interventions do not fit in {rounds} rounds")
        state["forced"] = {str(r): self.rng(label="forced_move", index=r).choice(CHOICES)
                           for r in sorted(self.rng(label="interventions", index=0).sample(candidates, n))}
        # Hidden from the player: which rounds are forced (it sees only the moves that were played).
        return state

    def opponent_move(self) -> str:
        history = self.state["history"]
        mine = [entry["player"] for entry in history]
        theirs = [entry["opponent"] for entry in history]
        rule = self.game_config["rule"]
        if "switch_round" in self.game_config and self.round_index >= self.game_config["switch_round"]:
            rule = self.game_config["rule_after"]
        return RULES[rule](mine, theirs, self.state["opening_move"])

    def opponent_says(self) -> str:
        if self.game_config.get("dialogue", "cycle") == "shuffled" and PHASES[self.round_index % len(PHASES)][3] is not None:
            generic = [line for *_, line in PHASES if line is not None]
            return self.rng(label="dialogue", index=self.round_index).choice(generic)
        return super().opponent_says()

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        forced = self.state.get("forced", {}).get(str(self.round_index))
        if forced is not None and player_id == PLAYER:
            return [forced]
        return super()._get_legal_actions(player_id=player_id)

    def forced_rounds(self) -> list[int]:
        """Round indices (0-based) whose move was forced."""
        return sorted(int(r) for r in self.state.get("forced", {}))

    def prompt_templates(self) -> list[str]:
        # this module, copycat's prompts (via super) and copycat's game module, whose phases,
        # lines and claims this game inherits
        return super().prompt_templates() + [inspect.getsource(copycat_game)]
