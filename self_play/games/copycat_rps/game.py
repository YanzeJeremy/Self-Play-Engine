"""Copycat rock paper scissors: one player against a fixed opponent over several rounds.

The opponent is part of the rules, not a seat. Its move: in round 1 a move drawn from the
seed, and from round 2 on the move the player made in the previous round. The player is
never told this; working it out is the game.

Each round is one dialogue phase, cycling through the four modes so every scheduling path
is exercised, and the opponent says one line per round:

    round 1, 5, ...  "boast"    ACTION           "I'm sure I'll beat you with this"
    round 2, 6, ...  "comment"  RESPONSE_WINDOW  "The player is doing whatever they want"
    round 3, 7, ...  "chat"     DISCUSSION       "I think I'm good at this"
    round 4, 8, ...  "claim"    SIMULTANEOUS     "I will play <move>.", true or a lie on a fair coin

The player has a thinking turn before round 1 (every game's opening) and after every round.

Prediction target "opponent_action": the opponent's move in the coming round, one of CHOICES.
The decision id is the round index (from 0), read from the observation; the observed move is
read from the round's history once it has been played (and announced).
"""

import inspect
from typing import Any

from self_play.game import THINK, DialogueMode, DialoguePhase, Game
from self_play.games.copycat_rps import prompts
from self_play.games.copycat_rps.prompts import answer_request, prediction_request, render_prompt, rules_text
from self_play.prediction import PredictionSpec

PLAYER = "player"
CHOICES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
ROUND_REWARDS = {PLAYER: 1.0, None: 0.5, "opponent": 0.0}

PHASES = [
    # (phase, mode, talk_rounds, opponent's line; None means the claim)
    ("boast", DialogueMode.ACTION, 1, "I'm sure I'll beat you with this"),
    ("comment", DialogueMode.RESPONSE_WINDOW, 0, "The player is doing whatever they want"),
    ("chat", DialogueMode.DISCUSSION, 1, "I think I'm good at this"),
    ("claim", DialogueMode.SIMULTANEOUS, 0, None),
]


def round_winner(*, player_move: str, opponent_move: str) -> str | None:
    """PLAYER, "opponent", or None for a draw."""
    if player_move == opponent_move:
        return None
    return PLAYER if BEATS[player_move] == opponent_move else "opponent"


class CopycatRPSGame(Game):
    """game_config: {"seed": int, "rounds": int >= 1}."""

    name = "copycat_rps"

    def initial_state(self) -> dict[str, Any]:
        rounds = self.game_config["rounds"]
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError(f"rounds must be a positive integer, not {rounds!r}")
        return {
            "opening_move": self.rng(label="opening", index=0).choice(CHOICES),
            # Hidden from the player until round 1 is played.
            "history": [],
            # One entry per played round: phase, opponent_says, player, opponent, winner.
        }

    # --- the fixed opponent --------------------------------------------------------

    @property
    def round_index(self) -> int:
        """The current round, from 0."""
        return len(self.state["history"])

    def opponent_move(self) -> str:
        """The opening move first, then the player's previous move."""
        history = self.state["history"]
        return history[-1]["player"] if history else self.state["opening_move"]

    def opponent_says(self) -> str:
        """This round's line. The claim names the opponent's real move on heads and one of
        the other two on tails; both draws are fixed by the seed and the round."""
        line = PHASES[self.round_index % len(PHASES)][3]
        if line is not None:
            return line
        move = self.opponent_move()
        if self.rng(label="claim_truth", index=self.round_index).random() >= 0.5:
            move = self.rng(label="claim_lie", index=self.round_index).choice([c for c in CHOICES if c != move])
        return f"I will play {move}."

    # --- what is happening now -------------------------------------------------

    @property
    def player_ids(self) -> list[str]:
        return [PLAYER]

    @property
    def done(self) -> bool:
        return self.round_index >= self.game_config["rounds"]

    def _get_dialogue_phase(self) -> DialoguePhase | None:
        if self.done:
            return None
        phase, mode, talk_rounds, _ = PHASES[self.round_index % len(PHASES)]
        return DialoguePhase(phase_id=self.round_index, phase=phase, mode=mode, eligible=(PLAYER,), talk_rounds=talk_rounds)

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        return list(CHOICES) if player_id == PLAYER else []

    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Played rounds, the score and the opponent's line for the coming round. Never the
        opponent's move for that round, the opening move, or whether a claim is true."""
        phase = self.get_dialogue_phase()
        return {
            "round": self.round_index + 1,
            "rounds": self.game_config["rounds"],
            "phase": phase.phase if phase else None,
            "opponent_says": None if self.done else self.opponent_says(),
            "history": [dict(entry) for entry in self.state["history"]],
            "score": self.score(),
        }

    def score(self) -> dict[str, int]:
        winners = [entry["winner"] for entry in self.state["history"]]
        return {"wins": winners.count(PLAYER), "losses": winners.count("opponent"), "draws": winners.count(None)}

    def get_outcome(self) -> dict[str, Any]:
        """Rewards once done: 1 for winning more rounds than the opponent, 0 for fewer, 0.5 for equal."""
        score = self.score()
        reward = 0.0
        if self.done:
            reward = 1.0 if score["wins"] > score["losses"] else 0.0 if score["wins"] < score["losses"] else 0.5
        return {"done": self.done, "rewards": {PLAYER: reward}, "score": score}

    def get_step_rewards(self) -> list[float]:
        """A move earns its round's result (1 win, 0.5 draw, 0 loss); a thinking turn earns 0."""
        if not self.done:
            raise RuntimeError("step rewards are decided only once the game is over")
        rounds = iter(self.state["history"])
        return [0.0 if step["action"] == THINK else ROUND_REWARDS[next(rounds)["winner"]] for step in self.steps]

    # --- text for players --------------------------------------------------------

    def get_rules(self, *, observation: dict[str, Any]) -> str:
        return rules_text(rounds=observation["rounds"])

    def render_observation(self, *, observation: dict[str, Any]) -> str:
        return render_prompt(rules=self.get_rules(observation=observation), observation=observation)

    def render_answer_request(self, *, legal_actions: list[Any]) -> str:
        return answer_request(legal_actions=legal_actions)

    # --- predictions ---------------------------------------------------------------

    prediction_targets = ("opponent_action",)

    def get_prediction_spec(self, *, player_id: str, observation: dict[str, Any], target_id: str) -> PredictionSpec | None:
        if target_id != "opponent_action" or player_id != PLAYER or observation["opponent_says"] is None:
            return None
        return PredictionSpec(
            target_id="opponent_action",
            description="the opponent's action for this round",
            labels=tuple(CHOICES),
            decision_id=observation["round"] - 1,
        )

    def resolve_prediction_target(self, *, player_id: str, step_id: int, target_id: str) -> str | None:
        """The opponent's move in the round that step `step_id` played: public once played."""
        if target_id != "opponent_action" or player_id != PLAYER:
            return None
        step = self.steps[step_id]
        if step["player_id"] != PLAYER or step["action"] == THINK:
            return None
        round_index = sum(1 for earlier in self.steps[:step_id] if earlier["action"] != THINK)
        history = self.state["history"]
        return history[round_index]["opponent"] if round_index < len(history) else None

    def render_prediction_request(self, *, spec: PredictionSpec) -> str:
        return prediction_request(labels=spec.labels)

    def prompt_templates(self) -> list[str]:
        return super().prompt_templates() + [inspect.getsource(prompts)]

    def parse_answer(self, *, player_id: str, answer: str) -> Any:
        """The move as one word, any case, surrounding punctuation ignored."""
        move = answer.strip().strip(".!\"'`*").strip().lower()
        if not move:
            raise ValueError("the answer is empty; write one of rock, paper or scissors")
        return move

    # --- transitions -------------------------------------------------------------

    def _apply(self, *, player_id: str, action: Any) -> list[str]:
        # The opponent's move is decided only now, from the history, after the player has chosen.
        phase = self._get_dialogue_phase()
        says, opponent = self.opponent_says(), self.opponent_move()
        winner = round_winner(player_move=action, opponent_move=opponent)
        self.state["history"].append(
            {"phase": phase.phase, "opponent_says": says, "player": action, "opponent": opponent, "winner": winner}
        )
        self.request_thinking(player_ids=[PLAYER])
        outcome = {PLAYER: "the player wins", "opponent": "the opponent wins", None: "a draw"}[winner]
        return [f"Round {self.round_index}: the player played {action}, the opponent played {opponent}; {outcome}."]
