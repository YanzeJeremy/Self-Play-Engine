"""What the player reads. Built only from an observation, so it can show nothing the
observation hides."""

from typing import Any

from self_play.prompting import prediction_format

RULES = """You are playing rock paper scissors against a computer opponent over {rounds} rounds.
Each round you and the opponent choose at the same time: rock beats scissors, scissors beats paper,
paper beats rock, and equal choices draw. Whoever wins more rounds wins the game.
The opponent may talk before you choose. What it says is not binding and may be untrue."""

OUTCOMES = {"player": "you won", "opponent": "you lost", None: "a draw"}


def rules_text(*, rounds: int) -> str:
    return RULES.format(rounds=rounds)


def render_prompt(*, rules: str, observation: dict[str, Any]) -> str:
    lines = [
        f"Round {i}: the opponent said \"{entry['opponent_says']}\"; you played {entry['player']}, "
        f"the opponent played {entry['opponent']}; {OUTCOMES[entry['winner']]}."
        for i, entry in enumerate(observation["history"], start=1)
    ]
    parts = [rules, "\n".join(lines) if lines else "No rounds have been played yet."]
    score = observation["score"]
    parts.append(f"Score: {score['wins']} won, {score['losses']} lost, {score['draws']} drawn.")
    if observation["opponent_says"] is not None:
        parts.append(
            f"Next is round {observation['round']} of {observation['rounds']}. "
            f"The opponent says: \"{observation['opponent_says']}\""
        )
    return "\n\n".join(parts)


def answer_request(*, legal_actions: list[str]) -> str:
    return f"Your answer is your move for this round: one of {', '.join(legal_actions)}."


PREDICTION_REQUEST = (
    "Before choosing your action, predict the opponent's action for this round using only the information "
    "currently available to you. Express your uncertainty as probabilities over the possible opponent actions. "
    "Put the probabilities in the prediction field, then choose your action according to the game's stated objective."
)
"""The forecast instruction (prediction mode opponent_action_v1). Names no opponent strategy."""


def prediction_request(*, labels: tuple[str, ...]) -> str:
    return f"{PREDICTION_REQUEST}\n{prediction_format(labels=labels)}"
