"""The contract every game implements.

A Game is the single source of truth: settings, state, rules, legality, transitions,
the end of the game and rewards. It never calls a model, never sees players' talk or
notes, and never decides who is asked first; the runner does those.

Division of labour:
    Game    -> WHO MAY act, WHAT is legal, what happens, and the prompt text.
    Runner  -> WHO IS ASKED FIRST, model calls, talk, thoughtpads, checkpoints, the record.

Actions are plain JSON values chosen by each game (a string, a dict, ...). The runner
never builds or inspects them: it passes a player's answer text to `parse_answer` and
the result to `step`.

Thinking: every game has THINKING phases, run by this base class. Each player's first
action is always `THINK`, and a game asks for more with `request_thinking`. In a thinking
turn the player rewrites its private thoughtpad (kept by the runner, never seen here);
the game only learns that the turn happened.

All mutable game state lives in `self.state`, a dict of JSON values. That is what makes
every game checkpointable without game-specific save code. Randomness during play is
derived from the seed (see `rng`), never kept as a live generator in the state.

Rewards come at two grains, both decided once the game is over: `get_outcome` gives each
player one reward for the game, and `get_step_rewards` gives each accepted step its own,
so a step can be credited by what it led to later.

Predictions are optional (see `self_play.prediction`): a game lists the targets it supports
in `prediction_targets`, describes each decision's target with `get_prediction_spec` (from
the observation only, never from hidden state), and names the observed outcome once a step
has revealed it with `resolve_prediction_target`. Games without them refuse prediction modes.
"""

import inspect
import json
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from self_play.prediction import PredictionSpec
from self_play.prompting import prediction_format

THINK = "THINK"
"""The action of a thinking turn, the same in every game. Games never use it themselves."""


class DialogueMode(StrEnum):
    """How the runner schedules a phase. The same five modes for every game."""

    ACTION = "action"
    """One player acts, optionally talking first."""

    RESPONSE_WINDOW = "response_window"
    """Eligible players respond in order. The game may close it early."""

    DISCUSSION = "discussion"
    """Talk rounds for everyone, then each eligible player acts."""

    SIMULTANEOUS = "simultaneous"
    """Every eligible player answers before anyone's action is applied."""

    THINKING = "thinking"
    """Each eligible player rewrites its thoughtpad, independently. Run by the base class."""


@dataclass(frozen=True)
class DialoguePhase:
    """What is happening right now in the game."""

    phase_id: int
    """Changes every time a new phase opens. The runner uses it to notice that the game
    closed a window early. Games number their phases from 0; thinking phases are negative."""

    phase: str
    """Game-specific rule state, for example "challenge_action"; "thinking" for thinking."""

    mode: DialogueMode
    eligible: tuple[str, ...]
    """Who may act now, in the game's default order."""

    talk_rounds: int = 0
    """How many times each speaker may talk before acting. 0 means straight to the action."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "phase": self.phase,
            "mode": self.mode.value,
            "eligible": list(self.eligible),
            "talk_rounds": self.talk_rounds,
        }


@dataclass
class StepResult:
    """The game's answer to one `step` call."""

    accepted: bool
    error: str | None = None
    """Why the action was rejected. A rejected action changes nothing."""

    public_events: list[str] = field(default_factory=list)
    """Human-readable descriptions of everything that became public in this step."""


class Game(ABC):
    name: ClassVar[str]
    """The game's registry name, e.g. "coup". Saved with every checkpoint."""

    def __init__(self, *, game_config: dict[str, Any]) -> None:
        self.game_config = dict(game_config)
        """Every game setting, the seed included. Games read settings only from here."""
        self.steps: list[dict[str, Any]] = []
        """Every accepted action, thinking turns included, in order: {"player_id", "action",
        "phase_id"}. A step's index here is its id; `get_step_rewards` returns one reward
        per entry. Written only by `step`; games read it."""
        self.state: dict[str, Any] = self.initial_state()
        """Everything the game's rules change during play, as JSON values only (lists, not
        tuples; string keys)."""
        self.pending_thinking: list[str] = list(self.player_ids)
        """Players who owe a thinking turn, in seat order. Everyone owes one at the start."""
        self.thinking_phases = 1
        """Thinking phases opened so far; the open one's phase_id is its negation."""

    @abstractmethod
    def initial_state(self) -> dict[str, Any]:
        """The starting state, built ONLY from `self.game_config`. The same config must
        produce the same game."""

    def rng(self, *, label: str, index: int) -> random.Random:
        """A generator for one random draw, fixed by the seed, a label and an index,
        e.g. rng(label="coin", index=round). The same draw always gives the same result,
        so nothing random has to be saved."""
        return random.Random(f"{self.game_config['seed']}:{label}:{index}")

    # --- what is happening now -------------------------------------------------

    @property
    @abstractmethod
    def player_ids(self) -> list[str]:
        """All players in seat order, including eliminated ones."""

    def get_dialogue_phase(self) -> DialoguePhase | None:
        """The open phase: a thinking phase while anyone owes a thinking turn, otherwise
        the game's own. None once the game is over (owed thinking turns lapse)."""
        phase = self._get_dialogue_phase()
        if phase is None or not self.pending_thinking:
            return phase
        return DialoguePhase(
            phase_id=-self.thinking_phases,
            phase="thinking",
            mode=DialogueMode.THINKING,
            eligible=tuple(self.pending_thinking),
        )

    @abstractmethod
    def _get_dialogue_phase(self) -> DialoguePhase | None:
        """The game's own open phase (never a thinking phase), or None once it is over."""

    def get_legal_actions(self, *, player_id: str) -> list[Any]:
        """Every action this player may take right now; empty if they may not act. This
        is the definition of legality: `step` accepts exactly these."""
        phase = self.get_dialogue_phase()
        if phase is None:
            return []
        if phase.mode is DialogueMode.THINKING:
            return [THINK] if player_id in phase.eligible else []
        return self._get_legal_actions(player_id=player_id)

    @abstractmethod
    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        """The game's own legal actions in its own phase. Never includes THINK."""

    def request_thinking(self, *, player_ids: list[str]) -> None:
        """Give these players a thinking turn before the game's next phase. Call it from
        `_apply`. Requests made once the game is over lapse."""
        if not self.pending_thinking:
            self.thinking_phases += 1
        owed = set(self.pending_thinking) | set(player_ids)
        self.pending_thinking = [pid for pid in self.player_ids if pid in owed]

    @abstractmethod
    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Everything this player may know, and nothing more. Every new observation
        needs a test that it hides what it should."""

    @abstractmethod
    def get_outcome(self) -> dict[str, Any]:
        """{"done": bool, "rewards": dict[player_id, float]}, rewards zero until done.
        Games may add public summary fields."""

    def get_step_rewards(self) -> list[float]:
        """One reward per entry in `self.steps`, for the player who took that step.
        Called only once the game is over, so a game may credit each step using
        everything that happened later. The default gives every step, wherever it
        falls, its player's final reward for the game."""
        outcome = self.get_outcome()
        if not outcome["done"]:
            raise RuntimeError("step rewards are decided only once the game is over")
        return [outcome["rewards"][step["player_id"]] for step in self.steps]

    # --- predictions (optional) ------------------------------------------------------

    prediction_targets: ClassVar[tuple[str, ...]] = ()
    """The prediction targets this game supports, e.g. ("opponent_action",). Empty: none."""

    def get_prediction_spec(self, *, player_id: str, observation: dict[str, Any], target_id: str) -> PredictionSpec | None:
        """What `player_id` is asked to forecast at its current decision, built only from its
        observation. None when the game has no such target now."""
        return None

    def resolve_prediction_target(self, *, player_id: str, step_id: int, target_id: str) -> str | None:
        """The observed label of the target for the decision that became step `step_id`,
        using only what that step made public. None when it is not publicly resolved."""
        return None

    def render_prediction_request(self, *, spec: PredictionSpec) -> str:
        """The forecast instruction added to a move call. Never describes how others behave."""
        return (
            f"Before choosing your action, predict {spec.description} using only the information currently "
            "available to you. Express your uncertainty as probabilities over its possible values. Put the "
            "probabilities in the prediction field, then choose your action according to the game's stated objective.\n"
            + prediction_format(labels=spec.labels)
        )

    def prompt_templates(self) -> list[str]:
        """Texts that define this game's prompts, for the checkpoint fingerprint. The default is
        the source of the game's module, so any change to it counts as a different condition."""
        return [inspect.getsource(sys.modules[type(self).__module__])]

    # --- text for players --------------------------------------------------------

    @abstractmethod
    def get_rules(self, *, observation: dict[str, Any]) -> str:
        """The rules as this player should read them now."""

    @abstractmethod
    def render_observation(self, *, observation: dict[str, Any]) -> str:
        """The game's part of every prompt: `get_rules(observation=...)` followed by the
        situation. No action menu: that is `render_answer_request`."""

    @abstractmethod
    def render_answer_request(self, *, legal_actions: list[Any]) -> str:
        """What to answer now and how to write it, e.g. "Your answer is your move: one of
        rock, paper, scissors." Added only to calls that ask for a game action."""

    @abstractmethod
    def parse_answer(self, *, player_id: str, answer: str) -> Any:
        """Turn a player's answer text into an action. Raise ValueError(reason) if it
        cannot be read. Reading is not checking legality: `step` does that."""

    # --- transitions -------------------------------------------------------------

    def step(self, *, player_id: str, action: Any) -> StepResult:
        """Apply one action if it is legal; otherwise change nothing and say why."""
        legal = self.get_legal_actions(player_id=player_id)
        if not legal:
            return StepResult(accepted=False, error=f"{player_id} may not act now")
        if action not in legal:
            listed = ", ".join(json.dumps(option) for option in legal)
            return StepResult(accepted=False, error=f"{json.dumps(action)} is not legal now; legal actions: {listed}")
        phase = self.get_dialogue_phase()
        if action == THINK:
            self.pending_thinking.remove(player_id)
            events = []
        else:
            events = self._apply(player_id=player_id, action=action)
        self.steps.append({"player_id": player_id, "action": action, "phase_id": phase.phase_id})
        return StepResult(accepted=True, public_events=events)

    @property
    def n_steps(self) -> int:
        """Accepted actions so far, thinking turns included."""
        return len(self.steps)

    @abstractmethod
    def _apply(self, *, player_id: str, action: Any) -> list[str]:
        """Apply a legal game action to `self.state`; return the public events it caused."""

    # --- checkpoints -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The whole game as JSON values, for checkpoints."""
        return json.loads(json.dumps({
            "game": self.name,
            "game_config": self.game_config,
            "steps": self.steps,
            "state": self.state,
            "pending_thinking": self.pending_thinking,
            "thinking_phases": self.thinking_phases,
        }))

    @classmethod
    def from_dict(cls, *, data: dict[str, Any]) -> "Game":
        """Rebuild a game saved with `to_dict`."""
        if data["game"] != cls.name:
            raise ValueError(f"the saved game is {data['game']!r}, not {cls.name!r}")
        game = cls(game_config=data["game_config"])
        game.steps = json.loads(json.dumps(data["steps"]))
        game.state = json.loads(json.dumps(data["state"]))
        game.pending_thinking = list(data["pending_thinking"])
        game.thinking_phases = data["thinking_phases"]
        return game
