"""Opponent-prediction protocol: what a forecast is, how it is read from a reply, and how it
is scored. Game-agnostic and verifiers-free, like the rest of the engine.

With prediction mode on, every move call (`call="answer"`) asks the player to forecast a
publicly observable target before acting (in copycat RPS: the opponent's move this round):

    <thinking>private reasoning</thinking>
    <prediction>{"rock": ..., "paper": ..., "scissors": ...}</prediction>
    <answer>the move</answer>
    [STOP]

The forecast is a measurement only. It never changes how the move is read, never triggers a
retry, forfeit or fallback, and is never shown back to the player. Games declare what can be
predicted (`Game.prediction_targets`, `Game.get_prediction_spec`) and resolve the observed
outcome after the step (`Game.resolve_prediction_target`).
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from self_play.chat_model import strip_stop

PREDICTION_MODES = ("off", "opponent_action_v1")
"""`off` keeps every prompt exactly as before the protocol existed."""

TARGET_FOR_MODE = {"opponent_action_v1": "opponent_action"}
"""The prediction target each mode asks for; a game must list it in `prediction_targets`."""

SCHEMA_VERSION = 1
"""Version of the decision records written under `trace.info["predictions"]`."""

SUM_TOLERANCE = 1e-3
"""A forecast whose probabilities sum to within this of 1 is valid (and normalized for
scoring, with the raw values kept); further off it is invalid, never repaired."""

LOG_LOSS_FLOOR = 1e-8
"""Probability floor for the log loss, so a zero-probability observation scores -log(1e-8)."""

STATUSES = (
    "valid",
    "missing",          # no top-level <prediction> block
    "unclosed_block",   # a top-level <prediction> without </prediction>
    "duplicate_block",  # more than one top-level <prediction> block
    "misplaced",        # the block is not before the top-level <answer>
    "invalid_json",     # not a JSON object (NaN and Infinity are not JSON, so they land here)
    "duplicate_keys",   # a JSON key appears twice
    "invalid_labels",   # keys are not exactly the target's labels
    "invalid_values",   # a value is not a finite number in [0, 1] (booleans are not numbers)
    "invalid_sum",      # the values do not sum to 1 within SUM_TOLERANCE
)

_THINKING = re.compile(r"<thinking>.*?</thinking>", re.S | re.I)
_BLOCK = re.compile(r"<prediction>(.*?)</prediction>", re.S | re.I)
_OPEN = re.compile(r"<prediction>", re.I)
_ANSWER_OPEN = re.compile(r"<answer>", re.I)


@dataclass(frozen=True)
class PredictionSpec:
    """What a player is asked to forecast at one decision."""

    target_id: str
    """E.g. "opponent_action"."""
    description: str
    """Public description of the target, e.g. "the opponent's action for this round"."""
    labels: tuple[str, ...]
    """Every observable outcome, in a fixed order (also the argmax tie-break order)."""
    decision_id: int
    """The game's identifier of this decision (in copycat RPS: the round index, from 0)."""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "labels": list(self.labels)}


@dataclass(frozen=True)
class ParsedPrediction:
    status: str
    raw: str | None = None
    """The text inside the block, exactly as written."""
    raw_probs: dict[str, float] | None = None
    """The parsed values before normalization (set once the values themselves are valid)."""
    probs: dict[str, float] | None = None
    """Valid forecasts only: the values normalized to sum to exactly 1."""
    normalized: bool = False
    """Whether normalization changed the values (their sum was within tolerance but not 1)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKey(keys)
    return dict(pairs)


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def parse_prediction(*, reply: str, labels: tuple[str, ...]) -> ParsedPrediction:
    """Read the forecast from a move reply. Independent of `parse_reply`: whatever this finds
    has no effect on the move. A block inside <thinking> is scratch work, not a forecast."""
    text = _THINKING.sub("", strip_stop(text=reply))
    blocks = list(_BLOCK.finditer(text))
    opens = len(_OPEN.findall(text))
    if opens == 0:
        return ParsedPrediction(status="missing")
    if opens > 1 or len(blocks) > 1:
        return ParsedPrediction(status="duplicate_block")
    if not blocks:
        return ParsedPrediction(status="unclosed_block")
    block = blocks[0]
    raw = block.group(1).strip()
    answer = _ANSWER_OPEN.search(text)
    if answer is None or block.end() > answer.start():
        return ParsedPrediction(status="misplaced", raw=raw)
    try:
        values = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except _DuplicateKey:
        return ParsedPrediction(status="duplicate_keys", raw=raw)
    except ValueError:
        return ParsedPrediction(status="invalid_json", raw=raw)
    if not isinstance(values, dict):
        return ParsedPrediction(status="invalid_json", raw=raw)
    if set(values) != set(labels) or len(values) != len(labels):
        return ParsedPrediction(status="invalid_labels", raw=raw)
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            return ParsedPrediction(status="invalid_values", raw=raw)
    raw_probs = {label: float(values[label]) for label in labels}
    total = sum(raw_probs.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        return ParsedPrediction(status="invalid_sum", raw=raw, raw_probs=raw_probs)
    probs = {label: value / total for label, value in raw_probs.items()}
    return ParsedPrediction(
        status="valid", raw=raw, raw_probs=raw_probs, probs=probs, normalized=abs(total - 1.0) > 1e-12
    )


# --- scoring (valid forecasts only; see the analysis script for coverage-aware summaries) ---

def argmax_label(*, probs: dict[str, float], labels: tuple[str, ...]) -> str:
    """The most probable label; ties go to the earliest label in `labels` order."""
    best = max(probs[label] for label in labels)
    return next(label for label in labels if probs[label] == best)


def brier(*, probs: dict[str, float], observed: str, labels: tuple[str, ...]) -> float:
    """sum over labels of (p - 1[label observed])^2: 0 is perfect, 2 the worst, uniform 3 labels 2/3."""
    return sum((probs[label] - (label == observed)) ** 2 for label in labels)


def log_loss(*, probs: dict[str, float], observed: str) -> float:
    """-log(max(p(observed), LOG_LOSS_FLOOR))."""
    return -math.log(max(probs[observed], LOG_LOSS_FLOOR))


WORST_BRIER = 2.0
"""What a coverage-adjusted Brier summary charges an invalid or missing forecast."""
