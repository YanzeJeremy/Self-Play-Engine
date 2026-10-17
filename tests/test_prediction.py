"""The prediction parser and scores (self_play.prediction)."""

import math

import pytest

from self_play.prediction import (
    LOG_LOSS_FLOOR,
    STATUSES,
    argmax_label,
    brier,
    log_loss,
    parse_prediction,
)

LABELS = ("rock", "paper", "scissors")


def move_reply(*, prediction='{"rock": 0.2, "paper": 0.5, "scissors": 0.3}', answer="rock", thinking="scratch"):
    return f"<thinking>{thinking}</thinking>\n<prediction>{prediction}</prediction>\n<answer>{answer}</answer>\n[STOP]"


def status(reply):
    return parse_prediction(reply=reply, labels=LABELS).status


def test_a_valid_forecast_is_read_with_its_raw_text():
    parsed = parse_prediction(reply=move_reply(), labels=LABELS)
    assert parsed.status == "valid" and not parsed.normalized
    assert parsed.probs == {"rock": 0.2, "paper": 0.5, "scissors": 0.3}
    assert parsed.raw == '{"rock": 0.2, "paper": 0.5, "scissors": 0.3}'
    assert parse_prediction(reply=move_reply(prediction='{"rock": 1, "paper": 0, "scissors": 0}'), labels=LABELS).status == "valid"


def test_a_sum_within_tolerance_is_normalized_and_flagged_beyond_it_invalid():
    parsed = parse_prediction(reply=move_reply(prediction='{"rock": 0.333, "paper": 0.333, "scissors": 0.3335}'), labels=LABELS)
    assert parsed.status == "valid" and parsed.normalized  # sums to 0.9995
    assert parsed.raw_probs == {"rock": 0.333, "paper": 0.333, "scissors": 0.3335}
    assert math.isclose(sum(parsed.probs.values()), 1.0)
    bad = parse_prediction(reply=move_reply(prediction='{"rock": 0.5, "paper": 0.5, "scissors": 0.5}'), labels=LABELS)
    assert bad.status == "invalid_sum" and bad.probs is None and bad.raw_probs == {"rock": 0.5, "paper": 0.5, "scissors": 0.5}
    assert status(move_reply(prediction='{"rock": 0.3, "paper": 0.3, "scissors": 0.3}')) == "invalid_sum"


def test_missing_and_scratch_work_blocks():
    assert status("<thinking>x</thinking><answer>rock</answer>[STOP]") == "missing"
    quoted = "<thinking>I could write <prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}</prediction></thinking><answer>rock</answer>"
    assert status(quoted) == "missing"  # a block inside thinking is not a forecast
    after_stop = "<thinking>x</thinking><answer>rock</answer>[STOP]<prediction>{}</prediction>"
    assert status(after_stop) == "missing"


@pytest.mark.parametrize(("reply", "expected"), [
    (move_reply() + "<prediction>{}</prediction>", "valid"),  # a second block after [STOP] is cut off
    ("<prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}</prediction>" * 2 + "<answer>rock</answer>", "duplicate_block"),
    ("<prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}<answer>rock</answer>", "unclosed_block"),
    ("<answer>rock</answer><prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}</prediction>", "misplaced"),
    ("<answer><prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}</prediction>rock</answer>", "misplaced"),
    ("<prediction>{\"rock\": 1, \"paper\": 0, \"scissors\": 0}</prediction>", "misplaced"),  # no answer to precede
])
def test_block_structure(reply, expected):
    assert status(reply) == expected


@pytest.mark.parametrize(("prediction", "expected"), [
    ("rock: 0.5, paper: 0.5", "invalid_json"),
    ("[0.2, 0.5, 0.3]", "invalid_json"),
    ('{"rock": 0.2, "rock": 0.5, "scissors": 0.3}', "duplicate_keys"),
    ('{"rock": 0.5, "paper": 0.5}', "invalid_labels"),
    ('{"rock": 0.2, "paper": 0.5, "scissors": 0.3, "lizard": 0}', "invalid_labels"),
    ('{"Rock": 0.2, "paper": 0.5, "scissors": 0.3}', "invalid_labels"),
    ('{"rock": true, "paper": false, "scissors": false}', "invalid_values"),
    ('{"rock": "0.2", "paper": 0.5, "scissors": 0.3}', "invalid_values"),
    ('{"rock": -0.1, "paper": 0.6, "scissors": 0.5}', "invalid_values"),
    ('{"rock": 1.2, "paper": -0.1, "scissors": -0.1}', "invalid_values"),
    ('{"rock": NaN, "paper": 0.5, "scissors": 0.5}', "invalid_json"),  # NaN/Infinity are not JSON
    ('{"rock": Infinity, "paper": 0, "scissors": 0}', "invalid_json"),
])
def test_contents(prediction, expected):
    parsed = parse_prediction(reply=move_reply(prediction=prediction), labels=LABELS)
    assert parsed.status == expected and parsed.probs is None
    assert parsed.raw == prediction  # the original text is always kept


def test_every_status_is_declared():
    assert {"valid", "missing", "duplicate_block", "invalid_json", "invalid_labels", "invalid_values", "invalid_sum"} <= set(STATUSES)


def test_scores_on_known_distributions():
    uniform = {label: 1 / 3 for label in LABELS}
    certain = {"rock": 1.0, "paper": 0.0, "scissors": 0.0}
    assert math.isclose(brier(probs=uniform, observed="paper", labels=LABELS), 2 / 3)
    assert brier(probs=certain, observed="rock", labels=LABELS) == 0.0
    assert brier(probs=certain, observed="paper", labels=LABELS) == 2.0
    assert math.isclose(log_loss(probs=uniform, observed="rock"), math.log(3))
    assert log_loss(probs=certain, observed="paper") == -math.log(LOG_LOSS_FLOOR)
    assert argmax_label(probs={"rock": 0.4, "paper": 0.4, "scissors": 0.2}, labels=LABELS) == "rock"  # ties: label order
    assert argmax_label(probs={"rock": 0.2, "paper": 0.4, "scissors": 0.4}, labels=LABELS) == "paper"


@pytest.mark.parametrize(("prediction", "expected"), [
    ('{"rock": ' + "9" * 400 + ', "paper": 0, "scissors": 0}', "invalid_values"),   # used to raise OverflowError
    ('{"rock": -' + "9" * 400 + ', "paper": 1, "scissors": 1}', "invalid_values"),
    ('{"rock": ' + "9" * 5000 + ', "paper": 0, "scissors": 0}', "invalid_json"),     # beyond Python's int-parsing limit
    ('{"rock": 1e400, "paper": 0, "scissors": 0}', "invalid_values"),                 # parses to inf
    ("[" * 100_000 + "]" * 100_000, "invalid_json"),                                  # nesting deeper than the parser allows
])
def test_hostile_numbers_and_nesting_are_statuses_not_crashes(prediction, expected):
    assert status(move_reply(prediction=prediction)) == expected
