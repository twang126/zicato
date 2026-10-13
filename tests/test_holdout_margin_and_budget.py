"""The holdout confirmation's own bounds (issue #118).

``tests/test_issue_118_pins.py`` pins the end-to-end claim: the board the issue
reported becomes promotable, and stays unpromotable at the defaults. This file
pins the pieces — the margin resolver, the budget under both monotonicity
scopes, and the pre-flight note that tells an operator the bounds exist before
they meet them as a run of ``holdout_not_confirmed`` rejections.
"""

from __future__ import annotations

from typing import Any

import pytest

from zicato.core import ScoringWeights
from zicato.epoch.preflight import holdout_window_note
from zicato.tournament.gate import effective_holdout_margin, holdout_confirms


def _agg(scalar: float, per_entry: dict[str, float], mean_score: float) -> dict[str, Any]:
    """A slice aggregate in the shape the gate reads."""
    return {
        "scalar": scalar,
        "mean_score": mean_score,
        "pass_rate": mean_score,
        "per_entry": {eid: {"score": score} for eid, score in per_entry.items()},
    }


# ---------------------------------------------------------------------------
# The margin resolver
# ---------------------------------------------------------------------------


def test_the_holdout_margin_falls_back_to_promote_margin() -> None:
    """``None`` is not "zero tolerance" — it is "reuse the train knob".

    This is what keeps every contract written before the field behaving
    identically, and it is why the field is ``float | None`` rather than a
    float with a sentinel default.
    """
    assert effective_holdout_margin(ScoringWeights(promote_margin=0.07)) == pytest.approx(0.07)
    assert effective_holdout_margin(
        ScoringWeights(promote_margin=0.07, holdout_margin=0.2)
    ) == pytest.approx(0.2)
    assert effective_holdout_margin(
        ScoringWeights(promote_margin=0.07, holdout_margin=0.0)
    ) == pytest.approx(0.0), "an explicit 0.0 is a real setting, not an unset field"


def test_negative_bounds_are_rejected_at_contract_load() -> None:
    """Both fields are tolerances; a negative one would invert the check.

    A negative holdout margin would turn the confirmation from "must not
    regress" into "must improve", which is exactly the second-gate semantics
    the holdout is defined not to have. Rejected at construction, like every
    other out-of-domain knob, rather than silently at duel time.
    """
    with pytest.raises(ValueError, match="holdout_margin must be >= 0"):
        ScoringWeights(holdout_margin=-0.01)
    with pytest.raises(ValueError, match="holdout_entry_regression_budget must be >= 0"):
        ScoringWeights(holdout_entry_regression_budget=-1)


def test_the_scalar_bound_uses_the_holdout_margin_and_says_so() -> None:
    """The rejection reason must name the bound it applied, not the train one."""
    parent = _agg(0.10, {"h0": 1.0, "h1": 1.0}, 1.0)
    child = _agg(0.25, {"h0": 1.0, "h1": 1.0}, 1.0)

    tight = ScoringWeights(promote_margin=0.05, pass_rate_monotonicity=False)
    reason = holdout_confirms(parent, child, tight, train_gain=0.0)
    assert reason.startswith("holdout_not_confirmed:")
    assert "0.050000" in reason, "the reason states the bound that rejected"

    loose = ScoringWeights(promote_margin=0.05, holdout_margin=0.2, pass_rate_monotonicity=False)
    assert holdout_confirms(parent, child, loose, train_gain=0.0) == ""


# ---------------------------------------------------------------------------
# The entry-regression budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["per_entry", "aggregate"])
def test_the_budget_means_one_entry_under_either_scope(scope: str) -> None:
    """One budget unit tolerates one flip, whichever scope the contract pins.

    The two scopes measure different things — a count of regressed entries vs
    the mean-score delta — so the budget has to be converted for the aggregate
    case (``budget / entries``) or an operator's ``1`` would mean something
    different depending on a knob they set for unrelated reasons.
    """
    entries = {f"h{i}": 1.0 for i in range(6)}
    parent = _agg(0.0, entries, 1.0)
    one_flip = _agg(0.0, {**entries, "h0": 0.0}, 5 / 6)
    two_flips = _agg(0.0, {**entries, "h0": 0.0, "h1": 0.0}, 4 / 6)

    def weights(budget: int) -> ScoringWeights:
        return ScoringWeights(
            holdout_margin=1.0,  # take the scalar bound out of the picture
            holdout_entry_regression_budget=budget,
            pass_rate_monotonicity_scope=scope,  # type: ignore[arg-type]
        )

    assert (
        holdout_confirms(parent, one_flip, weights(0), train_gain=0.0) != ""
    ), "budget 0 is today's rule"
    assert holdout_confirms(parent, one_flip, weights(1), train_gain=0.0) == ""
    assert (
        holdout_confirms(parent, two_flips, weights(1), train_gain=0.0) != ""
    ), "the budget is a bound, not a bypass"
    assert holdout_confirms(parent, two_flips, weights(2), train_gain=0.0) == ""


def test_the_budget_never_reaches_a_contract_that_disabled_monotonicity() -> None:
    """``pass_rate_monotonicity=False`` still turns the whole rule off.

    The budget widens the rule; it does not resurrect it. Pinned because the
    two knobs read as adjacent and an operator who disabled the check must not
    find it back on because they set a budget.
    """
    entries = {f"h{i}": 1.0 for i in range(6)}
    weights = ScoringWeights(
        holdout_margin=1.0,
        holdout_entry_regression_budget=0,
        pass_rate_monotonicity=False,
    )
    flipped = _agg(0.0, {**entries, "h0": 0.0}, 5 / 6)
    assert holdout_confirms(_agg(0.0, entries, 1.0), flipped, weights, train_gain=0.0) == ""


# ---------------------------------------------------------------------------
# The pre-flight note
# ---------------------------------------------------------------------------


def test_the_note_is_silent_without_a_holdout() -> None:
    """No split, nothing to say — the note must not fire on unsplit boards."""
    assert holdout_window_note(ScoringWeights(), 0) is None


def test_the_note_names_the_zero_budget_trap_before_the_margin_one() -> None:
    """At budget 0 the note must say no margin can fix it.

    This is the residual the issue itself did not name: raising the holdout
    margin is the obvious response to a rejected confirmation, and at budget 0
    it accomplishes nothing, because the pass-rate rule fires before any scalar
    bound is consulted.
    """
    note = holdout_window_note(ScoringWeights(promote_margin=0.01), 6)
    assert note is not None
    assert "EVERY margin" in note
    assert "holdout_entry_regression_budget=1" in note


def test_the_note_names_the_step_size_when_the_margin_is_under_it() -> None:
    """A margin below the slice's own 1/N step cannot tolerate any regression."""
    weights = ScoringWeights(
        promote_margin=0.01,
        holdout_margin=0.02,
        holdout_entry_regression_budget=1,
    )
    note = holdout_window_note(weights, 6)
    assert note is not None
    assert "step size" in note
    assert "N_train/N_holdout" in note, "and points at the commensurable-bounds rule"


def test_the_note_goes_quiet_once_both_bounds_are_comfortable() -> None:
    """A contract that took the advice must stop being told about it."""
    weights = ScoringWeights(
        promote_margin=0.08,
        holdout_margin=0.2,
        holdout_entry_regression_budget=1,
    )
    assert holdout_window_note(weights, 6) is None


# ---------------------------------------------------------------------------
# The "improve" bar that scales with the train gain (issue #574)
# ---------------------------------------------------------------------------


def _improve(**overrides: Any) -> ScoringWeights:
    return ScoringWeights(holdout_confirmation="improve", pass_rate_monotonicity=False, **overrides)


def test_the_improve_bar_scales_with_the_train_gain() -> None:
    """A large train win needs a proportionally large holdout gain."""
    weights = _improve(holdout_min_gain_ratio=0.5)
    parent = _agg(0.70, {"h0": 0.3}, 0.3)
    small_gain = _agg(0.66, {"h0": 0.34}, 0.34)  # the holdout loss falls by 0.04
    big_gain = _agg(0.60, {"h0": 0.4}, 0.4)  # the holdout loss falls by 0.10

    # A train gain of 0.06 needs more than 0.03 on the holdout: 0.04 passes.
    assert holdout_confirms(parent, small_gain, weights, train_gain=0.06) == ""
    # A train gain of 0.18 needs more than 0.09: 0.04 fails, and the reason says so.
    reason = holdout_confirms(parent, small_gain, weights, train_gain=0.18)
    assert reason.startswith("holdout_not_confirmed:")
    assert "0.090000" in reason
    assert holdout_confirms(parent, big_gain, weights, train_gain=0.18) == ""


def test_the_ratio_never_lowers_the_fixed_bar() -> None:
    weights = _improve(holdout_min_gain=0.05, holdout_min_gain_ratio=0.5)
    parent = _agg(0.70, {"h0": 0.3}, 0.3)
    child = _agg(0.66, {"h0": 0.34}, 0.34)  # 0.04 is under the fixed 0.05
    assert holdout_confirms(parent, child, weights, train_gain=0.02).startswith(
        "holdout_not_confirmed:"
    )
    # With the default ratio, the fixed bar alone applies, whatever the train gain.
    assert holdout_confirms(parent, child, _improve(), train_gain=0.5) == ""


def test_a_negative_train_gain_does_not_lower_the_fixed_bar() -> None:
    weights = _improve(holdout_min_gain=0.05, holdout_min_gain_ratio=0.5)
    parent = _agg(0.70, {"h0": 0.3}, 0.3)
    child = _agg(0.64, {"h0": 0.36}, 0.36)  # 0.06 clears the fixed 0.05
    assert holdout_confirms(parent, child, weights, train_gain=-0.4) == ""


def test_the_reason_names_the_bar_that_fired() -> None:
    weights = _improve(holdout_min_gain=0.01, holdout_min_gain_ratio=0.5)
    flat = _agg(0.70, {"h0": 0.3}, 0.3)
    scaled = holdout_confirms(flat, flat, weights, train_gain=0.2)
    assert "holdout_min_gain_ratio 0.5 x train gain 0.200000" in scaled
    fixed = holdout_confirms(flat, flat, weights, train_gain=0.0)
    assert "(holdout_min_gain)" in fixed


def test_the_improve_fields_have_no_effect_under_no_regression() -> None:
    weights = ScoringWeights(
        holdout_min_gain=0.5, holdout_min_gain_ratio=1.0, pass_rate_monotonicity=False
    )
    flat = _agg(0.70, {"h0": 0.3}, 0.3)
    assert holdout_confirms(flat, flat, weights, train_gain=0.3) == ""


@pytest.mark.parametrize("ratio", [1.5, -0.1])
def test_the_ratio_is_bounded_to_the_unit_interval(ratio: float) -> None:
    with pytest.raises(ValueError, match="holdout_min_gain_ratio"):
        ScoringWeights(holdout_min_gain_ratio=ratio)


def test_a_ratio_of_one_is_accepted() -> None:
    assert ScoringWeights(holdout_min_gain_ratio=1.0).holdout_min_gain_ratio == 1.0


def test_train_gain_is_a_required_keyword() -> None:
    flat = _agg(0.70, {"h0": 0.3}, 0.3)
    with pytest.raises(TypeError):
        holdout_confirms(flat, flat, _improve())  # type: ignore[call-arg]


def _gate_agg(scalar: float) -> dict[str, Any]:
    return {"scalar": scalar, "pass_rate": 1.0, "per_entry": {}}


def test_evaluate_gate_scales_the_bar_with_its_own_train_gain() -> None:
    """A big train win with a small holdout gain is rejected; a proportional one is promoted."""
    from zicato.tournament.gate import evaluate_gate  # noqa: PLC0415

    weights = _improve(promote_margin=0.03, holdout_min_gain_ratio=0.5)
    # The train loss drops by 0.20, so the holdout must drop by more than 0.10.
    small = evaluate_gate(
        _gate_agg(0.80),
        _gate_agg(0.60),
        weights,
        holdout_parent_agg=_gate_agg(0.80),
        holdout_child_agg=_gate_agg(0.76),
    )
    assert small.decision == "rejected"
    assert "holdout_min_gain_ratio 0.5 x train gain 0.200000" in small.reason
    large = evaluate_gate(
        _gate_agg(0.80),
        _gate_agg(0.60),
        weights,
        holdout_parent_agg=_gate_agg(0.80),
        holdout_child_agg=_gate_agg(0.65),
    )
    assert large.decision == "promoted"


def test_the_ladder_path_scales_the_bar_with_the_train_gain(tmp_path: Any) -> None:
    """``governance._ladder_mediated_outcome`` passes its train gain too."""
    from zicato.tournament import governance  # noqa: PLC0415
    from zicato.tournament.gate import GateOutcome  # noqa: PLC0415

    weights = _improve(promote_margin=0.03, holdout_min_gain_ratio=0.5)

    def run(holdout_child: float, epoch_id: str) -> GateOutcome:
        _state, reservation = governance._reserve_ladder_query(
            tmp_path, epoch_id, weights.overfitting.ladder
        )
        assert reservation is not None
        outcome, _block = governance._ladder_mediated_outcome(
            train_outcome=GateOutcome(
                decision="promoted", reason="", delta_scalar=-0.2, delta_pass_rate=0.0
            ),
            parent_agg=_gate_agg(0.80),
            child_agg=_gate_agg(0.60),
            holdout_parent_agg=_gate_agg(0.80),
            holdout_child_agg=_gate_agg(holdout_child),
            weights=weights,
            workspace_root=tmp_path,
            epoch_id=epoch_id,
            reservation=reservation,
        )
        return outcome

    assert run(0.76, "e_small").decision != "promoted"
    assert run(0.65, "e_large").decision == "promoted"


def test_the_preflight_note_reports_the_ratio() -> None:
    weights = ScoringWeights(
        promote_margin=0.08,
        holdout_margin=0.2,
        holdout_entry_regression_budget=1,
        holdout_confirmation="improve",
        holdout_min_gain_ratio=0.5,
    )
    note = holdout_window_note(weights, 5)
    assert note is not None
    assert "holdout_min_gain_ratio=0.5" in note
