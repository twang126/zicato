"""Unit tests for the record-only Pareto frontier.

See :mod:`zicato.tournament.pareto`. The frontier is observational. No caller
reads it back into a decision. These tests protect that condition first: the
shipped config is off, and the module shares no code with the scalar path that
controls promotion.
"""

from __future__ import annotations

import logging

import pytest

from zicato.core import DriftCount, LossProfile, MetricCount, ScoringWeights
from zicato.tournament.pareto import (
    DEFAULT_MAX_FRONTIER_SIZE,
    INTERNAL_DEV_PARETO_CONFIG,
    INTERNAL_PARETO_CONFIG,
    Axis,
    Candidate,
    Frontier,
    FrontierLedger,
    ParetoConfig,
    build_round_record,
    candidate_from_losses,
    dominates,
    monotonic_regressions,
    namespace_means,
    resolve_axes,
)

# The shipped defaults are drift: 1.0, cost: 0.001, latency: 0.0001,
# rubric: -1.0, output: 0.0, and schema: 5.0. The promote_margin is 0.01.
WEIGHTS = ScoringWeights()

DRIFT = Axis(namespace="drift:", weight=1.0)
COST = Axis(namespace="cost:", weight=0.001)
AXES = (DRIFT, COST)

# 0.01 scalar points. That is 0.01 raw on drift, and 10 tokens on cost.
MARGIN = 0.01


def _loss(
    entry_id: str,
    *,
    drift_loss: float = 0.0,
    tokens: int = 0,
    metric_counts: tuple[MetricCount, ...] = (),
    generation_id: str = "v1",
) -> LossProfile:
    """Return a small LossProfile. It holds only the fields that the frontier reads."""
    return LossProfile(
        run_id=f"run-{entry_id}",
        entry_id=entry_id,
        generation_id=generation_id,
        epoch_id="e0",
        drift_counts=(DriftCount(kind="off_topic", severity="info", count=0),),
        plan_revisions=0,
        task_failure_ratio=0.0,
        runtime_ms=1000,
        wall_clock_budget_exceeded=False,
        expectation_result=None,
        drift_loss=drift_loss,
        pass_fail=True,
        metric_counts=metric_counts,
        tokens_spent=tokens,
    )


def _cand(gid: str, drift: float, cost: float, **kw: object) -> Candidate:
    """A candidate positioned on both the namespace key and the metric key.

    A real candidate carries both levels, so a test fixture must too. The
    shipped dev config uses ``cost:tokens_spent``; the ``AXES`` constant in
    this file uses ``cost:``. Both must resolve against this fixture.
    """
    return Candidate(
        generation_id=gid,
        round_index=int(kw.pop("round_index", 0)),  # type: ignore[call-overload]
        axes={"drift:": drift, "cost:": cost, "cost:tokens_spent": cost},
        **kw,  # type: ignore[arg-type]
    )


def _frontier(*, max_size: int = DEFAULT_MAX_FRONTIER_SIZE) -> Frontier:
    return Frontier(epoch_id="e0", axes=AXES, margin=MARGIN, max_size=max_size)


# ---------------------------------------------------------------------------
# The shipped config is OFF
# ---------------------------------------------------------------------------


def test_shipped_config_is_disabled() -> None:
    """The single construction site must ship in the disabled state.

    This guards the main condition of the feature. A default that is enabled
    by accident would add unreviewed work to every production round.
    """
    assert INTERNAL_PARETO_CONFIG.enabled is False
    assert ParetoConfig().enabled is False


def test_disabled_config_records_nothing() -> None:
    record = build_round_record(
        epoch_id="e0",
        candidate=_cand("v1", 0.5, 1000.0),
        config=INTERNAL_PARETO_CONFIG,
        weights=WEIGHTS,
    )
    assert record is None


def test_dev_config_resolves_against_shipped_weights() -> None:
    """The first axis set must resolve. It must not drop without a message."""
    axes = resolve_axes(INTERNAL_DEV_PARETO_CONFIG, WEIGHTS)
    assert [a.namespace for a in axes] == ["drift:", "cost:tokens_spent"]


# ---------------------------------------------------------------------------
# Axis resolution + orientation
# ---------------------------------------------------------------------------


def test_resolve_axes_drops_zero_weight_namespace(caplog: pytest.LogCaptureFixture) -> None:
    """``output:`` ships at 0.0. It has no sign, so it cannot be an axis."""
    with caplog.at_level(logging.WARNING):
        axes = resolve_axes(ParetoConfig(enabled=True, metrics=("drift:", "output:")), WEIGHTS)
    assert [a.namespace for a in axes] == ["drift:"]
    assert "output:" in caplog.text


def test_resolve_axes_drops_unknown_namespace() -> None:
    axes = resolve_axes(ParetoConfig(enabled=True, metrics=("nope:",)), WEIGHTS)
    assert axes == ()


def test_resolve_axes_preserves_configured_order() -> None:
    axes = resolve_axes(ParetoConfig(enabled=True, metrics=("cost:", "drift:")), WEIGHTS)
    assert [a.namespace for a in axes] == ["cost:", "drift:"]


def test_negative_weight_flips_orientation() -> None:
    """``rubric:`` has a negative weight. A higher raw score is better there.

    The points method must therefore reverse the order for that axis.
    """
    rubric = Axis(namespace="rubric:", weight=-1.0)
    assert rubric.points(0.9) < rubric.points(0.4)
    # An axis with a positive weight keeps the rule that higher is worse.
    assert DRIFT.points(0.9) > DRIFT.points(0.4)


def test_negative_weight_axis_dominance_prefers_higher_raw() -> None:
    axes = (Axis(namespace="rubric:", weight=-1.0),)
    better = Candidate(generation_id="hi", round_index=0, axes={"rubric:": 0.9})
    worse = Candidate(generation_id="lo", round_index=0, axes={"rubric:": 0.4})
    assert dominates(better, worse, axes, MARGIN)
    assert not dominates(worse, better, axes, MARGIN)


# ---------------------------------------------------------------------------
# namespace_means
# ---------------------------------------------------------------------------


def test_namespace_means_empty_input() -> None:
    assert namespace_means([]) == {}


def test_namespace_means_drift_uses_drift_loss_not_mirrors() -> None:
    """Drift comes from the reducer field, not from the MetricCount copies.

    A profile can hold both ``drift_loss`` and ``drift:`` metric counts. The
    function must use only ``drift_loss``. If it used both, it would count the
    drift two times.
    """
    losses = [
        _loss("a", drift_loss=0.4, metric_counts=(MetricCount(name="drift:x", count=99.0),)),
        _loss("b", drift_loss=0.2, metric_counts=(MetricCount(name="drift:x", count=99.0),)),
    ]
    assert namespace_means(losses)["drift:"] == pytest.approx(0.3)


def test_namespace_means_is_unweighted() -> None:
    """The separate loop exists for one reason: it does not multiply by weights.

    The weighted function gives ``cost:`` as 150 * 0.001 = 0.15.
    """
    losses = [_loss("a", tokens=100), _loss("b", tokens=200)]
    assert namespace_means(losses)["cost:"] == pytest.approx(150.0)


def test_namespace_means_sums_within_a_run_then_averages() -> None:
    """Sum two values of one namespace in a run. Then take the mean of the runs."""
    losses = [
        _loss(
            "a",
            metric_counts=(
                MetricCount(name="cost:tokens_spent", count=100.0),
                MetricCount(name="cost:input_tokens", count=50.0),
            ),
        ),
        _loss("b", metric_counts=(MetricCount(name="cost:tokens_spent", count=50.0),)),
    ]
    assert namespace_means(losses)["cost:"] == pytest.approx(100.0)


def test_namespace_means_reports_both_namespace_and_metric_keys() -> None:
    """An axis can be a namespace key or a metric key. Both must be present."""
    losses = [
        _loss(
            "a",
            metric_counts=(
                MetricCount(name="cost:tokens_spent", count=100.0),
                MetricCount(name="cost:llm_calls", count=3.0),
            ),
        )
    ]
    means = namespace_means(losses)
    assert means["cost:tokens_spent"] == pytest.approx(100.0)
    assert means["cost:llm_calls"] == pytest.approx(3.0)
    # The namespace key still sums the two, which matches the scalar.
    assert means["cost:"] == pytest.approx(103.0)


def test_metric_key_axis_separates_what_the_namespace_conflates() -> None:
    """``cost:`` adds calls to tokens. ``cost:tokens_spent`` does not.

    Two candidates spend the same tokens but make a different number of calls.
    The namespace key separates them. The metric key correctly ties them.
    """

    def at(calls: float, tokens: float) -> dict[str, float]:
        return namespace_means(
            [
                _loss(
                    "a",
                    metric_counts=(
                        MetricCount(name="cost:llm_calls", count=calls),
                        MetricCount(name="cost:tokens_spent", count=tokens),
                    ),
                )
            ]
        )

    few = at(2.0, 1000.0)
    many = at(40.0, 1000.0)
    assert few["cost:"] != many["cost:"]
    assert few["cost:tokens_spent"] == many["cost:tokens_spent"]


def test_drift_kinds_appear_as_metric_keys_but_do_not_double_count() -> None:
    """``drift:`` stays the reducer aggregate. The kinds are separate keys."""
    losses = [
        _loss(
            "a",
            drift_loss=0.5,
            metric_counts=(MetricCount(name="drift:off_topic", count=2.0),),
        )
    ]
    means = namespace_means(losses)
    assert means["drift:"] == pytest.approx(0.5)
    assert means["drift:off_topic"] == pytest.approx(2.0)


def test_resolve_axes_accepts_a_metric_key() -> None:
    """A metric key reads the weight of its namespace."""
    axes = resolve_axes(ParetoConfig(enabled=True, metrics=("cost:tokens_spent",)), WEIGHTS)
    assert [a.namespace for a in axes] == ["cost:tokens_spent"]
    assert axes[0].weight == pytest.approx(WEIGHTS.namespace_weights["cost:"])


def test_resolve_axes_drops_a_metric_key_in_a_zero_weight_namespace() -> None:
    """``output:`` ships at 0.0, so ``output:chars`` is unusable too."""
    assert resolve_axes(ParetoConfig(enabled=True, metrics=("output:chars",)), WEIGHTS) == ()


def test_namespace_means_ignores_unnamespaced_metrics() -> None:
    losses = [_loss("a", metric_counts=(MetricCount(name="bare", count=5.0),))]
    assert "bare" not in namespace_means(losses)
    assert "" not in namespace_means(losses)


def test_namespace_means_run_without_namespace_contributes_zero() -> None:
    """A run with no value in a namespace lowers the mean. It is not ignored."""
    losses = [
        _loss("a", metric_counts=(MetricCount(name="schema:failures", count=4.0),)),
        _loss("b"),
    ]
    assert namespace_means(losses)["schema:"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------


def test_strictly_better_candidate_dominates() -> None:
    better = _cand("better", 0.400, 900.0)
    worse = _cand("worse", 0.500, 1000.0)
    assert dominates(better, worse, AXES, MARGIN)
    assert not dominates(worse, better, AXES, MARGIN)


def test_incomparable_candidates_do_not_dominate() -> None:
    """Better on quality and worse on cost. The scalar discards this trade-off."""
    cheap = _cand("cheap", 0.500, 500.0)
    accurate = _cand("accurate", 0.300, 5000.0)
    assert not dominates(cheap, accurate, AXES, MARGIN)
    assert not dominates(accurate, cheap, AXES, MARGIN)


def test_sub_margin_win_is_a_tie_not_dominance() -> None:
    """A drift lead of 0.001 is inside the noise. The margin covers that noise."""
    a = _cand("a", 0.499, 1000.0)
    b = _cand("b", 0.500, 1000.0)
    assert not dominates(a, b, AXES, MARGIN)
    assert not dominates(b, a, AXES, MARGIN)


def test_margin_is_scaled_per_axis_into_scalar_points() -> None:
    """0.01 scalar points equals 0.01 drift, or 10 tokens. One margin, two units."""
    # A difference of 5 tokens is inside the margin on cost.
    assert not dominates(_cand("a", 0.5, 995.0), _cand("b", 0.5, 1000.0), AXES, MARGIN)
    # A difference of 50 tokens clears the margin.
    assert dominates(_cand("a", 0.5, 950.0), _cand("b", 0.5, 1000.0), AXES, MARGIN)


def test_missing_axis_value_does_not_manufacture_dominance() -> None:
    """An axis with no measurement gives neither a win nor a loss."""
    partial = Candidate(generation_id="partial", round_index=0, axes={"drift:": 0.1})
    full = _cand("full", 0.5, 1000.0)
    assert dominates(partial, full, AXES, MARGIN)  # wins on the axis it has
    unmeasured = Candidate(generation_id="none", round_index=0, axes={})
    assert not dominates(unmeasured, full, AXES, MARGIN)
    assert not dominates(full, unmeasured, AXES, MARGIN)


# ---------------------------------------------------------------------------
# Frontier.update
# ---------------------------------------------------------------------------


def test_first_candidate_is_admitted() -> None:
    frontier = _frontier()
    update = frontier.update(_cand("v1", 0.5, 1000.0))
    assert update.admitted
    assert [m.generation_id for m in frontier.members] == ["v1"]


def test_dominated_candidate_is_rejected() -> None:
    frontier = _frontier()
    frontier.update(_cand("good", 0.300, 500.0))
    update = frontier.update(_cand("bad", 0.600, 5000.0))
    assert update.rejected
    assert update.blocked_by == "good"
    assert [m.generation_id for m in frontier.members] == ["good"]


def test_dominating_candidate_evicts_the_member_it_beats() -> None:
    frontier = _frontier()
    frontier.update(_cand("old", 0.500, 1000.0))
    update = frontier.update(_cand("new", 0.300, 500.0))
    assert update.admitted
    assert update.evicted == ("old",)
    assert [m.generation_id for m in frontier.members] == ["new"]


def test_incomparable_candidates_both_survive() -> None:
    """This behaviour is the reason for the feature."""
    frontier = _frontier()
    frontier.update(_cand("accurate", 0.300, 5000.0))
    update = frontier.update(_cand("cheap", 0.500, 500.0))
    assert update.admitted
    assert update.evicted == ()
    assert {m.generation_id for m in frontier.members} == {"accurate", "cheap"}


def test_tied_candidate_is_rejected_as_redundant() -> None:
    """Equal inside the margin on every axis. There is nothing to record.

    Without this rule, axes that move together fill the frontier with equal
    entries. The size guard then fires on correct data.
    """
    frontier = _frontier()
    frontier.update(_cand("first", 0.500, 1000.0))
    update = frontier.update(_cand("clone", 0.5005, 1002.0))
    assert update.rejected
    assert update.blocked_by == "first"
    assert len(frontier.members) == 1


def test_reoffering_same_generation_replaces_in_place() -> None:
    """A round that runs again moves its position. It is not counted two times."""
    frontier = _frontier()
    frontier.update(_cand("v1", 0.500, 1000.0))
    frontier.update(_cand("v1", 0.200, 400.0))
    assert [m.generation_id for m in frontier.members] == ["v1"]
    assert frontier.members[0].axes["drift:"] == pytest.approx(0.200)


def test_one_candidate_can_evict_several() -> None:
    frontier = _frontier()
    frontier.update(_cand("a", 0.500, 5000.0))
    frontier.update(_cand("b", 0.900, 1000.0))
    update = frontier.update(_cand("best", 0.100, 100.0))
    assert update.admitted
    assert set(update.evicted) == {"a", "b"}
    assert [m.generation_id for m in frontier.members] == ["best"]


def test_no_axes_records_nothing() -> None:
    """A frontier with no axes admits no candidate. It does not admit all of them."""
    frontier = Frontier(epoch_id="e0", axes=(), margin=MARGIN)
    assert frontier.update(_cand("v1", 0.1, 1.0)).rejected
    assert frontier.members == []


def test_size_guard_evicts_oldest_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """The guard limits the memory. The warning is the true signal."""
    frontier = _frontier(max_size=2)
    # No candidate dominates another. Each one is cheaper than the last one,
    # but worse on drift.
    frontier.update(_cand("a", 0.100, 5000.0))
    frontier.update(_cand("b", 0.500, 2000.0))
    with caplog.at_level(logging.WARNING):
        update = frontier.update(_cand("c", 0.900, 100.0))
    assert update.admitted
    assert update.overflowed == "a"
    assert [m.generation_id for m in frontier.members] == ["b", "c"]
    assert "dropped its oldest member" in caplog.text


# ---------------------------------------------------------------------------
# Leaders / summary
# ---------------------------------------------------------------------------


def test_leaders_reports_best_generation_per_axis() -> None:
    frontier = _frontier()
    frontier.update(_cand("accurate", 0.100, 5000.0))
    frontier.update(_cand("cheap", 0.900, 100.0))
    assert frontier.leaders() == {"drift:": "accurate", "cost:": "cheap"}


def test_leaders_ties_resolve_to_earliest_member() -> None:
    frontier = _frontier()
    frontier.update(_cand("first", 0.100, 5000.0))
    frontier.update(_cand("second", 0.900, 100.0))
    # Both members have measurements. The drift leader is clear. Repeat calls
    # give the same result.
    assert frontier.leaders() == frontier.leaders()


def test_summary_of_empty_frontier() -> None:
    assert _frontier().summary() == "epoch=e0 frontier=empty"


def test_summary_counts_clean_members() -> None:
    frontier = _frontier()
    frontier.update(_cand("ok", 0.100, 5000.0))
    frontier.update(_cand("dirty", 0.900, 100.0, regressions=("rubric:",)))
    summary = frontier.summary()
    assert "size=2" in summary
    assert "clean=1/2" in summary


# ---------------------------------------------------------------------------
# Monotonicity tagging
# ---------------------------------------------------------------------------


def test_monotonic_regression_flags_guarded_namespace() -> None:
    """``schema:`` ships as a guarded namespace. New failures are a regression."""
    got = monotonic_regressions({"schema:": 0.0}, {"schema:": 2.0}, WEIGHTS)
    assert got == ("schema:",)


def test_monotonic_regression_is_sign_aware() -> None:
    """``rubric:`` has a negative weight. A lower raw score is thus the regression."""
    assert monotonic_regressions({"rubric:": 0.9}, {"rubric:": 0.4}, WEIGHTS) == ("rubric:",)
    assert monotonic_regressions({"rubric:": 0.4}, {"rubric:": 0.9}, WEIGHTS) == ()


def test_unguarded_namespace_regression_is_not_flagged() -> None:
    """``drift:`` ships unguarded. A proposer can trade drift for a gain elsewhere."""
    assert monotonic_regressions({"drift:": 0.1}, {"drift:": 0.9}, WEIGHTS) == ()


def test_monotonic_regression_treats_absence_as_zero_not_unmeasured() -> None:
    """A clean parent emits no ``schema:`` value at all.

    The reducer writes a metric only when the metric is not zero. To read the
    absent value as "not measured" would miss the exact fault that the flag
    catches: a child that adds failures.
    """
    assert monotonic_regressions({}, {"schema:": 5.0}, WEIGHTS) == ("schema:",)
    # The opposite case: the child removed the failures, which is better.
    assert monotonic_regressions({"schema:": 5.0}, {}, WEIGHTS) == ()


def test_monotonic_regression_skips_namespace_neither_side_measured() -> None:
    assert monotonic_regressions({}, {}, WEIGHTS) == ()


def test_monotonic_tagging_does_not_block_admission() -> None:
    """The module tags the fault but does not apply it to admission.

    Milestone 1 records both, so that one run answers both questions.
    """
    frontier = _frontier()
    frontier.update(_cand("clean", 0.500, 5000.0))
    dirty = _cand("degenerate", 0.900, 10.0, regressions=("rubric:",))
    update = frontier.update(dirty)
    assert update.admitted
    assert not dirty.monotonic_ok
    assert "degenerate" in [m.generation_id for m in frontier.members]


# ---------------------------------------------------------------------------
# candidate_from_losses
# ---------------------------------------------------------------------------


def test_candidate_from_losses_scores_child_and_tags_regressions() -> None:
    parent = [
        _loss("a", drift_loss=0.4, tokens=1000, generation_id="v0"),
        _loss("b", drift_loss=0.4, tokens=1000, generation_id="v0"),
    ]
    child = [
        _loss("a", drift_loss=0.2, tokens=500),
        _loss(
            "b",
            drift_loss=0.2,
            tokens=500,
            metric_counts=(
                MetricCount(name="cost:tokens_spent", count=500.0),
                MetricCount(name="schema:failures", count=3.0),
            ),
        ),
    ]
    cand = candidate_from_losses(
        generation_id="v1",
        round_index=2,
        parent_losses=parent,
        child_losses=child,
        weights=WEIGHTS,
        decision="rejected",
    )
    assert cand.generation_id == "v1"
    assert cand.round_index == 2
    assert cand.decision == "rejected"
    assert cand.axes["drift:"] == pytest.approx(0.2)
    # The child added schema failures. The parent had none.
    assert cand.regressions == ("schema:",)
    assert not cand.monotonic_ok


def test_candidate_retains_the_gate_verdict() -> None:
    """The scalar makes a frontier member into evidence.

    You cannot calculate either value from the axes. The scalar includes terms
    that the namespace means do not hold.
    """
    cand = candidate_from_losses(
        generation_id="v1",
        round_index=0,
        parent_losses=[_loss("a", drift_loss=0.4, generation_id="v0")],
        child_losses=[_loss("a", drift_loss=0.2)],
        weights=WEIGHTS,
        decision="rejected",
        scalar=0.42,
        delta_scalar=-0.003,
    )
    assert cand.scalar == pytest.approx(0.42)
    # The gate refused this candidate although it improved. The value -0.003
    # does not clear the margin of 0.01. This is the exact candidate that the
    # frontier must remember.
    assert cand.delta_scalar == pytest.approx(-0.003)
    assert cand.decision == "rejected"


def test_candidate_scalar_defaults_to_none() -> None:
    """A caller that holds no scalar is still correct."""
    cand = candidate_from_losses(
        generation_id="v1",
        round_index=0,
        parent_losses=[],
        child_losses=[_loss("a", drift_loss=0.2)],
        weights=WEIGHTS,
    )
    assert cand.scalar is None
    assert cand.delta_scalar is None


def test_candidate_records_every_measured_namespace_not_only_configured_axes() -> None:
    """Extra namespaces cost nothing to record.

    A later change of the axis set can then use the rounds that already ran.
    """
    child = [_loss("a", drift_loss=0.1, metric_counts=(MetricCount(name="latency:ms", count=5.0),))]
    cand = candidate_from_losses(
        generation_id="v1",
        round_index=0,
        parent_losses=[],
        child_losses=child,
        weights=WEIGHTS,
    )
    assert "latency:" in cand.axes


# ---------------------------------------------------------------------------
# build_round_record + FrontierLedger
# ---------------------------------------------------------------------------


def _record(
    gid: str,
    drift: float,
    cost: float,
    *,
    epoch_id: str = "e0",
    delta_scalar: float | None = None,
) -> object:
    return build_round_record(
        epoch_id=epoch_id,
        candidate=_cand(gid, drift, cost, delta_scalar=delta_scalar),
        config=INTERNAL_DEV_PARETO_CONFIG,
        weights=WEIGHTS,
    )


def test_build_round_record_carries_resolved_terms() -> None:
    rec = build_round_record(
        epoch_id="e0",
        candidate=_cand("v1", 0.5, 1000.0),
        config=INTERNAL_DEV_PARETO_CONFIG,
        weights=WEIGHTS,
    )
    assert rec is not None
    assert [a.namespace for a in rec.axes] == ["drift:", "cost:tokens_spent"]
    assert rec.margin == pytest.approx(WEIGHTS.promote_margin)


def test_build_round_record_none_when_no_axis_resolves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        rec = build_round_record(
            epoch_id="e0",
            candidate=_cand("v1", 0.5, 1000.0),
            config=ParetoConfig(enabled=True, metrics=("output:",)),
            weights=WEIGHTS,
        )
    assert rec is None


def test_ledger_ignores_none() -> None:
    ledger = FrontierLedger()
    assert ledger.record(None) is None
    assert ledger.frontier is None


def test_ledger_accumulates_within_one_epoch() -> None:
    """A promotion does not reset the frontier. The board and the measures are the same."""
    ledger = FrontierLedger()
    ledger.record(_record("accurate", 0.100, 5000.0))  # type: ignore[arg-type]
    ledger.record(_record("cheap", 0.900, 100.0))  # type: ignore[arg-type]
    frontier = ledger.frontier
    assert frontier is not None
    assert {m.generation_id for m in frontier.members} == {"accurate", "cheap"}


def test_ledger_resets_on_epoch_roll() -> None:
    """A new contract gives new measurements.

    To keep the old values would compare scores from two different boards.
    """
    ledger = FrontierLedger()
    ledger.record(_record("old", 0.100, 100.0))  # type: ignore[arg-type]
    ledger.record(_record("new", 0.500, 5000.0, epoch_id="e1"))  # type: ignore[arg-type]
    frontier = ledger.frontier
    assert frontier is not None
    assert frontier.epoch_id == "e1"
    assert [m.generation_id for m in frontier.members] == ["new"]


def test_ledger_returns_the_update_for_logging() -> None:
    ledger = FrontierLedger()
    first = ledger.record(_record("a", 0.100, 5000.0))  # type: ignore[arg-type]
    assert first is not None and first.admitted
    dominated = ledger.record(_record("b", 0.900, 9000.0))  # type: ignore[arg-type]
    assert dominated is not None and dominated.rejected
    assert dominated.blocked_by == "a"


# ---------------------------------------------------------------------------
# Wiring: the orchestrator seam and the evolve-loop seam
# ---------------------------------------------------------------------------


class _FakeTournamentResult:
    def __init__(self, per_entry_losses: object) -> None:
        self.per_entry_losses = per_entry_losses


def _paired_result() -> _FakeTournamentResult:
    return _FakeTournamentResult(
        {
            "a": (_loss("a", drift_loss=0.4, tokens=1000), _loss("a", drift_loss=0.2, tokens=500)),
            "b": (_loss("b", drift_loss=0.4, tokens=1000), _loss("b", drift_loss=0.2, tokens=500)),
        }
    )


def test_round_seam_is_inert_while_shipped_config_is_disabled() -> None:
    """The production path makes no record. It also does no work to decide that."""
    from zicato import orchestrator

    assert orchestrator.INTERNAL_PARETO_CONFIG.enabled is False
    got = orchestrator._pareto_record_for_round(
        epoch_id="e0",
        generation_id="v1",
        round_index=0,
        tournament_result=_paired_result(),
        weights=WEIGHTS,
        decision="rejected",
    )
    assert got is None


def test_round_seam_builds_a_record_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from zicato import orchestrator

    monkeypatch.setattr(orchestrator, "INTERNAL_PARETO_CONFIG", INTERNAL_DEV_PARETO_CONFIG)
    got = orchestrator._pareto_record_for_round(
        epoch_id="e0",
        generation_id="v1",
        round_index=3,
        tournament_result=_paired_result(),
        weights=WEIGHTS,
        decision="promoted",
    )
    assert got is not None
    assert got.epoch_id == "e0"
    assert got.candidate.generation_id == "v1"
    assert got.candidate.decision == "promoted"
    assert got.candidate.axes["drift:"] == pytest.approx(0.2)
    assert got.candidate.scalar is None


def test_round_seam_threads_the_gate_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller supplies the scalar. There is only one calculation of it."""
    from zicato import orchestrator

    monkeypatch.setattr(orchestrator, "INTERNAL_PARETO_CONFIG", INTERNAL_DEV_PARETO_CONFIG)
    got = orchestrator._pareto_record_for_round(
        epoch_id="e0",
        generation_id="v1",
        round_index=0,
        tournament_result=_paired_result(),
        weights=WEIGHTS,
        decision="rejected",
        scalar=0.42,
        delta_scalar=-0.003,
    )
    assert got is not None
    assert got.candidate.scalar == pytest.approx(0.42)
    assert got.candidate.delta_scalar == pytest.approx(-0.003)


def test_round_seam_returns_none_without_paired_losses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast mode can give a duel with no paired losses for each entry."""
    from zicato import orchestrator

    monkeypatch.setattr(orchestrator, "INTERNAL_PARETO_CONFIG", INTERNAL_DEV_PARETO_CONFIG)
    assert (
        orchestrator._pareto_record_for_round(
            epoch_id="e0",
            generation_id="v1",
            round_index=0,
            tournament_result=_FakeTournamentResult({}),
            weights=WEIGHTS,
            decision="rejected",
        )
        is None
    )


def test_round_seam_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record that no caller reads must not lose a completed round."""
    from zicato import orchestrator

    monkeypatch.setattr(orchestrator, "INTERNAL_PARETO_CONFIG", INTERNAL_DEV_PARETO_CONFIG)
    malformed = _FakeTournamentResult({"a": "not-a-pair"})
    assert (
        orchestrator._pareto_record_for_round(
            epoch_id="e0",
            generation_id="v1",
            round_index=0,
            tournament_result=malformed,
            weights=WEIGHTS,
            decision="rejected",
        )
        is None
    )


def test_outcome_defaults_to_no_pareto_record() -> None:
    """Compatibility: every existing construction site omits the new field."""
    from zicato.orchestrator import EvolveRoundOutcome

    outcome = EvolveRoundOutcome(
        parent_generation_id="v0",
        proposed_generation_id="v1",
        tournament_decision="rejected",
        rejection_reason="",
        parent_scalar=0.0,
        child_scalar=0.0,
        delta_scalar=0.0,
    )
    assert outcome.pareto_record is None


def test_loop_seam_is_a_noop_without_a_record(caplog: pytest.LogCaptureFixture) -> None:
    from zicato.evolve.loop import _record_pareto
    from zicato.orchestrator import EvolveRoundOutcome

    ledger = FrontierLedger()
    outcome = EvolveRoundOutcome(
        parent_generation_id="v0",
        proposed_generation_id="v1",
        tournament_decision="rejected",
        rejection_reason="",
        parent_scalar=0.0,
        child_scalar=0.0,
        delta_scalar=0.0,
    )
    with caplog.at_level(logging.INFO):
        _record_pareto(ledger, outcome)
    assert ledger.frontier is None
    assert caplog.text == ""


def test_loop_seam_folds_the_record_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    from zicato.evolve.loop import _record_pareto
    from zicato.orchestrator import EvolveRoundOutcome

    ledger = FrontierLedger()
    outcome = EvolveRoundOutcome(
        parent_generation_id="v0",
        proposed_generation_id="v1",
        tournament_decision="rejected",
        rejection_reason="",
        parent_scalar=0.0,
        child_scalar=0.0,
        delta_scalar=0.0,
        pareto_record=_record("v1", 0.2, 500.0),  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.INFO):
        _record_pareto(ledger, outcome)
    frontier = ledger.frontier
    assert frontier is not None
    assert [m.generation_id for m in frontier.members] == ["v1"]
    assert "admitted v1" in caplog.text


def test_loop_seam_logs_the_gate_verdict_alongside_the_frontier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The log must show the size of the refusal by the scalar.

    A refusal by a small amount and a refusal by a large amount must look
    different in the log.
    """
    from zicato.evolve.loop import _record_pareto
    from zicato.orchestrator import EvolveRoundOutcome

    outcome = EvolveRoundOutcome(
        parent_generation_id="v0",
        proposed_generation_id="v1",
        tournament_decision="rejected",
        rejection_reason="",
        parent_scalar=0.0,
        child_scalar=0.0,
        delta_scalar=0.0,
        pareto_record=_record("v1", 0.2, 500.0, delta_scalar=-0.003),  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.INFO):
        _record_pareto(FrontierLedger(), outcome)
    assert "-0.0030" in caplog.text


def test_loop_seam_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    """This observational work runs after the round writes its journal."""

    class _Exploding:
        @property
        def candidate(self) -> object:
            raise RuntimeError("boom")

    class _Outcome:
        pareto_record = _Exploding()

    from zicato.evolve.loop import _record_pareto

    _record_pareto(FrontierLedger(), _Outcome())  # type: ignore[arg-type]
