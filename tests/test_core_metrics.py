"""Named measurements and their derived scoring channels."""

from __future__ import annotations

import dataclasses

import pytest

from zicato.core import (
    ExpectedMetricMovement,
    HypothesisSpec,
    LossProfile,
    MetricCount,
    MetricMovementActual,
    OutcomeRecord,
    ScoringWeights,
)
from zicato.tournament.scoring import aggregate_namespaced_metrics

# ---------------------------------------------------------------------------
# Named metric construction
# ---------------------------------------------------------------------------


def test_metric_count_defaults_are_empty_severity_and_zero_count() -> None:
    mc = MetricCount(name="cost:input_tokens")
    assert mc.name == "cost:input_tokens"
    assert mc.severity == ""
    assert mc.count == 0.0


def test_metric_count_accepts_float_count() -> None:
    mc = MetricCount(name="rubric:slide_structure", count=3.5)
    assert mc.count == 3.5


def test_metric_count_accepts_drift_severity_buckets() -> None:
    for sev in ("info", "warning", "critical"):
        mc = MetricCount(name="drift:off_topic", severity=sev, count=1.0)  # type: ignore[arg-type]
        assert mc.severity == sev


def test_metric_count_is_frozen() -> None:
    mc = MetricCount(name="cost:tokens_spent", count=100.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mc.count = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Derived scoring channels
# ---------------------------------------------------------------------------


def _bare_profile(**overrides: object) -> LossProfile:
    """Build a minimal :class:`LossProfile` for scoring_metrics tests."""
    kwargs: dict[str, object] = {
        "run_id": "r1",
        "entry_id": "e1",
        "generation_id": "v0",
        "epoch_id": "epoch-001",
        "metric_counts": (),
        "plan_revisions": 0,
        "task_failure_ratio": 0.0,
        "runtime_ms": 1000,
        "wall_clock_budget_exceeded": False,
        "expectation_result": None,
        "drift_loss": 0.0,
        "pass_fail": None,
    }
    kwargs.update(overrides)
    return LossProfile(**kwargs)  # type: ignore[arg-type]


def test_scoring_metrics_with_only_metric_counts_yields_drift_namespace() -> None:
    profile = _bare_profile(
        metric_counts=(
            MetricCount(name="drift:off_topic", severity="warning", count=2),
            MetricCount(name="drift:tool_error", severity="info", count=1),
        ),
    )
    unified = profile.scoring_metrics()
    names = [m.name for m in unified]
    # The drift entries first, then the always-derived run-outcome channels.
    assert names == [
        "drift:off_topic",
        "drift:tool_error",
        "failure:tasks",
        "failure:not_completed",
        "runtime:seconds",
    ]
    drift = [m for m in unified if m.name.startswith("drift:")]
    assert {m.severity for m in drift} == {"warning", "info"}


def test_scoring_metrics_of_a_clean_run_is_the_derived_channels_at_zero() -> None:
    """A run with no signals still states its outcome, at zero.

    The ``failure:`` and ``runtime:`` members are emitted unconditionally so
    the metric key set does not depend on whether the run went wrong — a
    channel that appears only on failure cannot be aggregated across a
    generation.
    """
    assert {m.name: m.count for m in _bare_profile().scoring_metrics()} == {
        "failure:tasks": 0.0,
        "failure:not_completed": 0.0,
        "runtime:seconds": 1.0,
    }


def test_scoring_metrics_leaves_scorer_metrics_out_by_default() -> None:
    """A scorer metric is an observation until the contract opts its namespace in."""
    profile = _bare_profile(metrics={"rubric:clarity": 0.8, "size:words": -0.1})
    unified = {m.name for m in profile.scoring_metrics()}
    assert "rubric:clarity" not in unified
    assert "size:words" not in unified


def test_scoring_metrics_admits_only_listed_scorer_namespaces() -> None:
    profile = _bare_profile(
        metrics={"size:words": -0.1, "rubric:clarity": 0.8, "weighted_score": 0.9},
    )
    unified = {m.name: m.count for m in profile.scoring_metrics(("size:",))}
    assert unified["size:words"] == -0.1
    assert "rubric:clarity" not in unified
    assert "weighted_score" not in unified


def test_scoring_metrics_skips_non_finite_scorer_metrics() -> None:
    profile = _bare_profile(metrics={"size:words": float("nan")})
    assert "size:words" not in {m.name for m in profile.scoring_metrics(("size:",))}


def test_scorer_metrics_reach_the_aggregate_only_when_listed() -> None:
    """The default rubric: weight does not score a scorer's rubric: key."""
    profile = _bare_profile(metrics={"rubric:clarity": 0.8, "size:words": -0.1})
    default = ScoringWeights()
    assert aggregate_namespaced_metrics([profile], default)["rubric:"] == 0.0
    opted_in = dataclasses.replace(
        default,
        namespace_weights={**default.namespace_weights, "size:": 1.0},
        scorer_metric_namespaces=("size:",),
    )
    aggregates = aggregate_namespaced_metrics([profile], opted_in)
    assert aggregates["size:"] == pytest.approx(-0.1)
    assert aggregates["rubric:"] == 0.0


@pytest.mark.parametrize("namespace", ["failure:", "cost:"])
def test_scorer_metric_namespaces_refuses_a_reserved_namespace(namespace: str) -> None:
    # A scorer must not be able to add terms to a measured channel.
    with pytest.raises(ValueError, match="is reserved"):
        ScoringWeights(scorer_metric_namespaces=(namespace,))


@pytest.mark.parametrize("namespace", ["size", "a:b:", ":", " :", "my size:"])
def test_scorer_metric_namespaces_refuses_a_malformed_namespace(namespace: str) -> None:
    with pytest.raises(ValueError, match="must be one non-empty namespace"):
        ScoringWeights(scorer_metric_namespaces=(namespace,))


@pytest.mark.parametrize("value", [(1,), (["latency:"],), "latency:"])
def test_scorer_metric_namespaces_refuses_a_value_that_is_not_a_list_of_strings(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="list of namespace strings"):
        ScoringWeights(scorer_metric_namespaces=value)  # type: ignore[arg-type]


def _weights_scoring(*namespaces: str) -> ScoringWeights:
    default = ScoringWeights()
    return ScoringWeights(
        namespace_weights={**default.namespace_weights, **dict.fromkeys(namespaces, 1.0)},
        scorer_metric_namespaces=namespaces,
    )


def test_scorer_metric_namespaces_accepts_a_plain_namespace() -> None:
    assert _weights_scoring("size:").scorer_metric_namespaces == ("size:",)


def test_scorer_metric_namespaces_is_a_set_for_the_contract_hash() -> None:
    """Order and repeats do not change what is scored, so they do not change the hash."""
    from zicato.epoch.contract import scoring_to_canon

    listed = _weights_scoring("size:", "a:")
    shuffled = dataclasses.replace(listed, scorer_metric_namespaces=("a:", "size:", "a:"))
    assert shuffled.scorer_metric_namespaces == ("a:", "size:")
    as_set = dataclasses.replace(listed, scorer_metric_namespaces=frozenset({"size:", "a:"}))
    assert as_set.scorer_metric_namespaces == ("a:", "size:")
    assert scoring_to_canon(shuffled) == scoring_to_canon(listed)


def test_a_scorer_namespace_without_a_weight_is_refused() -> None:
    """It would aggregate at zero and never reach the scalar."""
    with pytest.raises(ValueError, match="has no namespace_weights coefficient"):
        ScoringWeights(scorer_metric_namespaces=("size:",))


def test_the_replicate_fold_averages_a_scored_metric_over_the_replicates_that_report_it() -> None:
    """The fold keeps the metrics rule "not measured is not zero" (issue #575)."""
    from zicato.tournament.scoring import average_replicate_losses

    weights = _weights_scoring("size:")
    four = _bare_profile(metrics={"size:words": 4.0})
    two = _bare_profile(metrics={"size:words": 2.0})
    silent = _bare_profile(metrics=None)
    folded = average_replicate_losses([{"e1": four}, {"e1": two}])["e1"]
    assert aggregate_namespaced_metrics([folded], weights)["size:"] == pytest.approx(3.0)
    # An aborted replicate reports no metric and does not vote; it pays
    # through the failure channel instead.
    folded = average_replicate_losses([{"e1": four}, {"e1": silent}])["e1"]
    assert aggregate_namespaced_metrics([folded], weights)["size:"] == pytest.approx(4.0)


def test_a_scored_namespace_is_meaned_over_the_entries_that_report_it() -> None:
    """A silent entry cannot pull a lower-is-better scorer term toward zero."""
    from zicato.tournament.scoring import average_replicate_losses

    weights = _weights_scoring("size:")
    ten = _bare_profile(metrics={"size:words": 10.0})
    silent = _bare_profile(metrics=None)
    assert aggregate_namespaced_metrics([ten, silent], weights)["size:"] == pytest.approx(10.0)
    folded = average_replicate_losses([{"e1": ten}, {"e1": silent}])["e1"]
    assert aggregate_namespaced_metrics([folded], weights)["size:"] == pytest.approx(10.0)
    # A measured namespace keeps counting a silent run as zero.
    cost = _bare_profile(
        metric_counts=(MetricCount(name="cost:llm_calls", severity="", count=4.0),)
    )
    assert aggregate_namespaced_metrics([cost, silent], weights)["cost:"] == pytest.approx(
        weights.namespace_weights["cost:"] * 2.0
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # The scorer chose a different name in each replicate.
        ({"size:x.txt": 10.0}, {"size:y.txt": 10.0}, 10.0),
        # One replicate did not report one of its names.
        ({"size:a": 4.0, "size:b": 2.0}, {"size:a": 4.0}, 5.0),
    ],
)
def test_the_fold_of_one_entry_gives_the_mean_of_its_replicate_totals(
    first: dict[str, float], second: dict[str, float], expected: float
) -> None:
    """The gate reads the fold, so the fold must not change a namespace total."""
    from zicato.tournament.scoring import average_replicate_losses

    weights = _weights_scoring("size:")
    runs = [_bare_profile(metrics=first), _bare_profile(metrics=second)]
    assert aggregate_namespaced_metrics(runs, weights)["size:"] == pytest.approx(expected)
    folded = average_replicate_losses([{"e1": runs[0]}, {"e1": runs[1]}])["e1"]
    assert aggregate_namespaced_metrics([folded], weights)["size:"] == pytest.approx(expected)


def test_a_scored_namespace_that_no_run_reports_has_no_aggregate() -> None:
    """Not measured is not zero: the gate and the Pareto frontier skip the namespace."""
    weights = _weights_scoring("size:")
    silent = _bare_profile(metrics=None)
    aggregates = aggregate_namespaced_metrics([silent, silent], weights)
    assert "size:" not in aggregates
    assert aggregates["cost:"] == 0.0


def test_a_measured_metric_in_a_scored_namespace_keeps_its_divisor() -> None:
    """Only the scorer part of a listed namespace skips the silent runs."""
    weights = _weights_scoring("size:")
    measured = _bare_profile(
        metric_counts=(MetricCount(name="size:files", severity="", count=4.0),)
    )
    silent = _bare_profile(metrics=None)
    assert aggregate_namespaced_metrics([measured, silent], weights)["size:"] == pytest.approx(2.0)
    scored = _bare_profile(
        metric_counts=(MetricCount(name="size:files", severity="", count=4.0),),
        metrics={"size:words": 6.0},
    )
    assert aggregate_namespaced_metrics([scored, silent], weights)["size:"] == pytest.approx(
        2.0 + 6.0
    )


def test_scoring_metrics_with_explicit_metric_counts_includes_all_namespaces() -> None:
    profile = _bare_profile(
        metric_counts=(
            MetricCount(name="drift:off_topic", severity="warning", count=1),
            MetricCount(name="cost:input_tokens", count=1500.0),
            MetricCount(name="cost:output_tokens", count=500.0),
            MetricCount(name="rubric:slide_structure", count=4.0),
            MetricCount(name="latency:p95_turn_ms", count=2400.0),
        )
    )
    names = {m.name for m in profile.scoring_metrics()}
    assert "drift:off_topic" in names
    assert "cost:input_tokens" in names
    assert "cost:output_tokens" in names
    assert "rubric:slide_structure" in names
    assert "latency:p95_turn_ms" in names
    # Distinct namespaces all surfaced, alongside the derived ones.
    namespaces = {m.name.split(":", 1)[0] for m in profile.scoring_metrics()}
    assert namespaces == {"drift", "cost", "rubric", "latency", "failure", "runtime"}


# ---------------------------------------------------------------------------
# HypothesisSpec — predictions keyed by measured metric name
# ---------------------------------------------------------------------------


def test_hypothesis_spec_accepts_metric_movements_alongside_drift() -> None:
    hyp = HypothesisSpec(
        core_idea="Cut token cost by trimming the prompt.",
        modulating=("router__system_prompt",),
        why="Cost dominates value at high token counts.",
        expected_pass_rate_delta="+0.00",
        expected_metric_movements=(
            ExpectedMetricMovement(
                metric_name="cost:tokens_spent", direction="decrease", magnitude="medium"
            ),
            ExpectedMetricMovement(
                metric_name="rubric:slide_structure",
                direction="increase_or_neutral",
                magnitude="small",
            ),
        ),
    )
    names = {m.metric_name for m in hyp.expected_metric_movements}
    assert names == {"cost:tokens_spent", "rubric:slide_structure"}


# ---------------------------------------------------------------------------
# OutcomeRecord — realized metric movements
# ---------------------------------------------------------------------------


def test_outcome_record_default_metric_movements_is_empty() -> None:
    outcome = OutcomeRecord(
        ran_at="2026-01-01T00:00:00Z",
        metric_movements=(),
        pass_rate_delta=0.0,
        drift_loss_delta=0.0,
        scalar_score_delta=0.0,
        tournament_decision="deferred",
    )
    assert outcome.metric_movements == ()


def test_outcome_record_accepts_metric_movements() -> None:
    outcome = OutcomeRecord(
        ran_at="2026-01-01T00:00:00Z",
        pass_rate_delta=0.0,
        drift_loss_delta=0.0,
        scalar_score_delta=0.0,
        tournament_decision="promoted",
        metric_movements=(
            MetricMovementActual(
                metric_name="cost:tokens_spent",
                from_value=2000.0,
                to_value=1500.0,
                hypothesis_match=True,
                note="trim worked",
            ),
        ),
    )
    assert outcome.metric_movements[0].metric_name == "cost:tokens_spent"
    assert outcome.metric_movements[0].from_value == 2000.0
    assert outcome.metric_movements[0].to_value == 1500.0
