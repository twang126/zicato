"""Pareto frontier of challengers. A record-only multi-objective ledger.

The promote gate makes one scalar from every objective. See
:func:`zicato.tournament.scoring.aggregate_generation_score`. The gate keeps
the one generation that clears ``promote_margin``.

That scalar is a weighted sum. Therefore the gate can discard the cheapest
challenger the loop ever built. The shipped weights show why: ``cost:`` is
``0.001`` and ``drift:`` is ``1.0``. A saving of 40% of the tokens moves the
scalar less than the noise floor moves it.

This module keeps a second record beside the champion lineage. The record is
observational only. It holds the challengers that no other challenger
dominates, across a configured set of objectives. A candidate is on the
frontier if no other candidate is as good on every objective and better on one
objective. You cannot get its strengths without a loss somewhere else.

This module does not change promotion. The module scores the frontier after
the gate decides. The frontier stays in memory for one evolve loop. No caller
reads it. The frontier exists to answer one question from real epochs: are the
discarded candidates real trade-offs? Answer that question before you build
the proposer channel or the dashboard view.

Design notes
------------
**An axis is a namespace key or a metric key.** A namespace key ends with a
colon, such as ``"cost:"``. A metric key names one metric, such as
``"cost:tokens_spent"``. Both read their weight from the namespace, because
every metric in one namespace shares the units of that namespace.

Prefer a metric key when a namespace mixes units. ``cost:`` sums
``cost:llm_calls`` and ``cost:tokens_spent``, so it adds a count of calls to a
count of tokens. That sum has no unit, and the two parts can move in opposite
directions. Note that the scalar path has the same behaviour.

Keep the axis count small either way. See the note on frontier size below.

**The sign of the weight gives the direction.** This module holds no table of
directions. A positive weight means that a higher value is worse. Drift, cost,
and schema have positive weights. A negative weight means that a higher value
is better. Rubric has a negative weight. Multiply a raw value by its signed
weight. Every axis then has the same direction: lower is better. This module
therefore agrees with the scalar about the meaning of "worse".

**The threshold is ``promote_margin``.** This module adds no threshold of its
own. The margin is the answer of the system to this question: which difference
is real? :mod:`zicato.tournament.calibration` measures an A/A noise floor and
sets the margin from it. The margin is a measurement, not a guess. Dominance
uses the same margin. The phrase "better on one objective" thus has one
meaning in the loop. Scalar points make one margin usable on all axes. The raw
units of the axes differ by several orders of magnitude.

The scalar margin is a strict bar for one axis, and this is deliberate. Assume
the axes are independent. The variance of the scalar is then the sum of the
variances of the axes: ``sigma_scalar**2 = sum((w_i * sigma_i)**2)``.
Therefore ``w_i * sigma_i <= sigma_scalar`` for each axis. The bar is thus at
least as strict as the noise of that axis needs. A strict bar gives a smaller
frontier. This is the safe direction, because noise is what makes a frontier
grow too large.

**The module stores raw values.** The weights stay frozen for an epoch, so
raw values and scalar points both work inside one epoch. Raw values keep their
meaning if the record outlives the config that scored it. A person who reads a
log also wants the raw values.

**The axis count controls the frontier size. A cap does not.** Dominance gets
weaker as the number of axes grows. With many axes, no candidate dominates
another candidate, and the frontier holds every candidate. For ``n``
candidates and ``k`` independent axes, the size grows as
``(ln n)**(k-1) / (k-1)!``. Two or three axes give a few members. Six axes
give half of the field. :data:`ParetoConfig.max_frontier_size` is a guard
against a fault, not a policy. If the guard fires, the axis set is wrong.

Epoch scoping
-------------
The frontier uses the epoch as its key. The frontier survives a promotion
inside one epoch. An epoch is a frozen contract. The board, the judges, and
the :class:`~zicato.core.ScoringWeights` all stay frozen. The loop therefore
measures each candidate of the epoch in the same way. The raw values stay
comparable, whichever generation is the champion.

An epoch roll changes what the loop measures. The frontier thus resets on an
epoch roll. It does not compare values across two contracts.

The genealogy pool of the proposer works in the opposite way. See
:func:`zicato.proposer.genealogy.sample_genealogy`. That pool uses the reign
as its scope, because it holds banded outcomes that are relative to the
champion. Such an outcome loses its meaning when the champion changes. This
record holds absolute measurements, which keep their meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from zicato.core import LossProfile, ScoringWeights

log = logging.getLogger(__name__)

#: Guard for a frontier that grows without a bound. Too many axes cause this
#: growth. See the note on frontier size in the module docstring. A config of
#: two or three axes stays far below this value.
DEFAULT_MAX_FRONTIER_SIZE: int = 32


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParetoConfig:
    """The objectives that the frontier tracks.

    :data:`INTERNAL_PARETO_CONFIG` is the single construction site. Change
    that constant to start a recording.

    Fields
    ------
    enabled:
        ``False`` is the default. Every entry point then does no work, makes
        no allocation, and leaves the evolve loop unchanged.
    metrics:
        The metric names to use as axes. Each name is one of two forms:

        * a namespace key, which ends with a colon, such as ``"cost:"``. It
          sums every metric in the namespace, which matches the scalar.
        * a metric key, such as ``"cost:tokens_spent"``. It measures one
          thing.

        Prefer a metric key when a namespace mixes units. ``cost:`` adds
        ``cost:llm_calls`` to ``cost:tokens_spent``, so its sum has no unit.

        Both forms read their weight from the namespace of the name.
        :func:`resolve_axes` drops a name whose namespace has no weight or a
        zero weight, and logs a warning. A zero weight has no sign, so it
        gives no direction. A zero weight also cannot scale a raw value into
        the scalar points that the margin uses. This rule drops every
        ``output:`` axis, because that namespace ships at ``0.0``.
        ``output:chars`` is the axis we want most, but we cannot use it yet.
        It needs a noise floor for each axis, which is separate work.
    max_frontier_size:
        A guard against a fault. Above this size the frontier drops its oldest
        member and logs a warning. A bounded record is better than an
        unbounded record, and the warning is the true signal.
    """

    enabled: bool = False
    metrics: tuple[str, ...] = ()
    max_frontier_size: int = DEFAULT_MAX_FRONTIER_SIZE


#: The one construction site. It is disabled, so the evolve loop is unchanged.
INTERNAL_PARETO_CONFIG: ParetoConfig = ParetoConfig(enabled=False)

#: The first axis set for a recording. Quality against spend is the clearest
#: trade-off in the metrics that the reducer emits today.
#:
#: The cost axis is the metric key ``cost:tokens_spent``, not the namespace
#: ``cost:``. The namespace sums ``cost:llm_calls`` and ``cost:tokens_spent``,
#: which adds a count of calls to a count of tokens. That sum has no unit, and
#: the two parts can move in opposite directions. The metric key measures one
#: thing.
#:
#: Remember this caution when you read the first results. Drift and spend
#: usually move together, because more reasoning is usually both better and
#: more expensive. Axes that move together give a small frontier. A small
#: frontier here is thus weak evidence against the idea. It is stronger
#: evidence that these two axes cannot separate the candidates.
#:
#: Two changes would add an independent axis. The reducer must emit
#: ``latency:``, which needs only a mirror of `LossProfile.runtime_ms`. A
#: custom judge must also emit under ``rubric:`` instead of ``drift:custom``,
#: which would separate quality from drift.
INTERNAL_DEV_PARETO_CONFIG: ParetoConfig = ParetoConfig(
    enabled=True,
    metrics=("drift:", "cost:tokens_spent"),
)


@dataclass(frozen=True, slots=True)
class Axis:
    """One resolved objective. It holds a metric name and its signed weight.

    ``namespace`` holds the name of the axis. The name is either a namespace
    key such as ``"cost:"``, or a metric key such as ``"cost:tokens_spent"``.
    Both forms read their weight from the namespace, because every metric in
    one namespace shares the units of that namespace.

    ``weight`` is never zero, because :func:`resolve_axes` drops a zero
    weight. The sign of the weight gives the direction. The size of the weight
    changes a raw value into scalar points. One margin thus serves every axis.
    """

    namespace: str
    weight: float

    def points(self, raw: float) -> float:
        """Return the raw value in scalar points. Lower is better on every axis.

        A negative weight reverses the order. ``rubric:`` has a negative
        weight, because a higher rubric score is better. The reversal makes
        all axes agree: after this step, each axis has a minimum as its best
        value.
        """
        return raw * self.weight


def resolve_axes(config: ParetoConfig, weights: ScoringWeights) -> tuple[Axis, ...]:
    """Change the configured metric names into axes. Drop the unusable ones.

    Each name is either a namespace key such as ``"cost:"``, or a metric key
    such as ``"cost:tokens_spent"``. The function reads the weight from the
    namespace of the name in both cases.

    This function drops a name whose namespace has no weight or a zero weight.
    A zero weight has no sign, so it gives no direction. A zero weight also
    scales every candidate to the same point. Such an axis would add nothing
    to dominance, and it would do so without a signal. The function therefore
    logs each name that it drops.

    The function returns the axes in the configured order. An empty result is
    correct and means "record nothing". :meth:`Frontier.update` then does no
    work. It does not admit every candidate to a frontier that has no axes.
    """
    axes: list[Axis] = []
    for name in config.metrics:
        namespace = _namespace_of(name) or name
        weight = float(weights.namespace_weights.get(namespace, 0.0))
        if weight == 0.0:
            log.warning(
                "pareto: dropped the axis %r. The weight of its namespace %r is %s. "
                "Such a weight gives no direction and cannot scale into scalar points.",
                name,
                namespace,
                "absent" if namespace not in weights.namespace_weights else "0.0",
            )
            continue
        axes.append(Axis(namespace=name, weight=weight))
    return tuple(axes)


# ---------------------------------------------------------------------------
# Scoring a candidate
# ---------------------------------------------------------------------------


def _namespace_of(metric_name: str) -> str:
    """Return the ``"<prefix>:"`` of a metric name. Return ``""`` if it has none.

    This follows the rule on :class:`zicato.core.MetricCount`. The helper
    stays local. It does not call the private helper of the same shape in
    :mod:`zicato.tournament.scoring`. This module shares no code with the
    scalar path, and that separation is deliberate.
    """
    head, sep, _ = metric_name.partition(":")
    return f"{head}:" if sep else ""


def namespace_means(losses: Sequence[LossProfile]) -> dict[str, float]:
    """Return the mean of each metric across ``losses``. No weights. Pure.

    This is the unweighted form of
    :func:`zicato.tournament.scoring.aggregate_namespaced_metrics`. It is a
    separate loop, not a change to that function. The scalar path carries the
    promotion decision. This record must not be able to disturb it.

    The result holds TWO levels of granularity, so an axis can be either one:

    * A namespace key, which ends with a colon. Example: ``"cost:"``. The
      value sums every metric in that namespace, which matches the scalar.
    * A metric key, which has a name after the colon. Example:
      ``"cost:tokens_spent"``.

    The two levels never collide, because a namespace key always ends at the
    colon and a metric key never does.

    A metric key is often the better axis. ``cost:`` sums ``cost:llm_calls``
    and ``cost:tokens_spent``, which adds a count of calls to a count of
    tokens. The sum has no unit, and the two parts can move in opposite
    directions. ``cost:tokens_spent`` measures one thing.

    The rules for each value:

    * ``"drift:"`` is the mean of
      :attr:`~zicato.core.LossProfile.drift_loss`. That field is the aggregate
      of the reducer. Do not sum the ``drift:`` MetricCount copies instead,
      because that counts the drift two times. The individual ``drift:<kind>``
      keys still appear, because they measure one kind each.
    * For each other name, sum the :class:`~zicato.core.MetricCount` values of
      one run, then take the mean across the runs. A run with no value for a
      name adds zero to the sum. The rest of the aggregation uses the same
      rule.
    * The function ignores a metric name that has no namespace.

    The function returns ``{}`` for empty input. A candidate with no
    measurement has no position.
    """
    if not losses:
        return {}

    n = len(losses)
    means: dict[str, float] = {"drift:": sum(loss.drift_loss for loss in losses) / n}

    sums: dict[str, float] = {}
    for loss in losses:
        per_loss: dict[str, float] = {}
        for mc in loss.unified_metrics():
            namespace = _namespace_of(mc.name)
            if not namespace:
                continue
            # The metric key always accumulates. The namespace key skips
            # drift:, because LossProfile.drift_loss above is the canonical
            # drift aggregate and a second sum would double-count it.
            per_loss[mc.name] = per_loss.get(mc.name, 0.0) + mc.count
            if namespace != "drift:":
                per_loss[namespace] = per_loss.get(namespace, 0.0) + mc.count
        for name, value in per_loss.items():
            sums[name] = sums.get(name, 0.0) + value

    for name, total in sums.items():
        means[name] = total / n
    return means


def monotonic_regressions(
    parent_means: Mapping[str, float],
    child_means: Mapping[str, float],
    weights: ScoringWeights,
) -> tuple[str, ...]:
    """Return the guarded namespaces where the child is worse than the parent.

    This function reads the same
    :attr:`~zicato.core.ScoringWeights.namespace_monotonicity` flags that the
    promote gate applies. It defines "worse" in the same way, from the sign of
    the weight of the namespace. It returns the namespaces in sorted order. An
    empty result means that the child broke no guarded namespace.

    The module records this tag on each candidate. The module does not apply
    the tag to admission, and that choice is deliberate. The frontier rewards
    a candidate that is best at one objective. A candidate can cut the tokens
    by 90% and destroy the quality. No other candidate dominates it, so it is
    a correct frontier member. These flags are the correction for that fault.
    But an early correction hides its own cost: we could not see which good
    candidates it removes. The tag costs one boolean value. One run can then
    answer both questions.
    """
    offenders: list[str] = []
    for namespace, guarded in weights.namespace_monotonicity.items():
        if not guarded:
            continue
        weight = float(weights.namespace_weights.get(namespace, 0.0))
        if weight == 0.0:
            continue
        parent = parent_means.get(namespace)
        child = child_means.get(namespace)
        if parent is None and child is None:
            # No side measured it. There is nothing to compare.
            continue
        # A value that one side omits reads as zero, not as "not measured".
        # The reducer emits a metric only when the metric is not zero. See
        # ``if self.schema_failures:``. A clean parent thus emits no
        # ``schema:`` value at all. If we read that as "not measured", we miss
        # the exact fault that these flags catch: a child that adds failures
        # where the parent had none. Zero also matches the rule that
        # :func:`namespace_means` uses, where a run with no value in a
        # namespace adds zero.
        parent_value = 0.0 if parent is None else parent
        child_value = 0.0 if child is None else child
        # Use the sign. A positive weight means that a higher raw value is
        # worse. Multiply both sides by the weight. "Worse" then means "went
        # up" on every axis.
        if child_value * weight > parent_value * weight:
            offenders.append(namespace)
    return tuple(sorted(offenders))


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One measured challenger with its position on the configured axes.

    ``axes`` holds the raw mean of each namespace, without weights. The module
    docstring gives the reason for the raw form. ``monotonic_ok`` is the tag
    from :func:`monotonic_regressions`. ``regressions`` names each guarded
    namespace that the child broke. A log line can thus name what broke.

    The verdict of the gate
    -----------------------
    ``scalar`` and ``delta_scalar`` hold the result of the promotion path for
    this same challenger. They make the record into evidence. The words "on
    the frontier" alone do not show the size of the disagreement. That size is
    the case for the feature, or against it. Look at a frontier member whose
    scalar equals the scalar of the champion, but which uses 40% fewer tokens.
    That member is the argument for the feature. A member that lost the scalar
    by a large amount is a candidate that the weights correctly discarded.

    You cannot calculate either value from :attr:`axes`. The scalar includes
    terms that the namespace means do not hold. Those terms are the pass rate
    and the optional parsimony term.

    The two values are valid over different ranges:

    * ``scalar`` is absolute. It stays comparable across every round of the
      epoch. This property also lets the frontier survive a promotion.
    * ``delta_scalar`` is relative to the champion that this challenger met.
      It answers a useful question: did the challenger lose by a small
      amount? But two deltas from two different reigns are not comparable.
    """

    generation_id: str
    round_index: int
    axes: Mapping[str, float]
    decision: str = ""
    regressions: tuple[str, ...] = ()
    #: The scalar score of the challenger. ``None`` if the caller had none.
    scalar: float | None = None
    #: ``child_scalar - parent_scalar``. A negative value is an improvement.
    #: It is relative to the champion of its round. See the class docstring.
    delta_scalar: float | None = None

    @property
    def monotonic_ok(self) -> bool:
        """Return ``True`` if the candidate broke no guarded namespace."""
        return not self.regressions


def _beats_somewhere(a: Candidate, b: Candidate, axes: Sequence[Axis], margin: float) -> bool:
    """Return ``True`` if ``a`` is better than ``b`` on at least one axis.

    The margin sets the size of a true win. A difference of one margin or less
    in scalar points is a tie, not a win. Without the margin, the usual noise
    of an evaluation makes almost every pair incomparable. The frontier then
    fills with noise.

    A missing value on an axis counts as no win. It does not count as a loss.
    An axis with no measurement must not create dominance in either
    direction.
    """
    for axis in axes:
        a_raw = a.axes.get(axis.namespace)
        b_raw = b.axes.get(axis.namespace)
        if a_raw is None or b_raw is None:
            continue
        if axis.points(a_raw) < axis.points(b_raw) - margin:
            return True
    return False


def dominates(a: Candidate, b: Candidate, axes: Sequence[Axis], margin: float) -> bool:
    """Return ``True`` if ``a`` dominates ``b``: better on one axis, worse on none."""
    return _beats_somewhere(a, b, axes, margin) and not _beats_somewhere(b, a, axes, margin)


@dataclass(frozen=True, slots=True)
class FrontierUpdate:
    """The result of one :meth:`Frontier.update` call. The caller logs it."""

    admitted: bool
    #: The member that dominated or tied the candidate, if the update refused
    #: the candidate.
    blocked_by: str = ""
    #: The members that the new candidate dominated and replaced.
    evicted: tuple[str, ...] = ()
    #: The oldest member, if the size guard dropped it.
    overflowed: str = ""

    @property
    def rejected(self) -> bool:
        return not self.admitted


@dataclass(slots=True)
class Frontier:
    """The non-dominated set of one epoch. It stays in memory. Nothing saves it.

    The frontier keeps the members in the order of their addition. The size
    guard needs that order, because the guard drops the oldest member first.
    """

    epoch_id: str
    axes: tuple[Axis, ...]
    margin: float
    max_size: int = DEFAULT_MAX_FRONTIER_SIZE
    members: list[Candidate] = field(default_factory=list)

    def update(self, candidate: Candidate) -> FrontierUpdate:
        """Offer ``candidate`` to the frontier. Return the result.

        The frontier admits the candidate only if the candidate is better than
        each current member on at least one axis. This one rule covers both
        cases of refusal:

        * A member dominates the candidate. The candidate beats it on no axis.
        * A member ties the candidate on every axis, inside the margin. Again
          the candidate wins on no axis, so the candidate adds nothing. The
          frontier does not record it.

        The tie case is more important than it looks. Axes that move together,
        such as drift and cost, put many candidates inside the margin of each
        other on all axes. If the frontier admitted them, it would fill with
        equal entries. The size guard would then fire on correct data.

        If the frontier already holds this generation id, the update replaces
        that member. A round that runs again thus moves its own position. The
        frontier does not count it two times.
        """
        if not self.axes:
            return FrontierUpdate(admitted=False)

        self.members = [m for m in self.members if m.generation_id != candidate.generation_id]

        for member in self.members:
            if not _beats_somewhere(candidate, member, self.axes, self.margin):
                return FrontierUpdate(admitted=False, blocked_by=member.generation_id)

        evicted = tuple(
            m.generation_id for m in self.members if dominates(candidate, m, self.axes, self.margin)
        )
        if evicted:
            self.members = [m for m in self.members if m.generation_id not in set(evicted)]
        self.members.append(candidate)

        overflowed = ""
        if len(self.members) > self.max_size:
            overflowed = self.members[0].generation_id
            self.members = self.members[1:]
            log.warning(
                "pareto: the frontier of epoch %s went above %d members. It dropped "
                "its oldest member (%s). Dominance is too weak here. The configured "
                "axes (%s) are probably too many, or they move together too much to "
                "separate the candidates.",
                self.epoch_id,
                self.max_size,
                overflowed,
                ", ".join(a.namespace for a in self.axes),
            )

        return FrontierUpdate(admitted=True, evicted=evicted, overflowed=overflowed)

    def leaders(self) -> dict[str, str]:
        """Return the best generation id for each axis. This is the clearest view.

        If two members tie, the function keeps the member that the frontier
        added first. The result is thus the same on each call. The function
        omits an axis that no member measured.
        """
        best: dict[str, str] = {}
        for axis in self.axes:
            leader: tuple[float, str] | None = None
            for member in self.members:
                raw = member.axes.get(axis.namespace)
                if raw is None:
                    continue
                points = axis.points(raw)
                if leader is None or points < leader[0]:
                    leader = (points, member.generation_id)
            if leader is not None:
                best[axis.namespace] = leader[1]
        return best

    def summary(self) -> str:
        """Return the state as one line of text, for the log of each round."""
        if not self.members:
            return f"epoch={self.epoch_id} frontier=empty"
        clean = sum(1 for m in self.members if m.monotonic_ok)
        leaders = ", ".join(f"{ns}->{gid}" for ns, gid in sorted(self.leaders().items()))
        return (
            f"epoch={self.epoch_id} size={len(self.members)} "
            f"clean={clean}/{len(self.members)} leaders=[{leaders}]"
        )


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """The result of one round: the candidate and the terms that scored it.

    The record holds the resolved axes and the margin with the candidate. The
    evolve loop can thus own the frontier. The loop does not load the
    :class:`~zicato.core.ScoringWeights` of the epoch itself. The round
    already holds the weights. To read the epoch config again in the loop
    would cost one file read for each round, for a record that no caller
    reads.

    The record is frozen and complete. It is thus also the full input for a
    test.
    """

    epoch_id: str
    candidate: Candidate
    axes: tuple[Axis, ...]
    margin: float
    max_frontier_size: int = DEFAULT_MAX_FRONTIER_SIZE

    def new_frontier(self) -> Frontier:
        """Return an empty frontier that uses the terms of this record."""
        return Frontier(
            epoch_id=self.epoch_id,
            axes=self.axes,
            margin=self.margin,
            max_size=self.max_frontier_size,
        )


def build_round_record(
    *,
    epoch_id: str,
    candidate: Candidate,
    config: ParetoConfig,
    weights: ScoringWeights,
) -> RoundRecord | None:
    """Make a record for the ledger. Return ``None`` if there is nothing to record.

    The function returns ``None`` in two cases: the config is disabled, or no
    configured namespace gives a usable axis. The caller treats both cases in
    the same way, as "the feature is off".
    """
    if not config.enabled:
        return None
    axes = resolve_axes(config, weights)
    if not axes:
        log.warning(
            "pareto: the config is enabled, but no usable axis came from %r. "
            "The module records nothing.",
            config.metrics,
        )
        return None
    return RoundRecord(
        epoch_id=epoch_id,
        candidate=candidate,
        axes=axes,
        margin=float(weights.promote_margin),
        max_frontier_size=config.max_frontier_size,
    )


class FrontierLedger:
    """The frontiers of one evolve loop. The epoch is the key.

    The ledger holds the epoch rule. A record for a new epoch starts a new
    frontier and drops the previous frontier. An epoch roll changes the board,
    the judges, or the weights. Values from the old contract are thus not
    comparable with values from the new contract. To keep them would compare
    two different measurements, and no message would show the error. A
    promotion inside one epoch changes nothing. A new champion does not make
    the raw scores of the other candidates invalid.

    The ledger keeps only the frontier of the current epoch. Nothing saves the
    frontier. The frontier of a rolled epoch is thus lost when the loop
    continues. This is the intended scope of milestone 1. The log line of each
    round is the record.
    """

    def __init__(self) -> None:
        self._frontier: Frontier | None = None

    @property
    def frontier(self) -> Frontier | None:
        """Return the frontier of the current epoch, or ``None`` before the first record."""
        return self._frontier

    def record(self, rec: RoundRecord | None) -> FrontierUpdate | None:
        """Add one round to the frontier. ``None`` input gives ``None`` output.

        The method returns the update, so that the caller can log the result.
        It returns ``None`` if there was nothing to record.
        """
        if rec is None:
            return None
        if self._frontier is None or self._frontier.epoch_id != rec.epoch_id:
            self._frontier = rec.new_frontier()
        return self._frontier.update(rec.candidate)


def candidate_from_losses(
    *,
    generation_id: str,
    round_index: int,
    parent_losses: Iterable[LossProfile],
    child_losses: Iterable[LossProfile],
    weights: ScoringWeights,
    decision: str = "",
    scalar: float | None = None,
    delta_scalar: float | None = None,
) -> Candidate:
    """Place one challenger on every namespace that its runs measured. Pure.

    The function scores the child from the runs of the child. It compares the
    child with the parent for one purpose only: to fill
    :attr:`Candidate.regressions`.

    The function records every namespace in the runs, not only the configured
    axes. The extra values cost nothing. A later change of the axis set can
    thus use the rounds that already ran.

    ``scalar`` and ``delta_scalar`` hold the verdict of the gate for this same
    challenger. The caller supplies them. The function does not calculate them
    again. The caller already holds these values. A second calculation of the
    promotion score here could differ from the true score.
    """
    child_list = list(child_losses)
    child_means = namespace_means(child_list)
    parent_means = namespace_means(list(parent_losses))
    return Candidate(
        generation_id=generation_id,
        round_index=round_index,
        axes=child_means,
        decision=decision,
        regressions=monotonic_regressions(parent_means, child_means, weights),
        scalar=scalar,
        delta_scalar=delta_scalar,
    )


__all__ = [
    "DEFAULT_MAX_FRONTIER_SIZE",
    "INTERNAL_DEV_PARETO_CONFIG",
    "INTERNAL_PARETO_CONFIG",
    "Axis",
    "Candidate",
    "Frontier",
    "FrontierLedger",
    "FrontierUpdate",
    "ParetoConfig",
    "RoundRecord",
    "build_round_record",
    "candidate_from_losses",
    "dominates",
    "monotonic_regressions",
    "namespace_means",
    "resolve_axes",
]
