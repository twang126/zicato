# A contract can score the scorer's own metrics

| Measurement | Delta |
|---|---:|
| Total | +287 |
| Production | +126 |
| Production logic | +66 |

`ScoringWeights.scorer_metric_namespaces` lists the namespaces of scorer
metrics that the scalar scores, and the loader refuses the reserved
namespaces and a namespace with no weight (issue #575). The proposer's metric
priorities and loss summary see a listed namespace by its prefix only, and a
listed namespace is meaned over the runs that report it. The
other lines are the tests, the re-recorded
contract-hash, reindex, mock-evolve and reader-parity records, and the
documentation.
