# A contract can replicate the evolve loop's holdout confirmation

| Measurement | Delta |
|---|---:|
| Total | +259 |
| Production | +56 |
| Production logic | +17 |

`ScoringWeights.replicate_holdout` makes the evolve loop's holdout
confirmation run each holdout entry at the crowning duel's replicate count,
the holdout record saves the count, and the cost estimate counts holdout
replicates only when the setting is on (issue #573). The other lines are the
tests, the re-recorded contract-hash, reindex, mock-evolve and reader-parity
records, and the documentation.
