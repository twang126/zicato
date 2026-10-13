# A contract can require the holdout to improve by a share of the train gain

| Measurement | Delta |
|---|---:|
| Total | +379 |
| Production | +123 |
| Production logic | +67 |

`ScoringWeights.holdout_confirmation`, `holdout_min_gain` and
`holdout_min_gain_ratio` let a contract require the holdout loss to fall by
more than `max(holdout_min_gain, holdout_min_gain_ratio × train gain)`
(issue #574). The gate and the Ladder path pass the train gain, `set_gate`
accepts the fields, and the preflight note reports the ratio. The other lines
are the tests, the re-recorded contract-hash, reindex, mock-evolve and
reader-parity records, and the documentation.
