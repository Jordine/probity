# Preparation recovery and accounting

**Complete with verified release evidence.** Snapshot: 2026-10-18T18:20:33.971103+00:00.

568 NEW private preparation jobs; excludes the 96 old pilot preparation jobs. Reused receipt copies are not attempts, samples or new charges.

| Model | New jobs | Accepted | Awaiting accepted result | First attempt accepted | Accepted after retry |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen/qwen3.5-27b | 284 | 284 | 0 | 276 | 8 |
| qwen/qwen3.8-27b | 284 | 284 | 0 | 284 | 0 |

8 jobs had a failed first receipted attempt; 8 subsequently obtained an accepted result. There are 10 failed receipted attempts in total, including 9 unanswered attempts and 1 quarantined over-limit responses.

578 POST attempts have receipts; 0 further POST-wrapper entries have no receipt yet. Of the 0 jobs without an accepted result, 0 have no recorded POST attempt and 0 have a recorded attempt. These are snapshot coverage gaps, not declared terminal failures. Pending requests and not-yet-started jobs must not be counted as failed completed observations.

Receipted POST attempts plus separately reported pending POST-wrapper entries. Transport failure does not prove provider receipt. Accepted counts are unique canonical problem/model/speaker jobs.

## Financial accounting

USD amounts below describe different scopes and overlap. Provider-reported charges are distinguished from estimates and retained unknown-charge allowances.

| Scope | Amount (USD) | Interpretation |
| --- | ---: | --- |
| Preparation: provider-reported charges | $6.299243 | All fresh receipted preparation attempts, including active batches and rejected responses. |
| Preparation: catalog estimates | $0.000000 | Estimated charges where no provider cost was returned. |
| Preparation: original unknown-call allowances | $0.458021 | Original receipt allowances for unanswered calls. |
| Preparation: additional unknown-liability allowances | $5.750784 | Extra liability retained even after a successful retry. |
| Preparation: settled ledger accounting | $12.508048 | Settled batch accounting plus retained liability top-ups; includes estimates, not only measured charges. |
| Preparation: active outer envelope | $0.000000 | Reservation containing active-batch charges and inner guard liabilities. |
| Campaign: settled ledger accounting | $33.504400 | All campaign stages already settled, including preparation allowances. |
| Campaign: active outer envelopes | $0.000000 | All active stage reservations, including the preparation envelope. |
| Campaign: total committed | $33.504400 | Campaign settled accounting + campaign active envelopes. |
| Campaign: ceiling | $45.000000 | Authorized campaign accounting limit. |

**Do not add these rows together.** Preparation is part of the campaign. Active-batch recorded charges and per-request inner guards are already inside the active outer envelope. The total committed is therefore campaign settled accounting plus campaign active envelopes, not that total plus recorded active charges or guard values.

9 unresolved historical liability entries remain. A settled ledger status means the allowance has been recorded; it does not establish the provider’s actual charge. Successful retries do not resolve earlier unanswered charges. Raw evidence and per-job identities remain private.

## Final native judging and rental release

The final native judge produced 1328 decisions: 192 exact cache hits and 1136 fresh decisions. Rental `56588961` is independently verified absent and its campaign-ledger accounting is settled at $0.352737.
