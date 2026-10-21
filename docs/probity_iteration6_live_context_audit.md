# Live-screen raw context and receipt audit

This supplement covers the new iteration6 display and prompt screens. The
[historical raw-context audit](probity_iteration6_raw_context_audit.md) covers
only the iteration5-derived 166-question A/B corpus.

## Accepted-call coverage

| Screen | Accepted private plans | Accepted public actions | Accepted Haiku judgments | Native judgments |
| --- | ---: | ---: | ---: | ---: |
| Display, 24 families × 7 | Reused historical plans | 672 | 168 | 168 |
| Prompt, 8 families × 4 | 32 | 192 | 32 | 32 |

The display screen copies the exact first two historical no-feedback turns in
each family, then makes four new public calls in every branch. The prompt screen
generates all six turns. Feedback-on/off prompt pairs share a private plan for
each speaker, but their first two public turns are separate generations.

Every accepted API receipt contains the canonical request, request hash, raw
provider response body, response hash, usage, finish reason, parsed result and
accepted state transition. Display action providers were DeepInfra (440),
Alibaba (172), and SiliconFlow (60); all 168 display Haiku judgments used Azure.
Prompt action calls used DeepInfra (192); plan calls used DeepInfra (19),
Alibaba (7), and SiliconFlow (6); all 32 Haiku judgments used Azure.

All 32 accepted prompt plan responses contain provider-returned reasoning. None
of the 192 prompt public actions or 672 display public actions contains returned
reasoning because public reasoning was disabled. Public prompts receive the
parsed structured plan, not the plan call's returned reasoning. Hidden provider
state, templates, or reasoning that was never returned cannot be recovered.

## Canonical and transported requests

For all display calls, the canonical receipt request is also the submitted
body. Prompt recovery pinned the route without changing messages, response
schema, sampling, or model alias. Consequently, all 192 prompt actions and
three prompt plan receipts contain a `transport_request`: this is the actual
on-wire body, while `request` remains the immutable scientific request used by
the protocol identity. Both are retained and hashed. The other 29 prompt plans
and all judge requests were submitted from their canonical body.

Failed and superseded transport attempts remain under the prompt recovery
directories. They are never counted as additional trajectories or silently
substituted for an accepted response.

## Model-visible timeline

Every advocate call receives the authentic story, question/options, verified
authentic answer, assigned public answer, parsed private preparation, and the
complete public debate so far. The verified answer and private instructions are
in the advocate's private system message; the final judge never receives them.

| Harness | Turn 1 | Turn 2 | Turns 3–6 |
| --- | --- | --- | --- |
| Historical B | Feedback availability announced; empty packet | Turn-1 score visible | All completed earlier scores visible |
| Display | Copied historical A turn | Copied historical A turn | Configured packet inside the private user message |
| Prompt feedback off | Fresh generation, no feedback instruction | Fresh generation, no feedback instruction | No packet |
| Prompt feedback on | Fresh generation, no feedback instruction | Fresh generation, no feedback instruction | Feedback instruction and configured bundle |

Where a policy supplies scores, both speakers see the completed scores for both
speakers, the corresponding running means, and/or the recommendation selected
by that policy. A current or future score is never present. Every generated
turn is scored locally before the next action, including conditions where the
score remains hidden.

## Archive layout

- Display canonical records:
  `runs/iteration6/live/display24_v1/{requests,raw_responses,states,score_requests,score_results}`.
- Prompt canonical records:
  `runs/iteration6/live/prompt8_v1/{jobs,requests,raw_responses,states,score_requests,score_results}`.
- Display sample browser: `runs/iteration6/live/inspection_v1/index.html`, three
  families and all seven policies.
- Prompt sample browser: `runs/iteration6/live/prompt_inspection_v2/index.html`,
  two families and all four cells.

The sample browsers are navigation aids, not full exports. The complete raw
trees are gitignored. The prompt `export_v1` directory is a judge/analysis view
of arguments and scores and omits complete request envelopes and planning
reasoning. A checksummed private handoff archive is required alongside the
public aggregate reports for independent raw inspection.

The current host-side bundle is
`runs/iteration6/handoff/iteration6_private_evidence_v2.tar.gz`; its adjacent
SHA-256 file records
`cf023966d1f87f62ca834a95f5beab45bc2acfd5337a694ea3070812ae20974d`.
