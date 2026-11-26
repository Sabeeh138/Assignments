# Q1 completion - 26 November 2026

Completed all report sections, the single model table and field-level source
register, executed notebook, cost/capacity graphs, actual config screenshot,
and 160 balanced real timing records across 25/26 November.

Replaced two reasoning-cost lower-bound rows with non-reasoning Amazon Nova Lite
and Qwen3-30B-A3B-Instruct-2507. There are four model developers per side.
Selected available serving routes are explicit; older native deployment status
is not confused with a currently listed Azure/router route.

Reported speeds now drive latency estimates with a context-growth sensitivity.
The API shortlist is Gemini 2.5 Flash-Lite / Amazon Nova Lite. Changed workflow
estimates 2.374691 / 3.728891 seconds, $27.686286 / $16.611772 per month.
Self-hosting has no finite break-even under the stated 4s/28-slot assumptions.

The uncertainty paragraph discloses actual new-paid TPM entitlement and routed
DeepSeek cache granularity/lifetime as unavailable. These are not fabricated.
The real stale observations are withheld from current summaries by the tested
freshness guard. No paid inference, language pilot or GPU load test is claimed.

Q2 still requires actual GPU training and evaluation.
