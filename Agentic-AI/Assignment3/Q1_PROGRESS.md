# Q1 progress - 26 November 2026

- Collected genuine API timings on 25 and 26 November 2026, Karachi dates.
- Each date has 20 successful measurements for each of weather, observed air,
  heat index and knowledge-base lookup: 160 balanced primary records total.
- Pooled median: 0.519909850 seconds. All attempts retained in the raw CSV.
- Recomputed latency scenarios, session-residency assumptions and cost curves.
- Updated the executed calculation notebook, report and submission checklist.

The OpenAQ responses contain historical station observations. These are valid
HTTP response timings, but the values must not be used as current Karachi air
quality. The nearby two-pollutant coverage audit and actual response timestamps
are preserved. A suitable current observed PM10 source remains a limitation.

Still outstanding: Q1 new-paid quota/cache/speed fields marked NR and reasoning
token averages for two cost lower bounds; Q2 actual GPU training and evaluation.
No model performance or training evidence has been invented. The API key is
kept outside the submission and checked during packaging.
