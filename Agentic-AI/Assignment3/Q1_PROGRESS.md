# Q1 progress - 25 November 2026

Completed in this work session:

- Recorded 20 successful observed-air API calls and 40 successful weather calls.
- Kept every attempt in the append-only raw CSV, including initial failures.
- Exported a balanced primary CSV with the latest 20 successful measurements for
  each of weather, OpenAQ, heat-index computation and knowledge-base lookup.
- Calculated the primary 80-record median: 0.713524150 seconds.
- Captured and inspected the actual Hugging Face configuration screenshot.
- Added two sourced public benchmark scores for each of the twelve models,
  retaining the benchmark versions and reasoning-mode qualifications.
- Verified additional caching rules and documented published provider limits
  separately from unverified new-account entitlements.
- Updated calculations, plots, notebook, report and model comparison.
- Confirmed the API key is Git-ignored and absent from all submission files/ZIP.

Important data finding: the timed OpenAQ station returned observations dated
28 February 2024. Of 54 nearby locations returned within 25 km, four expose both
PM2.5 and PM10 and their latest location timestamps are in 2024. Fresher PM2.5-only
stations exist. The API calls are real timing measurements, but the old values
must not be presented as current Karachi pollution information.

Still outstanding:

1. A second timing collection on an actual different Karachi date, such as
   26 November 2026. Do not change timestamps or pretend this has happened.
2. Remaining current new-paid-account quota, cache-policy and model-speed fields
   marked NR, plus reasoning-token averages needed to replace two cost lower bounds.
3. A suitable current observed PM10 source or a clearly documented coverage
   limitation; modelled air-quality forecasts are not ground observations.

The local OpenAQ key is already configured for the collector. To continue with
Codex on the next date, ask it to perform the second-day Q1 measurements and
refresh the report. No Q2 GPU is needed for these API measurements.
