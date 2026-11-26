# Complete the two-day timing evidence

The raw CSV retains earlier failed attempts. The two balanced primary CSVs contain 20 weather, 20 observed-AQ, 20 HI and
20 KB calls each on 25 and 26 November 2026 (160 total). The observed station values are stale: retain timestamps and never
use them as current pollution information.
Do not count network failures or local-only medians as successful API latency.

From this folder run `python collect_evidence.py --timings --count 20` on at least
two distinct Karachi dates. Requests is required (`python -m pip install requests`).
Open-Meteo AQ is modelled, not ground-station observations.

For the assignment's observed PM2.5/PM10 requirement, supply your own OpenAQ key
as the OPENAQ_API_KEY environment variable before running. The script searches
within 25 km of Karachi for a location exposing both pollutants, times its latest
readings, retains their timestamps and fails honestly if that location is absent.
Inspect station freshness and units; an HTTP success alone does not prove freshness.
No API key is included in the package. Record only your own actual observations.

The CSV is append-only. Do not change timestamps to manufacture a second day.
After successful two-day measurements, update the report's latency calculation
with the pooled observed weather/AQ/HI/KB measurements and actual model timings.
The updated sequential scenario uses the measured two-day pooled median,
0.519909850 seconds. Model TTFT/TPS remain planning assumptions.
