# Submission checklist - read before uploading

**This folder is organized for upload, but the assignment is not empirically complete.**
The remaining requirements are listed explicitly instead of replaced with invented evidence.

| Question / requirement | State | File / remaining work |
| --- | --- | --- |
| Q1 parameters and full token arithmetic | Complete | q1/Question_1_Report.pdf; executed notebook |
| Q1 six closed + six open models, >=4 developers/side | Present | q1/Question_1_Model_Comparison.pdf; research/model_catalog.csv |
| Q1 parameters/context/prices/features/licence research | Partial | Quoted fields and sources included; some fields require recapture |
| Q1 two benchmark scores for every model | Present | Twelve sourced pairs; modes/versions are explicit, not interchangeable |
| Q1 new-paid RPM/TPM, reported TTFT/TPS | Incomplete | Provider account tiers and representative endpoint speed evidence needed |
| Q1 current cache minima/TTL/write policies for every model | Partial | Conservative no-discount policy where unknown; revalidate quoted minima |
| Q1 twelve monthly costs, cache cases, budget flags | Calculated | q1/results/monthly_api_costs.csv; reasoning surcharges remain unknown in two rows |
| Q1 peak load, memory, GPU count | Calculated scenario | Config and metadata verified; concurrency depends on assumed residency |
| Q1 config screenshot | Complete | Captured browser PNG in sources/ and report appendix |
| Q1 real cloud price and step-function break-even plot | Calculated scenario | Advertised price, explicit assumptions and no-crossing proof included |
| Q1 two candidates and architectural recalculation | Scenario only | Real LLM speed and successful mixed tool-delay measurements still required |
| Q1 raw timing log | Included, incomplete requirement | First-day 80-call balanced primary log complete; second date pending; observed values are stale |
| Q1 AI declaration | Complete | Report and root AI_USE_DECLARATION.txt |
| Q2 Section 1 (a)-(d) | Complete | q2/Question_2_Theory.pdf and notebook |
| Q2 full experiment implementation | Included, format/compile checked | q2/Assignment_Beta_Question_no_2_23K-0002.ipynb; all eight run controls enabled |
| Q2 original + four rank-8 rows | GPU execution pending | No real trained scores supplied |
| Q2 ranks 4/8/16/32 and accuracy graph | GPU execution pending | Executable pipeline automatically exports after training |
| Q2 <=4-page report | Four pages, evidence-limited | q2/Question_2_Report.pdf; final measured PDF exported by notebook |
| Q2 five actual wrong examples and observed surprise | Pending real generations | No fabricated customer/model response pairs |
| Q3 (a)-(g), gradient and post-step loss | Complete | q3 PDF, executed notebook, numerical JSON |
| Q4 (1)-(8), metrics/CI/SD/text/perplexity | Complete | q4 PDF, executed notebook, CSV/JSON and graph |
| Q5 two-page paper summary | Complete | q5/Question_5_Two_Page_Summary.pdf; paper reference included |
| Cross-question AI-use declaration | Complete | AI_USE_DECLARATION.txt |

To finish Q2, follow q2/RUN_ON_COLAB.md on a CUDA GPU. To finish the Q1 timing
requirement, follow q1/RUN_TIMINGS.md on two real dates with observed air data.
The local environment has CPU-only PyTorch. Automatic permission review for
direct public-API access initially failed, then succeeded. The supplied OpenAQ
key is stored only in an ignored local directory and is excluded from packaging.
