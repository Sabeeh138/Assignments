# Run Q2 on Kaggle

1. Import the notebook in this folder into Kaggle.
2. Enable Internet and select an NVIDIA GPU in Settings / Session options.
   Prefer T4 if available. The notebook uses only GPU 0, even on T4 x2.
3. Run the installation cell first. If it requests a session restart, restart
   the Python session once, then rerun from that cell. Python 3.10-3.13 is supported.
   Python 3.13 selects PyTorch 2.8/CUDA 12.6 and NumPy 2.2.6; older supported
   Python versions retain PyTorch 2.6/CUDA 12.4 and NumPy 2.0.2.
4. Run all remaining cells in order. All eight run controls are enabled.
   No Google Drive mount is needed. Results go to
   /kaggle/working/Assignment_Beta_Q2.
5. Complete the baseline, four rank-8 methods, selected-method ranks 4/16/32,
   all evaluation/graph/error cells and the measured PDF export.
6. The last cell checks for the completed experiment index and report, creates
   /kaggle/working/Assignment_Beta_Q2_results.zip and displays a download link.
7. Download that ZIP and the executed notebook with outputs before ending the
   session. Preserve a Kaggle version with outputs as well.
8. Upload the executed notebook and generated report/tables/graphs/error evidence
   into GitHub's q2 folder, replacing the evidence-limited draft report.

Seven training runs plus original-model evaluation are required. GPU availability
and session limits are platform-dependent. Do not treat an interrupted run as
complete or change model/data/sequence settings between comparisons. Retain the
output checkpoints if you need to resume with the same runtime and protocol.

The benchmark uses general Glaive function calling when the exact four support
tools do not supply 2000 eligible examples. This domain limitation is disclosed.
No actual GPU experiment has been completed in the supplied delivery notebook.
