# Run the real GPU study

1. Upload the notebook in this folder to Google Colab.
2. Choose an NVIDIA GPU runtime and Run all. The eight run switches are enabled.
3. The installation cell may request one kernel restart; follow its displayed instruction.
4. Mount Google Drive when prompted. Use the same output directory to resume.
5. The study evaluates the original model, trains LoRA/QLoRA/DoRA/AdaLoRA at rank 8,
   and trains the training-side selected method at ranks 4,16,32.
6. Download the study output, including tables, graphs, predictions, protocol,
   checkpoints/adapters and the automatically generated Question_2_Report.pdf.
7. Replace this folder's evidence-limited Question_2_Report.pdf with that actual report.

No real CUDA experiment has been completed in this submission folder. No experimental result is supplied in place of actual GPU execution.

The notebook uses a general Glaive function-calling benchmark if the exact four
customer-support tools do not have 2000 eligible examples. The four-page report
discloses that domain limitation. It also discloses in-sample method selection and
post-hoc best-test-row error analysis.

If a very long real error makes the generated report exceed four pages, retain
all raw error examples separately and adjust only the report layout before submission.
