# Colab factorial experiment

**Protocol v2 (2026-10-01):** corrected pilots are now stored under
`optimML_factorial/protocol_v2`. Earlier measurements are legacy instrumentation
checks, not publication evidence. See [the protocol audit](../results/PROTOCOL_AUDIT.md).
Each stage has an atomic progress JSON; final JSON is saved automatically after
judging. Local output remains ephemeral and must be exported before disconnecting.
`OPTIMML_LOCAL_OUTPUT=1` bypasses Drive mounting; `OPTIMML_RUN_INDEX` selects a run
for scripted execution. Existing complete v2 files cause a hard stop, not overwrite.

`factorial_preference_study.ipynb` runs one cell of the real-data factorial design per Colab session and writes immutable JSON results to `MyDrive/optimML_factorial`.

The notebook defaults to the `pilot` profile so an accidental click does not launch a multi-hour full run. Three profiles are available:

- `smoke`: 2 optimizer steps, 64/16 train/evaluation pairs, for graph and memory validation;
- `pilot`: 10 steps, 256/64 pairs, for measured throughput and ETA;
- `full`: the registered 400-step, 4000/800-pair factorial cell.

Pilot and smoke results are stored separately and are never included in the full-study aggregate.

The complete default design contains 288 runs:

- two losses: pairwise and pointwise;
- three model classes: 3B, 8B and 14B;
- four datasets: UltraFeedback, Reddit TL;DR, HelpSteer2 and HH-RLHF;
- four training-label noise levels: 0.0, 0.1, 0.2 and 0.3;
- three seeds: 11, 29 and 47.

The notebook measures likelihood-ranking accuracy, generation win rate against the SFT checkpoint, Brier score, 15-bin ECE, common shift `c`, margin `d`, categorical KL to SFT, response length, empirical-Fisher and Hessian spectral estimates, elapsed time, throughput and peak GPU memory.

## Execution

1. Open the notebook in an authenticated Google Colab session.
2. Select a GPU runtime. A100 40 GB or larger is recommended for the 14B cells.
3. Run the installation, configuration and manifest cells.
4. Assign one unfinished `RUN_INDEX`, choose `smoke`, `pilot`, or `full`, and run the remaining cells.
5. Repeat with another index. Existing completed JSON files are not overwritten.
6. Run the aggregation cell after all required indices are complete.

The exact 8B grid uses `Qwen/Qwen3-8B`, while the 3B and 14B cells use Qwen2.5 checkpoints. Therefore inferential analysis must treat checkpoint as the factor and must not attribute every difference solely to parameter count. A same-generation sensitivity grid can instead use Qwen2.5 3B, 7B and 14B.

No table in the article should be populated until the corresponding JSON files exist and pass consistency checks.

Before training, the notebook performs a nonzero finite-gradient preflight. Training logs every optimizer step with a measured ETA and writes restartable LoRA/optimizer/RNG checkpoints to Drive. A resumed run therefore restarts at an optimizer boundary instead of silently repeating the entire cell.

## Verified smoke run

`results/factorial_smoke/000_pairwise_3B_ultrafeedback_noise0.0_seed11.json` is a real-data end-to-end validation run completed on a Tesla T4. It uses Qwen2.5-3B-Instruct, pairwise loss, UltraFeedback, zero injected label noise and seed 11. The run verifies dataset loading, QLoRA training, checkpoint resume, ranking and calibration metrics, KL, generation judging, finite-difference Hessian-vector products, empirical Fisher estimates and cost accounting.

This artifact is deliberately labelled `smoke`: it has two optimizer steps, 16 evaluation pairs and two generation prompts. It is evidence that the pipeline executes and persists coherent measurements, not evidence for a comparison between losses, model sizes, datasets or noise levels. Inferential claims require the preregistered `full` results from all relevant seeds.

## Verified three-seed pilot series

`results/factorial_pilot` contains measured pilots for seeds 11, 29 and 47 on a Tesla T4. Every run uses 256 training pairs, 64 evaluation pairs, ten optimizer steps and eight generation prompts. For the pairwise, zero-noise cell, training took 193.12 +/- 4.79 seconds, throughput was 0.207 +/- 0.005 examples per second, likelihood-ranking accuracy was 0.573 +/- 0.039, and generation win rate against SFT was 0.458 +/- 0.144 (mean +/- sample SD).

The zero-noise pairwise and pointwise series are complete. Their summaries are stored in `pairwise_3B_ultrafeedback_noise0.0_summary.json` and `pointwise_3B_ultrafeedback_noise0.0_summary.json`; the seed-matched contrast is in `paired_loss_contrast_3B_ultrafeedback_noise0.0_summary.json`. Pointwise minus pairwise ranking accuracy was -0.0052 +/- 0.0090, and generation win rate was -0.0417 +/- 0.1443. These are descriptive pilot effects based on only three seeds, not inferential evidence that the objectives are equivalent or different.

The pairwise noise-0.1 series at indices 3, 4 and 5 is also complete. Mean ranking accuracy was 0.5833 +/- 0.0361, Brier score was 0.2496 +/- 0.0043, and generation win rate was 0.4167 +/- 0.4018. Its large between-seed generation variance and unstable Hessian estimate rule out a scientific noise-effect claim at pilot scale. The next registered runs are the matched pointwise noise-0.1 series at indices 147, 148 and 149. Within each seed, its data hash must equal the corresponding pairwise run at indices 3, 4 and 5 before a paired loss contrast is computed.

All currently complete pilot cells are instrumentation checks. The finite-difference Hessian estimates are extremely unstable and must remain numerical diagnostics until epsilon and power-iteration sensitivity analyses are complete.
