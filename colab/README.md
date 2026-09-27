# Colab factorial experiment

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
