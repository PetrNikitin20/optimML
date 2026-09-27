# Colab factorial experiment

`factorial_preference_study.ipynb` runs one cell of the real-data factorial design per Colab session and writes immutable JSON results to `MyDrive/optimML_factorial`.

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
4. Assign one unfinished `RUN_INDEX` and run all remaining cells.
5. Repeat with another index. Existing completed JSON files are not overwritten.
6. Run the aggregation cell after all required indices are complete.

The exact 8B grid uses `Qwen/Qwen3-8B`, while the 3B and 14B cells use Qwen2.5 checkpoints. Therefore inferential analysis must treat checkpoint as the factor and must not attribute every difference solely to parameter count. A same-generation sensitivity grid can instead use Qwen2.5 3B, 7B and 14B.

No table in the article should be populated until the corresponding JSON files exist and pass consistency checks.
