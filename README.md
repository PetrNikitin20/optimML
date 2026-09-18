# optimML

Reproducible materials for the Russian draft of the SUMMA 2026 conference paper on gradient geometry and pairwise versus pointwise preference optimization.

## What is measured

The experiment uses 2,000 real human-preference pairs sampled as deterministic non-overlapping pages from the 160,800-row `Anthropic/hh-rlhf` training split. Each chosen and rejected dialogue is mapped to a deterministic signed hashing representation with word unigrams and bigrams. The study then:

- measures mini-batch gradient rotation, gradient norm, effective sample size, and saturation over `beta = {0.1, 0.3, 1, 3, 10}`;
- trains pairwise and pointwise logistic preference models with the same representation, optimizer budget, split, and five random seeds;
- reports ranking accuracy, logistic loss, Brier score, expected calibration error, margins, and bootstrap intervals;
- verifies analytical gradients by central finite differences.

This is a controlled real-data diagnostic of preference-loss geometry. It is not a claim of end-to-end fine-tuning of a large language model.

## Reproduce

```powershell
$python = "C:\path\to\python.exe"
& $python -m unittest discover -s tests -v
& $python scripts\run_real_experiment.py --pairs 2000 --dimension 4096 --seed 20260918
```

Only `numpy` and `Pillow` are required. The script downloads rows through the Hugging Face datasets server and caches them under `data/cache`, which is excluded from Git. Dataset provenance and the SHA-256 hash of the canonical JSONL snapshot are written to `results/results.json`.

## Repository layout

- `src/optimml`: data retrieval, deterministic features, losses, metrics, and experiment runner;
- `scripts/run_real_experiment.py`: command-line entry point;
- `tests/test_math.py`: finite-difference and invariant tests;
- `results`: machine-readable run-level and aggregate results plus the generated figure;
- `article`: conference manuscript and the official template copy.

## Data source

Anthropic HH-RLHF: <https://huggingface.co/datasets/Anthropic/hh-rlhf>. Please review the dataset card and applicable terms before redistribution or reuse.

## License

Code in this repository is released under the MIT License. Dataset terms remain with the original dataset provider.
