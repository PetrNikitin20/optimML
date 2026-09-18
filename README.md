# optimML

Reproducible materials for the Russian draft of the SUMMA 2026 conference paper on gradient geometry and pairwise versus pointwise preference optimization.

## What is measured

The experiment uses 2,000 real human-preference pairs sampled as deterministic non-overlapping pages from the 160,800-row `Anthropic/hh-rlhf` training split. Each chosen and rejected dialogue is mapped to a deterministic signed hashing representation with word unigrams and bigrams. The study then:

- measures mini-batch gradient rotation, gradient norm, effective sample size, and saturation over `beta = {0.1, 0.3, 1, 3, 10}`;
- trains pairwise and pointwise logistic preference models with the same representation, optimizer budget, split, and five random seeds;
- reports ranking accuracy, logistic loss, Brier score, expected calibration error, margins, and bootstrap intervals;
- verifies analytical gradients by central finite differences.

The second experiment fine-tunes a real pretrained Transformer reward model. It compares a frozen `prajjwal1/bert-tiny` encoder with a trainable reward head against LoRA rank 8 on attention query/value projections. Both methods use the same 2,000 pairs, deterministic 1,400/300/300 split, three training seeds, and validation-only checkpoint selection. The LoRA experiment is CPU-reproducible but is not a claim about full-size generative LLMs.

## Reproduce

```powershell
$python = "C:\path\to\python.exe"
& $python -m unittest discover -s tests -v
& $python scripts\run_real_experiment.py --pairs 2000 --dimension 4096 --seed 20260918
& $python scripts\run_lora_experiment.py --pairs 2000 --training-seeds 11 29 47 --epochs 4 --batch-size 16
```

The linear experiment requires `numpy` and `Pillow`; the LoRA experiment additionally uses pinned `torch`, `transformers`, `peft`, `accelerate`, and `safetensors` versions. The scripts download rows through the Hugging Face datasets server and cache them under `data/cache`, which is excluded from Git. Dataset provenance and SHA-256 hashes are written to the machine-readable result files.

## Repository layout

- `src/optimml`: data retrieval, deterministic features, losses, metrics, and experiment runner;
- `scripts/run_real_experiment.py`: command-line entry point;
- `scripts/run_lora_experiment.py`: Transformer reward-model and LoRA experiment;
- `tests/test_math.py`: finite-difference and invariant tests;
- `results`: machine-readable run-level and aggregate results plus the generated figure;
- `article`: conference manuscript and the official template copy.
- `LITERATURE.md`: annotated modern primary literature on LoRA and preference optimization.

## Data source

Anthropic HH-RLHF: <https://huggingface.co/datasets/Anthropic/hh-rlhf>. Please review the dataset card and applicable terms before redistribution or reuse.

## License

Code in this repository is released under the MIT License. Dataset terms remain with the original dataset provider.
