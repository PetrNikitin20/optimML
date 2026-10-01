# optimML

**Research status, 2026-10-01:** a protocol audit identified problems in the
legacy Colab instrumentation (data parsing, initialization seeds, split leakage,
calibration and stochastic curvature probes). These pilots are preserved for
traceability, but must not be used as confirmatory publication evidence.
Corrected v2 experiments are kept separate. The existing SWSYS manuscript is a
historical draft, not submission-ready; its empirical section must be rebuilt
from validated v2 measurements. See [PROTOCOL_AUDIT.md](results/PROTOCOL_AUDIT.md).

The [full 288-run coverage plan](results/factorial_full_plan_20261001/README.md)
and real-data preflight are now available. Full-budget coverage is still 0/288.
The data audit found normalized train/holdout prompt overlaps in Reddit TL;DR
and HelpSteer2; corrected data preparation is separate from pilot evidence.
Expensive execution requires the listed scientific and durable-compute gates.

Reproducible materials for the Russian and English drafts of the SUMMA 2026 conference paper on gradient geometry and pairwise versus pointwise preference optimization.

## What is measured

The experiment uses 2,000 real human-preference pairs sampled as deterministic non-overlapping pages from the 160,800-row `Anthropic/hh-rlhf` training split. Each chosen and rejected dialogue is mapped to a deterministic signed hashing representation with word unigrams and bigrams. The study then:

- measures mini-batch gradient rotation, gradient norm, effective sample size, and saturation over `beta = {0.1, 0.3, 1, 3, 10}`;
- trains pairwise and pointwise logistic preference models with the same representation, optimizer budget, split, and five random seeds;
- reports ranking accuracy, logistic loss, Brier score, expected calibration error, margins, and bootstrap intervals;
- verifies analytical gradients by central finite differences.

The second experiment fine-tunes a real pretrained Transformer reward model. It compares a frozen `prajjwal1/bert-tiny` encoder with a trainable reward head against LoRA rank 8 on attention query/value projections. Both methods use the same 2,000 pairs, deterministic 1,400/300/300 split, three training seeds, and validation-only checkpoint selection.

The external pilot adds `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with a scalar sequence-classification head and LoRA rank 4 on `q_proj`/`v_proj`. It trains 565,248 of 1,035,079,680 parameters on 24 HH-RLHF pairs and evaluates before/after adaptation on 48 held-out HH-RLHF pairs and a deterministic 92-example RewardBench sample (four examples from each of 23 subsets). This is a CPU-feasible transfer diagnostic, **not** an official RewardBench leaderboard submission or a claim of statistical improvement.

The Colab factorial pilot now contains matched three-seed comparisons of pairwise and pointwise LoRA objectives for `Qwen/Qwen2.5-3B-Instruct` on real UltraFeedback preference pairs at both zero and 0.1 injected label noise. The 0.1-noise pointwise series is complete at indices 147-149 (seeds 11, 29, and 47), with data hashes matched to pairwise indices 3-5 within seed. Each run uses 256 training pairs, 64 evaluation pairs, 10 optimizer steps, and eight deterministic generation prompts. The repository records raw run JSON, separate summaries, and paired pointwise-minus-pairwise contrasts. The pilot is explicitly non-inferential; its purpose is to validate the protocol and estimate compute before expanding across model size, dataset, and noise factors.

The historical Russian Word manuscript for the journal *Programmnye produkty i sistemy* is available at `article/SWSYS_RU_pairwise_pointwise_real_data.docx`. Its legacy pilot tables require revision after the protocol audit. Do not submit it or treat a rebuild with the current legacy-table generator as a validated v2 article.

## Reproduce

```powershell
$python = "C:\path\to\python.exe"
& $python -m unittest discover -s tests -v
& $python scripts\run_real_experiment.py --pairs 2000 --dimension 4096 --seed 20260918
& $python scripts\run_lora_experiment.py --pairs 2000 --training-seeds 11 29 47 --epochs 4 --batch-size 16
& $python scripts\run_external_benchmark.py
& $python scripts\recompute_external_statistics.py
```

The linear experiment requires `numpy` and `Pillow`; the LoRA experiment additionally uses pinned `torch`, `transformers`, `peft`, `accelerate`, and `safetensors` versions. The scripts download rows through the Hugging Face datasets server and cache them under `data/cache`, which is excluded from Git. Dataset provenance and SHA-256 hashes are written to the machine-readable result files.

## Repository layout

- `src/optimml`: data retrieval, deterministic features, losses, metrics, and experiment runner;
- `scripts/run_real_experiment.py`: command-line entry point;
- `scripts/run_lora_experiment.py`: Transformer reward-model and LoRA experiment;
- `scripts/run_external_benchmark.py`: registered BERT-tiny evaluation and TinyLlama 1.1B LoRA pilot on RewardBench;
- `scripts/recompute_external_statistics.py`: bootstrap intervals and paired exact sign test from row-level predictions;
- `colab/factorial_preference_study.ipynb`: sharded Google Colab experiment for the 288-cell real-data factorial design;
- `tests/test_math.py`: finite-difference and invariant tests;
- `results`: machine-readable run-level and aggregate results plus the generated figure;
- `article`: conference manuscript and the official template copy.
- `LITERATURE.md`: annotated modern primary literature on LoRA and preference optimization.
- `APPLICATIONS.md`: sector-specific deployment scenarios, data requirements, and safety checks.

## Data source

Anthropic HH-RLHF: <https://huggingface.co/datasets/Anthropic/hh-rlhf>. Please review the dataset card and applicable terms before redistribution or reuse.

## License

Code in this repository is released under the MIT License. Dataset terms remain with the original dataset provider.
