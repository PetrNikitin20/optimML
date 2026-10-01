# Protocol audit, 2026-10-01

Earlier smoke/pilot runs are real computations but use legacy v1 instrumentation.
Do not use them as confirmatory publication evidence or pool them with v2.

Corrections in protocol v2:

- Reddit comparison summaries are parsed from `text`, without policy/note metadata.
- The random seed is set before LoRA initialization.
- A fixed split seed (20261001), one pair per exact prompt, and disjoint prompts
  keep train/holdout data constant across training seeds, losses and checkpoints.
  Near-duplicate detection remains a prerequisite for a full publication study.
- Response truncation reserves at least half the token budget for prompt context.
  The 192-token pilot remains a resource check, not a realistic long-context study.
- Raw policy likelihood ranking and reference-ratio ranking are separate metrics.
  Sequence log probabilities remain length-normalized; this is not standard
  summed-log-probability DPO and must be stated explicitly in the paper.
- Binary probability ECE uses balanced deterministic response orientations.
- Curvature probes disable dropout and restore parameters by exact copy.
  Fisher is the conditional Bernoulli preference Fisher, not objective-gradient
  outer products. Hessian power iteration estimates the dominant-magnitude
  Rayleigh quotient, not necessarily the largest algebraic eigenvalue. Two
  iterations and finite differences are diagnostics only; epsilon/convergence
  sensitivity is still required.
- Quantized model parameter counts use `num_parameters()` rather than packed
  storage tensor sizes.
- JSON progress is atomically saved after each stage; final results are saved
  automatically after generation judging, with raw-pair, row and generation
  artifacts hashed. Existing completed results cannot be overwritten.

Local Colab output is ephemeral. Stage saving alone does not survive destruction
of the runtime. Export raw artifacts and adapters or mount persistent storage.
The final JSON is also printed for immediate recovery, but console transcription
is not a replacement for raw artifacts.

Run 14 legacy seed 47 was observed through all metric stages, but its runtime was
lost before final saving. Its observation record is in `factorial_pilot/observations`
and is excluded from completed-run aggregation.

Remaining publication requirements: full training budgets, matched three-seed
comparisons, dataset/model revision pinning, independent RewardBench evaluation,
validated generation judging, sufficient holdout size and uncertainty estimates,
near-duplicate audit, curvature sensitivity, and all requested factorial cells
or an explicitly preregistered smaller research question. No factorial effects
can be claimed from the current pilot coverage.
