# Full factorial study: execution not complete

The plan contains 288 full-budget runs: two losses, three checkpoints (3B, 8B,
14B), four datasets, four noise probabilities (0, 0.1, 0.2, 0.3), three seeds
(11, 29, 47). `cells.csv` is a coverage/planning artifact, NOT experiment results.
There are zero completed full-budget runs. The one corrected v2 pilot is excluded.

## Actual data preflight on 2026-10-01

The preflight downloaded the real datasets and executed their current parsers
without loading model weights. All four datasets have enough unique prompts
for a 4,000/800 train/holdout partition. The current exact-string partition has
one normalized prompt overlap in Reddit TL;DR and one in HelpSteer2. These
must be removed before confirmatory training. Blank/equal responses must also
be filtered. `colab/full_design_preflight.py` provides a separate proposed v3
normalized partition and preserves the original texts and pilot evidence.
NFKC/case/whitespace normalization is NOT semantic near-duplicate detection.

| Dataset | Valid comparisons | Unique exact prompts | Unique normalized prompts | Normalized overlap in current v2 full split |
|---|---:|---:|---:|---:|
| UltraFeedback | 60,630 | 60,620 | 60,594 | 0 |
| Reddit TL;DR | 92,842 | 14,769 | 14,757 | 1 |
| HelpSteer2 preference | 7,117 | 7,116 | 7,108 | 1 |
| HH-RLHF | 159,878 | 159,060 | 158,970 | 0 |

This is a data audit, not evidence of training improvements or factorial effects.

Both measured reports are exported byte-for-byte and SHA-256 checked:
`preflight_original.json` (3161 bytes,
`fe9a669d33031d17ed55779b5c8310a151f30ba577449865a200544a3409969a`)
and `preflight_corrected.json` (5214 bytes,
`1fa1b91a700cd13ae9284386764144ddfc031f222d522ad8fee3bb026ecbfe00`).
The correction was actually executed on all four real datasets using preparation
source commit `dd0b0d6`, SHA-256
`f149d415fa0c683de2a469141fd34a54e21343fbeb590045369e112c07b9d8c0`.
Each proposed corrected split contains 4,000 training and 800 holdout pairs,
with zero normalized prompt overlaps. Original v2 diagnostic fields remain
unchanged in the report; the new split is under `proposed_v3_split`. Raw pair
exports are currently only in ephemeral Colab storage; they are NOT backed up
by these text-free audit reports. They can be regenerated from pinned data and
source, but must be moved to durable research storage before actual training.

## Gates before expensive execution

1. Register a new confirmatory protocol after the data/context corrections;
   do not silently pool it with v1 or v2 pilots.
2. Pin every dataset, checkpoint, reward judge, tokenizer and executed source.
3. Use normalized prompt grouping and audit near-duplicates. Holdout queries
   stay fixed across loss, model checkpoint, noise and training seed.
4. Harmonize context construction/truncation between training, ranking and
   generation. Measure retained tokens and truncation on each dataset before
   selecting the budget. The current 256-token full default is NOT approved
   as adequate for long Reddit posts and multi-turn HH-RLHF dialogues.
5. Record length-normalized likelihood objectives explicitly; they are not
   ordinary summed-log-probability DPO. Qwen3-8B versus Qwen2.5 checkpoints
   confounds generation/training recipe with parameter count. Interpret the
   checkpoint factor, not an isolated causal effect of model size.
6. The reference is the instruction-tuned initial checkpoint, not independently
   trained task-SFT. Either use that honest terminology or preregister task-SFT
   on a separate split and account for its compute and randomization.
7. Replace unvalidated full-space finite-difference Hessian measurements with
   a validated operator. Check repeated HVPs, symmetry, residuals/convergence,
   dtype and sample sensitivity. Restricted final-layer Hessians cannot stand
   in for full-LoRA Hessians without changing the stated scientific question.
8. Validate the reward judge and increase generation evaluation if needed.
   Thirty-two auto-judged prompts per run are not sufficient to establish
   small response-quality gains. Include confidence ECE/reliability curves,
   fixed-sample KL and independent RewardBench transfer evaluation.
9. Use durable raw-data/result/adapter/checkpoint storage. Colab local stage
   saving does NOT survive runtime destruction. Optimizer state matters for
   exact interrupted-run resumption; saving only an adapter is insufficient.
10. Require all 288 budget-complete records, independently verify individual
    predictions/metrics and report paired contrasts with seed and prompt
    uncertainty. Three seeds limit between-training-run inference. Register
    primary contrasts, interactions and multiple-comparison handling in advance.

## Compute and rental gate

Measured 3B/T4 pilot training took 200.734 seconds for 40 examples. Merely
scaling to 6,400 examples gives 8.922 hours per 3B run and 856.465 hours for
the 96 3B runs, BEFORE evaluation, generation, curvature or longer contexts.
This is a crude extrapolation, NOT an A100/H100 benchmark or a total-project
quote. 8B/14B throughput has not been measured. Longer contexts change costs.

Before booking a large allocation, obtain a user-approved small rental cap,
benchmark all three sizes and both losses on the corrected protocol, test the
14B memory/curvature requirement, then estimate the remaining 288-run budget
including setup, judging, validation, persistent storage and interruptions.
Parallel independent jobs shorten elapsed time but do not remove GPU-hour costs.
No paid resources have been booked or charged by this task.

## Reproduce the planning tools

```bash
python tools/plan_full_factorial.py --output results/a_new_coverage_snapshot
python colab/full_design_preflight.py --source colab/factorial_preference_colab.py --output preflight.json
python colab/full_design_preflight.py --source colab/factorial_preference_colab.py --revision-report preflight.json --output corrected_preflight.json --export-pairs-dir prepared_clean_pairs
```

The preflight needs `datasets` and `huggingface_hub` but no large model weights.
Prepared pair files contain dataset text; keep them in research storage, outside
public Git, and respect upstream dataset terms. This tool does not train models.
`tools/plan_full_factorial.py` needs only the standard Python library and refuses
to overwrite an existing coverage directory. An independently verified full
experiment and a revised journal manuscript remain outstanding.
