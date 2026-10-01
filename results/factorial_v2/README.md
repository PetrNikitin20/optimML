# Corrected protocol v2: measured pilot, not factorial conclusions

Run 14 completed on 2026-10-01: Qwen2.5-3B-Instruct, Reddit TL;DR,
pairwise length-normalized reference-ratio loss, noise 0, seed 47, T4.
Training used 10 optimizer steps, 40 examples from a 256-pair pool;
evaluation used 64 disjoint prompts and generation judging used eight prompts.

| Quantity | Observed value |
|---|---:|
| Policy likelihood ranking | 41/64 = 0.640625 |
| Initial checkpoint likelihood ranking | 41/64 = 0.640625 |
| Reference-ratio ranking | 29/64 = 0.453125 |
| Brier | 0.25458319384389994 |
| Probability-bin ECE (15 bins) | 0.03224266622529457 |
| Confidence-bin ECE (15 bins) | 0.06148758997882755 |
| Mean common shift c | 0.6988731408491731 |
| Mean margin d | -0.16962864063680172 |
| KL(policy to initial checkpoint) | 1.3437192715971569 |
| Reward-model win rate | 6/8 = 0.75 |
| Mean response tokens, policy / reference | 101.375 / 83.0 |
| Training seconds / peak allocated GPU GB | 200.73396750100005 / 3.9495673179626465 |

The initial checkpoint is instruction-tuned; no independent task-specific SFT
stage was performed. Historical JSON field names containing `sft` refer to the
adapter-disabled initial checkpoint, not a newly trained SFT model.

Accuracy, Brier, both ECE variants, c/d, judge wins/rewards and lengths were
independently recomputed from individual exported records. Raw text, original
Parquet, model adapter and source snapshot were backed up locally outside Colab.
Numeric row records and judge scores are published here. Dataset text and
generated responses are not redistributed in Git; respect upstream data terms.
The selected data SHA-256 was verified against the local raw-pair export.

## Curvature validation

The full-LoRA finite-difference HVP failed an epsilon sensitivity check.
HVP norms were 44.4546, 10.9888 and 3.3012 for epsilon 0.001, 0.003 and 0.01;
adjacent cosine similarities were 0.1718 and 0.000489. Repeated gradients at
identical weights were identical, and weights were restored exactly.
Do not interpret its power-iteration value (-38.4557) as a validated eigenvalue.

An additional double-backward autograd check, using math SDPA and disabled
dropout, operated ONLY on 32,768 parameters of the final MLP LoRA-B projection,
using ONE real held-out pair. Eight power iterations gave a dominant-magnitude
Rayleigh quotient -0.0466983318 and relative residual 0.00218472. Repeated HVPs
matched; the symmetry relative error was 2.22697e-6. This is useful evidence of
operator consistency in the named restricted subspace, NOT the full model's
Hessian or the largest algebraic eigenvalue. It is not directly comparable to
the full-space finite-difference measurement.

The conditional-preference Fisher estimates use two pairs and remain pilot
diagnostics. Neither KL nor full-space Fisher was independently rerun by the
CPU row-metric verifier.

## Limits and next work

One seed cannot support between-seed uncertainty or comparisons between losses,
noise levels, datasets or model sizes. Eight auto-judged generations cannot
establish superior response quality. Training/evaluation truncation is severe
at 192 tokens, and generation truncation must be harmonized with pair encoding
before a confirmatory long-context experiment. Audit near-duplicates, preregister
the longer-context protocol and training budget, add matched seeds and losses,
and evaluate transfer independently on RewardBench. The registered 288-run design
has NOT been executed. The legacy SWSYS manuscript still requires revision.

Run `python -m unittest discover -s tests -v` to recheck public numeric records.
`tools/verify_protocol_v2.py` validates a complete local raw-data export;
`tools/collect_protocol_v2.py` collects its verified, text-free public artifacts.
