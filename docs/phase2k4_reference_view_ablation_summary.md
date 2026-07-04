# Phase 2K.4 Reference-View Ablation Summary

## Result

Phase 2K.4 showed that front / texture-informative input views 004 and 005 substantially improve rendered-view fidelity compared with the previous input protocol.

## Key Aggregate Results

Previous baseline:
- base vs reference MAE: 5.968
- fine vs reference MAE: 5.958
- base front MAE: 15.399
- fine front MAE: 15.480

Input 004:
- base vs reference MAE: 3.976
- fine vs reference MAE: 4.074
- base front MAE: 8.869
- fine front MAE: 9.101

Input 005:
- base vs reference MAE: 3.999
- fine vs reference MAE: 4.071
- base front MAE: 8.768
- fine front MAE: 9.016

## Interpretation

The main improvement comes from reference input view selection, not from the current true-PBR 200-step fine-tuned checkpoint.

The fine-tuned checkpoint remains stable but does not consistently outperform the base model under the corrected front-view input protocol.

A remaining limitation is backside / unseen-surface texture leakage when a front-only input image is used.

## Next Step

Move to input-view-aware Data v2 planning. Future evaluations should use selected informative input views and report front fidelity, all-view consistency, and backside leakage separately.
