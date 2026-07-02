# Phase 2J.1 True-PBR Initialization Probe Success Summary

## Goal

Phase 2J.1 tested whether a training model initialized from the local official Hunyuan3D-Paint PBR pipeline directory matches the official inference base UNet numerically.

## Result

Phase 2J.1 succeeded.

Job id: 255899

Key success signals:
- PHASE2J1_TRUEPBR_INIT_PREFLIGHT_OK
- CUBLAS_MATMUL_OK
- Models Loaded.
- recommendation_status: RECOMMENDED
- recommended_candidate_path: model
- mean_of_mean_abs_delta: 0.0
- PHASE2J1_TRUEPBR_INIT_COMPARE_OK
- JOB END: SUCCESS

## Confirmed Strategy

Strategy A is valid.

Training config can use:

/vol/bitbucket/ct1022/hy3dpaint_finetune/caches/hf/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1

as stable_diffusion_config.pretrained_model_name_or_path.

## Interpretation

This confirms that a HunyuanPaint training model can be initialized from the same official PBR weights used by the inference pipeline. The earlier collapsed checkpoints were most likely caused by SD2-based initialization rather than true Hunyuan3D-Paint PBR fine-tuning.

## Next Step

Run a true-PBR 1-step learning-rate-zero checkpoint save smoke to verify the train.py save path before real fine-tuning.
