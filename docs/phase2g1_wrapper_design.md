# Phase 2G.1 Project-Local Inference Wrapper Design

Phase 2G.1 prepares a project-local wrapper for later base-vs-fine-tuned
Hunyuan3D-Paint inference. The wrapper lives in `hy3dpaint_finetune` and does
not modify the official Hunyuan source tree.

## Official Interface

`demo.py` imports these symbols from `textureGenPipeline.py`:

- `Hunyuan3DPaintConfig`
- `Hunyuan3DPaintPipeline`

The quick inference pattern in `demo.py` is:

```python
max_num_view = 6
resolution = 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
paint_pipeline = Hunyuan3DPaintPipeline(conf)
output_mesh_path = paint_pipeline(
    mesh_path="./assets/case_1/mesh.glb",
    image_path="./assets/case_1/image.png",
)
```

`demo.py` has no argparse interface and no checkpoint argument. It hardcodes the
input mesh and reference image under `./assets/case_1/`.

## Config Fields

`Hunyuan3DPaintConfig.__init__(max_num_view, resolution)` assigns:

- `device = "cuda"`
- `multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"`
- `custom_pipeline = "hunyuanpaintpbr"`
- `multiview_pretrained_path = "tencent/Hunyuan3D-2.1"`
- `dino_ckpt_path = "facebook/dinov2-giant"`
- `realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"`
- `raster_mode = "cr"`
- `bake_mode = "back_sample"`
- `render_size = 1024 * 2`
- `texture_size = 1024 * 4`
- `max_selected_view_num = max_num_view`
- `resolution = resolution`
- `bake_exp = 4`
- `merge_method = "fast"`
- candidate camera azimuth/elevation/weight lists for view selection

The project wrapper can set `conf.device` after construction, but the official
constructor itself defaults to CUDA.

## Pipeline Call Pattern

`Hunyuan3DPaintPipeline.__init__` creates:

- `MeshRender`
- `ViewProcessor`
- `imageSuperNet`
- `multiviewDiffusionNet`

`Hunyuan3DPaintPipeline.__call__` accepts:

- `mesh_path`
- `image_path`
- `output_mesh_path`
- `use_remesh=True`
- `save_glb=True`

It remeshes by default, renders normal and position maps, calls
`models["multiview_model"]`, super-resolves albedo/MR images, bakes textures,
saves an OBJ, optionally converts to GLB, and returns the OBJ path.

## Base Weight Loading

Base multiview weights are loaded in `utils/multiview_utils.py` inside
`multiviewDiffusionNet.__init__`:

- `huggingface_hub.snapshot_download(repo_id=config.multiview_pretrained_path,
  allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])`
- `DiffusionPipeline.from_pretrained(model_path, custom_pipeline=custom_pipeline,
  torch_dtype=torch.float16)`

The custom pipeline class is `hunyuanpaintpbr.pipeline.HunyuanPaintPipeline`.
That class wraps a diffusers Stable Diffusion pipeline and converts the UNet to
`UNet2p5DConditionModel` when needed.

## Checkpoint Loading

The official inference path does not appear to expose checkpoint loading through
`Hunyuan3DPaintConfig`, `Hunyuan3DPaintPipeline`, or `demo.py`.

`hunyuanpaintpbr/model.py` is not present in this checkout. The training class is
in `hunyuanpaintpbr/unet/model.py` as `HunyuanPaint`.

`train.py` is the clearest checkpoint-loading reference. Based on text
inspection only:

- `--resume` and `--resume_weights_only` exist for training.
- `config.resume_from` is loaded with `torch.load(path, map_location="cpu")["state_dict"]`.
- UNet keys are filtered by `model_unet_prefix`, initially `unet.unet.` and
  possibly `unet.unet.unet.` if nested.
- Matching keys are stripped of that prefix and loaded into `model_unet` with
  `load_state_dict(..., strict=True)`.
- ControlNet keys, if present, are filtered with `unet.controlnet.` and loaded
  separately.
- `opt.resume and opt.resume_weights_only` uses Lightning
  `load_from_checkpoint`, but the official `demo.py` inference object is not the
  same training `LightningModule`.

Therefore fine-tuned inference likely requires manual `torch.load` plus
`load_state_dict` against the inference pipeline's `multiview_model.pipeline.unet`
or a related nested UNet object. The exact mapping is still unknown because the
9.5GB checkpoint has not been loaded or inspected.

## Risks And Unknowns

- The saved checkpoint key layout may differ from the `train.py` assumptions.
- The inference pipeline UNet nesting may not match `model_unet_prefix` from
  training.
- `save_weights_only: true` may still wrap tensors under a Lightning
  `state_dict`, but this must be confirmed later on A100.
- Base inference may attempt Hugging Face cache access if the base model is not
  already cached.
- Fine-tuned mode must not silently fall back to base mode.

## Recommended First A100 Route

1. Run checkpoint metadata inspection without opening the checkpoint.
2. Run wrapper dry-runs for `base` and `finetuned`.
3. Run Phase 2G.1 readiness.
4. Review outputs before creating an A100 sbatch.
5. First A100 execution should verify base wrapper inference on the prepared
   case.
6. Only after base wrapper inference works, add an explicit checkpoint-key
   inspection/load step on A100 and keep `finetuned` mode failing loudly until
   the key mapping is confirmed.
