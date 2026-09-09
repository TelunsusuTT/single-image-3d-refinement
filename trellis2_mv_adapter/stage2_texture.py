#!/usr/bin/env python3
"""Generate multi-view textures with MV-Adapter and bake them to a PBR GLB."""

from __future__ import annotations

import argparse
import os
import sys

BASE_MODEL_LOCAL = "/root/autodl-tmp/weights/sdxl-base-1.0"
VAE_MODEL_LOCAL = "/root/autodl-tmp/weights/sdxl-vae-fp16-fix"
ADAPTER_LOCAL = "/root/autodl-tmp/weights/mv-adapter"
MV_ADAPTER_DIR = "/root/autodl-tmp/MV-Adapter"
CHECKPOINTS_DIR = os.path.join(MV_ADAPTER_DIR, "checkpoints")
BIRENET_LOCAL = "/root/autodl-tmp/weights/BiRefNet"

sys.path.insert(0, MV_ADAPTER_DIR)


def normalize_view_strip(mv_path: str, ref_path: str, out_path: str,
                         num_views: int, cap: float) -> str:
    """Match view foreground brightness to the reference; preserve backgrounds."""
    import numpy as np
    from PIL import Image

    def bg_of(view):
        flat = view.reshape(-1, 3)
        u, c = np.unique(flat, axis=0, return_counts=True)
        return u[np.argmax(c)]

    def fg_mask(view, bg, tol=18):
        return np.abs(view.astype(np.int16) - bg).sum(2) > tol

    ref_im = Image.open(ref_path)
    use_alpha = ref_im.mode == "RGBA"
    a = np.asarray(ref_im.convert("RGBA"), dtype=np.float32)
    m = a[:, :, 3] > 20 if use_alpha else a[:, :, :3].sum(2) > 75
    t_mean = float(a[:, :, :3][m].mean())
    print(f"[stage2] View normalisation: reference foreground brightness {t_mean:.1f} (mask={'alpha' if use_alpha else 'brightness threshold'})")

    A = np.asarray(Image.open(mv_path).convert("RGB"), dtype=np.float32)
    h, w = A.shape[:2]
    sw = w // num_views
    views = []
    for i in range(num_views):
        v = A[:, i*sw:(i+1)*sw].copy()
        bg = bg_of(v.astype(np.uint8))
        fg = fg_mask(v, bg)
        obj = v[fg]
        m0 = float(obj.mean())
        mult = float(np.clip(t_mean / max(m0, 1e-6), 1.0/cap, cap))
        v[fg] = np.clip(obj * mult, 0, 255)
        views.append(v.astype(np.uint8))
        print(f"[stage2]   View normalisation v{i+1}: foreground brightness {m0:.0f} → {m0*mult:.0f} (×{mult:.2f})")
    grid = np.concatenate(views, axis=1)
    Image.fromarray(grid).save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="Preprocessed stage1 geometry GLB with UVs")
    ap.add_argument("image", help="Reference image")
    ap.add_argument("--variant", default="sdxl", choices=["sdxl", "sd21"])
    ap.add_argument("--save_dir", default="./outputs")
    ap.add_argument("--save_name", default="textured")
    ap.add_argument("--remove_bg", action="store_true")
    ap.add_argument("--num_views", type=int, default=6)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ref_cond_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--text", default="high quality")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bake_only", action="store_true",
                    help="Bake from an existing six-view PNG")
    ap.add_argument("--mv_path", default=None, help="Six-view PNG for bake_only")
    ap.add_argument("--inpaint_mode", default="view", choices=["none", "uv", "view"],
                    help="view: SmartPainter inpainting (default); none: projection only")
    ap.add_argument("--no_uv_unwarp", action="store_true",
                    help="Skip process_raw and bake using existing mesh UVs")
    ap.add_argument("--no_preprocess", action="store_true",
                    help="Disable mesh repair and UV re-unwrapping in process_raw")
    ap.add_argument("--normalize_views", action="store_true",
                    help="Match each view foreground brightness to the reference before baking")
    ap.add_argument("--norm_cap", type=float, default=2.0,
                    help="Limit brightness scaling to [1/cap, cap]")
    ap.add_argument("--no_view_upscale", action="store_true",
                    help="Disable Real-ESRGAN view upscaling during baking")
    ap.add_argument("--upscale_factor", type=int, default=2,
                    help="View upscaling factor (default: 2, RealESRGAN_x2plus)")
    ap.add_argument("--upscaler_path", default=None,
                    help="Upscaler weights (default: MV-Adapter checkpoints/RealESRGAN_x2plus.pth)")
    ap.add_argument("--gen_only", action="store_true",
                    help="Save generated views as {save_name}_mv(_norm).png without baking")
    ap.add_argument("--decimate_target", type=int, default=0,
                    help="A2: override the process_raw face target (normally 50000)")
    ap.add_argument("--watertight", action="store_true",
                    help="A3: watertight voxel remeshing before generation and baking")
    ap.add_argument("--wt_grid", type=int, default=256, help="A3 voxel grid resolution")
    ap.add_argument("--wt_decimate", type=int, default=200000, help="A3 target face count (0 disables decimation)")
    ap.add_argument("--seal_grid", type=int, default=128,
                    help="B2 voxel grid resolution (coarser grids increase sealing and smoothing)")
    ap.add_argument("--smooth_normals", type=int, default=0,
                    help="B1: normal smoothing iterations; 0 disables; vertices stay fixed")
    ap.add_argument("--strong_seal", action="store_true",
                    help="B2: coarser voxel remeshing for stronger sealing and smoothing")
    args = ap.parse_args()

    import mesh_methods
    os.makedirs(args.save_dir, exist_ok=True)
    mesh_gen = args.mesh
    if args.decimate_target and args.decimate_target > 0:
        mesh_methods.patch_process_mesh_decimate_target(args.decimate_target)
    if args.smooth_normals and args.smooth_normals > 0:
        # Patch before imports bind load_mesh.
        mesh_methods.patch_load_mesh_smooth_normals(args.smooth_normals)
    if args.watertight:
        mesh_gen = mesh_methods.watertight_voxel_mesh(
            args.mesh, os.path.join(args.save_dir, f"{args.save_name}_wt.glb"),
            grid=args.wt_grid, decimate_to=args.wt_decimate)
    if args.strong_seal:
        mesh_gen = mesh_methods.watertight_voxel_mesh(
            args.mesh, os.path.join(args.save_dir, f"{args.save_name}_seal.glb"),
            grid=args.seal_grid, decimate_to=args.wt_decimate)
    if mesh_gen != args.mesh:
        print(f"[stage2] Mesh post-processing: {os.path.basename(args.mesh)} -> {os.path.basename(mesh_gen)}")

    import torch

    if args.variant == "sdxl":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scripts.inference_ig2mv_sdxl as impl
        base_model = BASE_MODEL_LOCAL if os.path.isdir(BASE_MODEL_LOCAL) else "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model = VAE_MODEL_LOCAL if os.path.isdir(VAE_MODEL_LOCAL) else "madebyollin/sdxl-vae-fp16-fix"
        height = width = 768
        uv_size = 4096
    else:
        import scripts.inference_ig2mv_sd as impl
        base_model = "stabilityai/stable-diffusion-2-1-base"
        vae_model = None
        height = width = 512
        uv_size = 2048

    adapter_path = ADAPTER_LOCAL if os.path.isdir(ADAPTER_LOCAL) else "huanngzh/mv-adapter"

    if args.bake_only:
        assert args.mv_path and os.path.exists(args.mv_path), f"bake_only requires an existing six-view image: {args.mv_path}"
        mv_path = args.mv_path
    else:
        prepare_pipeline, run_pipeline = impl.prepare_pipeline, impl.run_pipeline

        remove_bg_fn = None
        if args.remove_bg:
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                BIRENET_LOCAL if os.path.isdir(BIRENET_LOCAL) else "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
            ).to(args.device, dtype=torch.float32)
            tf = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            remove_bg_fn = lambda x: impl.remove_bg(x.convert("RGB"), birefnet, tf, args.device)

        print(f"[stage2] Preparing {args.variant} pipeline (base={os.path.basename(base_model)})")
        pipe = prepare_pipeline(
            base_model=base_model,
            vae_model=vae_model,
            unet_model=None,
            lora_model=None,
            adapter_path=adapter_path,
            scheduler=None,
            num_views=args.num_views,
            device=args.device,
            dtype=torch.float16,
        )

        from mvadapter.utils import make_image_grid
        os.makedirs(args.save_dir, exist_ok=True)
        images, _, _, _ = run_pipeline(
            pipe,
            mesh_path=mesh_gen,
            num_views=args.num_views,
            text=args.text,
            image=args.image,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            reference_conditioning_scale=args.ref_cond_scale,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            device=args.device,
            remove_bg_fn=remove_bg_fn,
        )
        mv_path = os.path.join(args.save_dir, f"{args.save_name}_mv.png")
        make_image_grid(images, rows=1).save(mv_path)
        print(f"[stage2] Multi-view image: {mv_path}")
        torch.cuda.empty_cache()

    if args.normalize_views:
        norm_path = os.path.join(args.save_dir, f"{args.save_name}_mv_norm.png")
        mv_path = normalize_view_strip(mv_path, args.image, norm_path,
                                       args.num_views, args.norm_cap)
        print(f"[stage2] Normalised views: {mv_path}")

    if args.gen_only:
        print(f"[stage2] gen_only: views saved; baking skipped: {mv_path}")
        return 0

    from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
    from mvadapter.utils.mesh_utils.mesh_process import process_raw

    import trimesh
    import torch
    from torch.nn import functional as F
    if not args.no_uv_unwarp:
        ext = os.path.splitext(mesh_gen)[-1]
        unwarp_path = mesh_gen.replace(ext, f"_unwarp{ext}")
        process_raw(mesh_gen, unwarp_path, preprocess=not args.no_preprocess)

        # Probe winding with the baking cameras; keep faces if coverage >= 30%.
        from mvadapter.utils.mesh_utils import (NVDiffRastContextWrapper,
                                                get_orthogonal_camera, load_mesh, render)

        def aoi_coverage(mesh_path: str) -> float:
            ctx = NVDiffRastContextWrapper(device=args.device)
            mesh = load_mesh(mesh_path, rescale=True, move_to_center=False,
                             front_x_to_y=False, default_uv_size=4096, device=args.device)
            cam = get_orthogonal_camera(
                elevation_deg=[0, 0, 0, 0, 89.99, -89.99], distance=[1.0] * 6,
                left=-0.55, right=0.55, bottom=-0.55, top=0.55,
                azimuth_deg=[-90, 0, 90, 180, 90, 90], device=args.device)
            with torch.no_grad():
                out = render(ctx, mesh, cam, 256, 256, render_normal=True)
            nrm = out.normal
            ncs = (nrm[:, :, :, None, :] * cam.w2c[:, None, None, :3, :3]).sum(-1)
            ncs = F.normalize(ncs, dim=-1, p=2)
            aoi = (ncs * torch.tensor([0.0, 0.0, 1.0], device=ncs.device)[None, None, None]).sum(-1)
            aoi = aoi.clamp(0.0, 1.0)
            m = out.mask[0]
            if m.sum() == 0:
                return 0.0
            return float((aoi[0][m] > 0.2).float().mean().item())

        mm = trimesh.load(unwarp_path, force="mesh", process=False)
        cov_before = aoi_coverage(unwarp_path)
        if cov_before < 0.3:
            print(f"[stage2] Winding probe aoi={cov_before*100:.1f}% < 30% -> reverse faces")
            mm.faces = mm.faces[:, [1, 0, 2]]
            mm._vertex_normals = None
            mm._face_normals = None
            mm._triangles = None
            _ = mm.vertex_normals      # Recompute normals after reversing faces.
            mm.export(unwarp_path)
        else:
            print(f"[stage2] Winding probe aoi={cov_before*100:.1f}% >= 30% -> keep faces")

        # Re-unwrap with xatlas if either UV span is below 0.1.
        import numpy as np
        _uv = mm.visual.uv
        _span = _uv.max(0) - _uv.min(0)
        if min(_span[0], _span[1]) < 0.1:
            print(f"[stage2] UV collapse: span=({_span[0]:.3f},{_span[1]:.3f}) bounding-box area "
                  f"{_span.prod():.4f}; min span < 0.1 -> xatlas unwrap")
            try:
                import xatlas
                _v = np.asarray(mm.vertices, dtype=np.float32)
                _f = np.asarray(mm.faces, dtype=np.uint32)
                _vm, _idx, _uvs = xatlas.parametrize(_v, _f)
                mm2 = trimesh.Trimesh(
                    vertices=_v[_vm], faces=_idx.astype(np.int32), process=False)
                mm2.visual = trimesh.visual.texture.TextureVisuals(uv=_uvs.astype(np.float32))
                mm2.export(unwarp_path)
                _s2 = (_uvs.max(0) - _uvs.min(0))
                print(f"[stage2] xatlas unwrap complete: span=({_s2[0]:.3f},{_s2[1]:.3f})")
            except ImportError:
                print("[stage2] Warning: xatlas unavailable; collapsed UVs may cause blurred textures")
        bake_mesh = unwarp_path
    else:
        bake_mesh = mesh_gen

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path=(args.upscaler_path if args.upscaler_path
                            else os.path.join(CHECKPOINTS_DIR, "RealESRGAN_x2plus.pth")),
        inpaint_ckpt_path=os.path.join(CHECKPOINTS_DIR, "big-lama.pt"),
        device=args.device,
    )
    out = texture_pipe(
        mesh_path=bake_mesh,
        save_dir=args.save_dir,
        save_name=args.save_name,
        uv_unwarp=False,          # UV preprocessing is already handled above.
        preprocess_mesh=False,
        uv_size=uv_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(
            view_upscale=not args.no_view_upscale,
            view_upscale_factor=args.upscale_factor,
            inpaint_mode=args.inpaint_mode,
        ),
        # Baking subtracts 90 degrees internally; keep generation and baking transforms aligned.
        camera_azimuth_deg=[0, 90, 180, 270, 180, 180],
        front_x=False,
    )
    print(f"[stage2] PBR GLB: {out.shaded_model_save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
