"""Image decoder validation — ConvT 와 Diffusion(+M1+M3) 양쪽 모두 평가.

ckpt 구조로 decoder type 자동 감지:
  - 'model' 키 있음 → ConvT (ImageDecoder)
  - 'lora' / 'z_proj' 키 있음 → Diffusion (DiffusionImageDecoder)

Metric (모두 빠른 것만):
  - FID         (clean-fid, recon 분포 vs GT 분포)
  - LPIPS-vgg   (paired perceptual)
  - CLIP-score  (recon ↔ GT caption, ViT-B/32 OAI — 학습 encoder 와 다른 모델로 circular eval 회피)
  - PSNR, SSIM  (pixel-level, baseline 비교용)

각 z_source 별로 metric 계산, Δ = z_txt − z_img 도 자동 계산.

Usage:
  # ConvT baseline
  python Code/SemComm/eval_image.py \
      --run-dir runs/img_convt_..._<ts> --n 1000

  # Diffusion (proposed)
  python Code/SemComm/eval_image.py \
      --run-dir runs/imgdiff_..._<ts> --ckpt epoch_010.pt \
      --n 1000 --cfg-scale 3.0 --noise-seed 7

  # 둘 다 같은 subset으로 평가하려면 --seed 동일하게 사용.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import DiffusionImageDecoder, ImageDecoder  # noqa: E402


def resolve(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def make_pixel_transform() -> T.Compose:
    """SD-style normalize [-1, 1]. ConvT와 Diffusion 양쪽 모두 이 range로 출력."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1] clamp."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------
# Decoder loading (auto-detect)

def detect_decoder_type(ckpt: dict) -> str:
    if "lora" in ckpt and "z_proj" in ckpt:
        return "diffusion"
    if "model" in ckpt:
        return "convt"
    raise ValueError(f"Unknown ckpt format. Keys: {list(ckpt.keys())}")


def load_decoder(ckpt_path: Path, device: torch.device, dtype: torch.dtype):
    """Return (model, decoder_type, img_cfg, caption_stream_on, text_decoder_run)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    img_cfg = cfg["image_decoder"]
    decoder_type = detect_decoder_type(ckpt)
    text_decoder_run = img_cfg.get("text_decoder_run", "") or ""

    if decoder_type == "convt":
        model = ImageDecoder(z_dim=1024, hidden_init=img_cfg.get("hidden_init", 512)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, decoder_type, cfg, img_cfg, False, ""

    # diffusion
    caption_stream_on = bool(text_decoder_run)
    model = DiffusionImageDecoder(
        z_dim=1024,
        n_z_tokens=img_cfg.get("n_z_tokens", 10),
        lora_rank=img_cfg.get("lora_rank", 8),
        lora_alpha=img_cfg.get("lora_rank", 8) * 2,
        caption_conditioning=caption_stream_on,
        dtype=dtype,
    ).to(device)
    model.eval()
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model.unet, ckpt["lora"])
    model.z_proj.load_state_dict(ckpt["z_proj"])
    return model, decoder_type, cfg, img_cfg, caption_stream_on, text_decoder_run


# ----------------------------------------------------------------------------
# Reconstruction loops

@torch.no_grad()
def reconstruct_convt(model: ImageDecoder, z: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ConvT: 단일 forward. (N, 3, 224, 224) in [-1, 1].
    ConvT 모델 파라미터 dtype 에 input 을 맞춰 캐스팅 — bf16/fp16 ckpt 와 fp32 양쪽 모두 호환."""
    model_dtype = next(model.parameters()).dtype
    outs = []
    for s in range(0, z.size(0), batch_size):
        zb = z[s:s + batch_size].to(model_dtype)
        x_hat = model(zb)
        outs.append(x_hat.float())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def reconstruct_diffusion(model: DiffusionImageDecoder, z: torch.Tensor,
                          captions: list[str] | None, batch_size: int, dtype: torch.dtype,
                          n_steps: int, cfg_scale: float, device: torch.device,
                          noise_seed: int | None) -> torch.Tensor:
    """Diffusion DDIM + CFG batching. (N, 3, 224, 224) in [-1, 1]."""
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler.from_pretrained(model.sd_model_id, subfolder="scheduler")
    ddim.set_timesteps(n_steps)
    use_cfg = cfg_scale != 1.0
    outs = []
    N = z.size(0)
    n_batches = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(0, N, batch_size), total=n_batches, desc="ddim batch", leave=False)
    for s in pbar:
        zb = z[s:s + batch_size].to(device)
        cb = captions[s:s + batch_size] if captions is not None else None
        B = zb.size(0)
        if noise_seed is not None:
            # 각 batch 시작마다 같은 seed로 reset → batching이 noise pattern에 영향 주지 않음.
            # 단, BS 가 바뀌면 결과도 바뀐다는 점은 일반 디퓨전과 같음.
            torch.manual_seed(noise_seed + s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(noise_seed + s)
        cond = model.build_condition(zb, cb)
        ctx = torch.cat([torch.zeros_like(cond), cond], dim=0) if use_cfg else cond
        latent = torch.randn(B, 4, 28, 28, device=device, dtype=dtype)
        for t in tqdm(ddim.timesteps, desc=f"  ddim t (batch {s//batch_size+1}/{n_batches})",
                      leave=False, total=len(ddim.timesteps)):
            ts = torch.tensor([t.item()] * B, device=device).long()
            if use_cfg:
                latent_2x = torch.cat([latent, latent], dim=0)
                ts_2x = torch.cat([ts, ts], dim=0)
                eps_2x = model.unet(latent_2x, ts_2x, encoder_hidden_states=ctx).sample
                eps_u, eps_c = eps_2x.chunk(2, dim=0)
                eps = eps_u + cfg_scale * (eps_c - eps_u)
            else:
                eps = model.unet(latent, ts, encoder_hidden_states=ctx).sample
            latent = ddim.step(eps, t, latent).prev_sample
        x_hat = model.decode_latent_to_image(latent)
        outs.append(x_hat.float())
        pbar.set_postfix(done=f"{min(s+batch_size, N)}/{N}")
    return torch.cat(outs, dim=0)


# ----------------------------------------------------------------------------
# Metric primitives

@torch.no_grad()
def compute_psnr(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """recon, gt: (B, 3, H, W) in [-1, 1]. Returns mean PSNR over batch."""
    r = denorm(recon); g = denorm(gt)
    mse = ((r - g) ** 2).mean(dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return psnr.mean().item()


@torch.no_grad()
def compute_lpips_stream(lpips_fn, recon: torch.Tensor, gt: torch.Tensor,
                         batch_size: int = 32) -> float:
    """LPIPS expects [-1, 1]. Batch-by-batch to control memory."""
    vals = []
    n_batches = (recon.size(0) + batch_size - 1) // batch_size
    for s in tqdm(range(0, recon.size(0), batch_size), total=n_batches,
                  desc="    lpips", leave=False):
        v = lpips_fn(recon[s:s + batch_size], gt[s:s + batch_size]).squeeze()
        if v.ndim == 0:
            v = v.unsqueeze(0)
        vals.append(v)
    return torch.cat(vals).mean().item()


@torch.no_grad()
def compute_ssim(recon: torch.Tensor, gt: torch.Tensor) -> float:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(recon.device)
    return ssim_fn(denorm(recon), denorm(gt)).item()


@torch.no_grad()
def compute_clip_score(clip_model, clip_preprocess, clip_tokenizer,
                       recon: torch.Tensor, captions: list[str],
                       batch_size: int = 64, device: torch.device = "cuda") -> float:
    """recon: (N, 3, 224, 224) in [-1, 1]. CLIP eval encoder는 학습용 RN50과 다른 ViT-B/32."""
    imgs_01 = denorm(recon)  # [0, 1]
    # CLIP eval은 자체 normalize 필요 — recon은 224x224, [0,1] 이면 CLIP norm만 적용.
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    imgs_norm = (imgs_01.to(device) - clip_mean) / clip_std

    img_feats = []
    for s in range(0, recon.size(0), batch_size):
        f = clip_model.encode_image(imgs_norm[s:s + batch_size])
        f = f / f.norm(dim=-1, keepdim=True)
        img_feats.append(f.float())
    img_feats = torch.cat(img_feats, dim=0)

    txt_feats = []
    for s in range(0, len(captions), batch_size):
        tokens = clip_tokenizer(captions[s:s + batch_size]).to(device)
        f = clip_model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        txt_feats.append(f.float())
    txt_feats = torch.cat(txt_feats, dim=0)

    return (img_feats * txt_feats).sum(-1).mean().item()


def compute_fid(recon: torch.Tensor, gt: torch.Tensor) -> float:
    """Save both to temp PNGs, then clean-fid. recon, gt in [-1, 1]."""
    from cleanfid import fid as cleanfid
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        recon_dir = td / "recon"; gt_dir = td / "gt"
        recon_dir.mkdir(); gt_dir.mkdir()
        r01 = denorm(recon).clamp(0, 1).cpu()
        g01 = denorm(gt).clamp(0, 1).cpu()
        for i in tqdm(range(recon.size(0)), desc="    saving PNG for FID", leave=False):
            T.ToPILImage()(r01[i]).save(recon_dir / f"{i:06d}.png")
            T.ToPILImage()(g01[i]).save(gt_dir / f"{i:06d}.png")
        score = cleanfid.compute_fid(str(gt_dir), str(recon_dir),
                                     mode="clean", batch_size=64, verbose=True)
    return float(score)


# ----------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="image decoder run dir")
    ap.add_argument("--ckpt", default="final.pt", help="ckpt under run-dir/checkpoints/")
    ap.add_argument("--n", type=int, default=1000, help="평가 sample 수 (FID는 1k 이상 권장)")
    ap.add_argument("--seed", type=int, default=42, help="val image selection seed (둘 비교 시 동일)")
    ap.add_argument("--noise-seed", type=int, default=7,
                    help="diffusion 초기 latent seed (둘 비교 시 동일)")
    ap.add_argument("--batch-size", type=int, default=64, help="reconstruction batch size")
    ap.add_argument("--steps", type=int, default=30, help="DDIM step (diffusion only)")
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="classifier-free guidance scale (diffusion only). 1.0 = no guidance.")
    ap.add_argument("--z-sources", nargs="+", default=["zimg", "ztxt"],
                    choices=["zimg", "ztxt", "centroid"])
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--skip-fid", action="store_true", help="FID 만 skip (cache PNG 저장 등 비용)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    run_dir = resolve(args.run_dir, HERE)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    # ------------------------------------------------------------------------
    # Load decoder
    print(f"[load] decoder ckpt: {ckpt_path}")
    # dtype 은 ckpt 의 precision 기준
    pre_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    precision = pre_ckpt["config"]["image_decoder"].get("precision", "bf16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    del pre_ckpt
    model, decoder_type, cfg, img_cfg, caption_stream_on, text_decoder_run = load_decoder(
        ckpt_path, device, dtype
    )
    print(f"[load] decoder_type={decoder_type}, caption_stream_on={caption_stream_on}, "
          f"text_decoder_run={text_decoder_run!r}")

    # ------------------------------------------------------------------------
    # Load val cache + select N images (same seed for both decoders → 공정 비교)
    cache_dir = resolve(cfg["cache_dir"], HERE)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)
    val_image_ids = idx["val"]["image_ids"]
    val_caption_image_idx = idx["val"]["caption_image_idx"]
    val_caption_texts = idx["val"]["caption_texts"]

    captions_zimg_val = captions_ztxt_val = None
    if caption_stream_on:
        prefix = f"captions_{text_decoder_run}"
        captions_zimg_val = torch.load(cache_dir / f"{prefix}_val_zimg.pt", weights_only=False)
        captions_ztxt_val = torch.load(cache_dir / f"{prefix}_val_ztxt.pt", weights_only=False)

    # image_idx → first caption idx
    first_cap_for_image: dict[int, int] = {}
    for cap_idx, im_idx in enumerate(val_caption_image_idx):
        if im_idx not in first_cap_for_image:
            first_cap_for_image[im_idx] = cap_idx
    available = sorted(first_cap_for_image.keys())
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(available), generator=g).tolist()
    selected_idx = [available[p] for p in perm[:args.n]]
    print(f"[select] N={len(selected_idx)} (seed={args.seed}) — first 5: {selected_idx[:5]}")

    coco_val = resolve(cfg["coco_root"], HERE) / "images" / "val2017"
    transform = make_pixel_transform()
    xs, z_imgs, z_txts, gt_caps, zimg_caps, ztxt_caps = [], [], [], [], [], []
    for i in selected_idx:
        j = first_cap_for_image[i]
        iid = val_image_ids[i]
        img = Image.open(coco_val / f"{iid:012d}.jpg").convert("RGB")
        xs.append(transform(img))
        z_imgs.append(z_img_val[i].float())
        z_txts.append(z_txt_val[j].float())
        gt_caps.append(val_caption_texts[j])
        if caption_stream_on:
            zimg_caps.append(captions_zimg_val[i])
            ztxt_caps.append(captions_ztxt_val[j])
    x_gt = torch.stack(xs).to(device).to(dtype)
    z_img_b = torch.stack(z_imgs)
    z_txt_b = torch.stack(z_txts)

    # ------------------------------------------------------------------------
    # CLIP eval encoder (ViT-B/32 OAI — 학습 RN50 과 다른 모델)
    print("[load] CLIP eval encoder (ViT-B-32, OpenAI) ...")
    clip_eval, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    clip_eval.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    for p in clip_eval.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------------
    # LPIPS
    import lpips as lpips_pkg
    print("[load] LPIPS-vgg ...")
    lpips_fn = lpips_pkg.LPIPS(net="vgg").to(device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------------
    # Per-z_source reconstruction + metric
    results: dict[str, dict[str, float]] = {}
    for zsrc in args.z_sources:
        print(f"\n[eval] z_source = {zsrc}")
        t0 = time.time()
        if zsrc == "zimg":
            z_in = z_img_b
            caps_in = list(zimg_caps) if caption_stream_on else None
        elif zsrc == "ztxt":
            z_in = z_txt_b
            caps_in = list(ztxt_caps) if caption_stream_on else None
        elif zsrc == "centroid":
            z_in = F.normalize((z_img_b + z_txt_b) / 2.0, dim=-1)
            caps_in = list(gt_caps) if caption_stream_on else None

        if decoder_type == "convt":
            x_hat = reconstruct_convt(model, z_in.to(device), args.batch_size)
        else:
            x_hat = reconstruct_diffusion(
                model, z_in, caps_in, args.batch_size, dtype,
                args.steps, args.cfg_scale, device, args.noise_seed,
            )
        x_hat = x_hat.to(device)
        print(f"  [recon] {x_hat.shape} in {time.time() - t0:.1f}s")

        # Metrics
        m: dict[str, float] = {}
        print(f"  [psnr/ssim] computing ...")
        m["psnr"] = compute_psnr(x_hat, x_gt.float())
        m["ssim"] = compute_ssim(x_hat, x_gt.float())
        print(f"  [lpips] computing ...")
        m["lpips"] = compute_lpips_stream(lpips_fn, x_hat, x_gt.float())
        print(f"  [clip_score] computing ...")
        m["clip_score"] = compute_clip_score(
            clip_eval, None, clip_tokenizer, x_hat, gt_caps,
            batch_size=64, device=device,
        )
        if not args.skip_fid:
            t_fid = time.time()
            print(f"  [fid] computing (saving {x_hat.size(0)} PNGs + clean-fid) ...")
            m["fid"] = compute_fid(x_hat, x_gt.float())
            print(f"  [fid] = {m['fid']:.2f} in {time.time() - t_fid:.1f}s")
        print(f"  metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
        results[zsrc] = m

        # Free recon to save memory before next z_source
        del x_hat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Δ = ztxt − zimg
    delta: dict[str, float] = {}
    if "zimg" in results and "ztxt" in results:
        for k in results["zimg"]:
            delta[k] = results["ztxt"][k] - results["zimg"][k]
        results["delta_ztxt_minus_zimg"] = delta
        print("\n[Δ ztxt − zimg] " + ", ".join(f"{k}={v:+.3f}" for k, v in delta.items()))

    # ------------------------------------------------------------------------
    # Save
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    tag_parts = [f"n{args.n}", f"seed{args.seed}"]
    if decoder_type == "diffusion":
        tag_parts.append(f"cfg{args.cfg_scale}")
        tag_parts.append(f"nseed{args.noise_seed}")
    tag = "_".join(tag_parts)
    out_path = out_dir / f"metrics_{tag}.json"
    payload = {
        "ckpt": str(ckpt_path.relative_to(HERE)) if ckpt_path.is_relative_to(HERE) else str(ckpt_path),
        "decoder_type": decoder_type,
        "caption_stream_on": caption_stream_on,
        "text_decoder_run": text_decoder_run,
        "n": args.n,
        "seed": args.seed,
        "noise_seed": args.noise_seed if decoder_type == "diffusion" else None,
        "cfg_scale": args.cfg_scale if decoder_type == "diffusion" else None,
        "steps": args.steps if decoder_type == "diffusion" else None,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
