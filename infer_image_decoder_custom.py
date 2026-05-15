"""Custom image inference for DiffusionImageDecoder v2.

임의의 이미지 + GT caption을 받아 paired (zimg vs ztxt) recon을 한 PNG로 출력.
COCO val 캐시가 없는 새 이미지로 ad-hoc 정성평가 / 데모 시 사용.

Pipeline (1장 처리):
  image (.jpg / .png / ...) ──► RN50 image enc (frozen Run 1) ──► z_img (1024-D)
  caption (str)             ──► RN50 text  enc (frozen Run 1) ──► z_txt (1024-D)
  z_img → TextDecoder.generate() → ĉ_img (M3 caption for zimg recon)
  z_txt → TextDecoder.generate() → ĉ_txt (M3 caption for ztxt recon)
  (z_img, ĉ_img) → DiffusionImageDecoder → recon_zimg
  (z_txt, ĉ_txt) → DiffusionImageDecoder → recon_ztxt
  → save_grid_paired_comparison (1 PNG per cfg scale)

Usage:
  python Code/SemComm/infer_image_decoder_custom.py \
      --run-dir runs/imgdiff_..._<ts> \
      --ckpt epoch_010.pt \
      --image /path/to/photo.jpg \
      --caption "A man with headphones sitting at a desk" \
      --noise-seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open_clip
import torch
import torchvision.transforms as T
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import DiffusionImageDecoder, TextDecoder  # noqa: E402
from infer_image_decoder_v2 import (  # noqa: E402
    make_pixel_transform, resolve, save_grid_paired_comparison,
)

MODGAP = HERE.parent / "ModalityGap"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def encoder_transform() -> T.Compose:
    """Same as encode_dataset.py — CLIP preprocessing (mean/std)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_clip_encoder(cfg: dict, device: torch.device):
    """Mirrors encode_dataset.load_encoder."""
    model_name = cfg["encoder_model"]
    ckpt_path = MODGAP / "runs" / cfg["encoder_run"] / "checkpoints" / cfg["encoder_ckpt_filename"]
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=None, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"[encoder] {model_name} loaded from {ckpt_path}")
    return model, tokenizer


def load_text_decoder(text_decoder_run: str, device: torch.device) -> TextDecoder:
    """Rebuild TextDecoder from saved ckpt (mirrors encode_captions.py)."""
    # 새 구조: checkpoints/models/final.pt, 옛 구조: checkpoints/final.pt 둘 다 지원.
    ckpt_path = HERE / "runs" / text_decoder_run / "checkpoints" / "models" / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = HERE / "runs" / text_decoder_run / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"text decoder ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    td_cfg = ckpt["config"]["text_decoder"]
    model = TextDecoder(
        lm_name=td_cfg["lm_name"],
        z_dim=1024,
        prefix_len=td_cfg["prefix_len"],
        mapper_type=td_cfg["mapper_type"],
        mapper_layers=td_cfg["mapper_layers"],
        mapper_heads=td_cfg["mapper_heads"],
        mapper_dropout=td_cfg["mapper_dropout"],
        freeze_lm=td_cfg["freeze_lm"],
        lm_dtype=td_cfg.get("lm_dtype", "auto"),
    ).to(device)
    model.mapper.load_state_dict(ckpt["mapper"])
    model.eval()
    print(f"[text-decoder] loaded from {ckpt_path} (z_source={td_cfg['z_source']})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="image decoder run dir (e.g. runs/imgdiff_..._<ts>)")
    ap.add_argument("--ckpt", default="final.pt",
                    help="ckpt filename under run-dir/checkpoints/ (default: final.pt)")
    ap.add_argument("--image", required=True,
                    help="path to custom image file (any format PIL can read)")
    ap.add_argument("--caption", required=True,
                    help="GT caption — used as z_txt source + shown in panel")
    ap.add_argument("--cfg-scales", type=float, nargs="+", default=[1.5, 3.0, 7.0],
                    help="CFG scales to try (default: 1.5 3.0 7.0). 각 scale마다 1 PNG.")
    ap.add_argument("--steps", type=int, default=30, help="DDIM step")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="diffusion 초기 latent noise seed (zimg ↔ ztxt 공정 비교용)")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--no-caption", action="store_true",
                    help="caption stream OFF (학습된 ckpt가 M3 on이어도 강제 off)")
    ap.add_argument("--text-decoder-run-override", default=None,
                    help="img_cfg.text_decoder_run 대신 다른 text decoder run 사용 (드물게 사용)")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    run_dir = resolve(args.run_dir, HERE)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    print(f"[load] image decoder ckpt: {ckpt_path}")
    img_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = img_ckpt["config"]
    img_cfg = cfg["image_decoder"]
    print(f"[load] epoch={img_ckpt['epoch']}, z_source(train)={img_cfg['z_source']}")

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    precision = img_cfg.get("precision", "bf16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    text_decoder_run = args.text_decoder_run_override or img_cfg.get("text_decoder_run", "") or ""
    caption_stream_on = bool(text_decoder_run) and not args.no_caption

    # 1) CLIP encoder load
    clip_model, tokenizer = load_clip_encoder(cfg, device)

    # 2) image → z_img
    img = Image.open(image_path).convert("RGB")
    pixel_for_clip = encoder_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z_img = clip_model.encode_image(pixel_for_clip)
    z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    z_img = z_img.float()
    print(f"[encode] z_img shape={tuple(z_img.shape)} norm={z_img.norm().item():.4f}")

    # 3) caption → z_txt
    tokens = tokenizer([args.caption]).to(device)
    with torch.no_grad():
        z_txt = clip_model.encode_text(tokens)
    z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
    z_txt = z_txt.float()
    print(f"[encode] z_txt shape={tuple(z_txt.shape)} norm={z_txt.norm().item():.4f}")
    print(f"[encode] caption: {args.caption!r}")

    # CLIP encoder 더 이상 필요 없음 → free VRAM
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4) Text decoder로 ĉ 생성 (M3 stream에 사용)
    decoded_zimg, decoded_ztxt = None, None
    if caption_stream_on:
        td_model = load_text_decoder(text_decoder_run, device)
        with torch.no_grad():
            decoded_zimg = td_model.generate(z_img, beam_size=1, max_new_tokens=20)
            decoded_ztxt = td_model.generate(z_txt, beam_size=1, max_new_tokens=20)
        print(f"[generate] ĉ from zimg: {decoded_zimg[0]!r}")
        print(f"[generate] ĉ from ztxt: {decoded_ztxt[0]!r}")
        del td_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[caption] stream OFF — z only conditioning")

    # 5) DiffusionImageDecoder load
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
    set_peft_model_state_dict(model.unet, img_ckpt["lora"])
    model.z_proj.load_state_dict(img_ckpt["z_proj"])
    print("[load] LoRA + z_proj state loaded")

    from diffusers import DDIMScheduler
    ddim = DDIMScheduler.from_pretrained(model.sd_model_id, subfolder="scheduler")
    ddim.set_timesteps(args.steps)

    # Display 용 GT image (SD norm [-1,1], denorm 시 [0,1])
    pixel_for_display = make_pixel_transform()(img).unsqueeze(0).to(device).to(dtype)

    def seed_noise():
        if args.noise_seed is not None:
            torch.manual_seed(args.noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.noise_seed)

    def run_one(z_in: torch.Tensor, cap_in: list[str] | None, cfg_scale: float):
        """CFG batching 적용: cond + uncond 를 한 batch 로 합쳐 UNet 1번 호출."""
        use_cfg = cfg_scale != 1.0
        with torch.no_grad():
            cond = model.build_condition(z_in.to(device), cap_in)
            ctx = torch.cat([torch.zeros_like(cond), cond], dim=0) if use_cfg else cond
            latent = torch.randn(1, 4, 28, 28, device=device, dtype=dtype)
            for t in ddim.timesteps:
                ts = torch.tensor([t.item()], device=device).long()
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
        return x_hat

    out_dir = run_dir / "samples" / args.ckpt.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_basename = image_path.stem
    nseed_suffix = f"_nseed{args.noise_seed}" if args.noise_seed is not None else ""
    print(f"[sample] {len(args.cfg_scales)} cfg scale(s) → {out_dir}")
    for cfg_scale in args.cfg_scales:
        seed_noise()
        x_hat_zimg = run_one(z_img, decoded_zimg, cfg_scale)
        seed_noise()  # ztxt도 동일 noise 로 reset → 공정 비교
        x_hat_ztxt = run_one(z_txt, decoded_ztxt, cfg_scale)
        out_path = out_dir / f"custom_{image_basename}_cfg{cfg_scale}{nseed_suffix}.png"
        save_grid_paired_comparison(
            pixel_for_display, x_hat_zimg, x_hat_ztxt,
            [args.caption],
            decoded_zimg, decoded_ztxt,
            out_path,
        )
        print(f"  [save] {out_path.name}")


if __name__ == "__main__":
    main()
