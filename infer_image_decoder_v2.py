"""Standalone inference for DiffusionImageDecoder v2 ckpt — 학습 중에도 별도 process로 실행 가능.

학습 중간에 ckpt(epoch_NNN.pt) 가 생성되면 즉시 그 시점의 sample 품질을 확인할 수 있다.
또한 train_image_decoder_v2.py가 매 epoch sample을 자동 생성하지 않는 경우(log_samples_every_n_epochs>1)
에도 임의 시점/임의 sample 수로 직접 inference 가능.

학습 process와 GPU를 공유하므로 메모리 영향 ~5GB. GB10 128GB 여유라 OK.

Usage:
  # 기본 (val 첫 8개 sample, BS 8, DDIM 30 step)
  python infer_image_decoder_v2.py --run-dir runs/imgdiff_..._<ts>

  # 특정 ckpt + 더 많은 sample
  python infer_image_decoder_v2.py --run-dir runs/imgdiff_... --ckpt epoch_005.pt --n 16

  # z_source 변경 (cross-modal swap inference)
  python infer_image_decoder_v2.py --run-dir runs/... --z-source ztxt --n 16

  # CFG scale 시도
  python infer_image_decoder_v2.py --run-dir runs/... --cfg-scale 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import DiffusionImageDecoder  # noqa: E402


def make_pixel_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def resolve(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def save_grid_with_captions(x_gt: torch.Tensor, x_hat: torch.Tensor,
                            gt_captions: list[str], decoded_captions: list[str],
                            zsrc: str, out_path: Path,
                            wrap_width: int = 28, max_cols: int = 6) -> None:
    """matplotlib으로 GT row + recon row + caption text를 합쳐 한 PNG로 저장.

    zsrc == "ztxt": GT caption(z_txt의 source)과 decoded caption(captions_ztxt[j]) 둘 다 표시.
    zsrc == "zimg" / others: z 자체에 source caption이 없으므로 decoded caption만 표시.
    decoded_captions가 None이면 caption 라벨 생략 (caption stream OFF 시).

    n > max_cols인 경우 (GT, Recon) 2-row 그룹을 여러 번 쌓아 아래로 wrap.
    """
    n = x_gt.size(0)
    n_groups = (n + max_cols - 1) // max_cols
    cols = min(n, max_cols)
    rows = 2 * n_groups
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, n_groups * 6.0), squeeze=False)
    for k in range(n):
        group = k // max_cols
        col = k % max_cols
        gt_row = 2 * group
        recon_row = 2 * group + 1
        gt_img = denorm(x_gt[k]).float().cpu().permute(1, 2, 0).numpy()
        hat_img = denorm(x_hat[k]).float().cpu().permute(1, 2, 0).numpy()
        axes[gt_row, col].imshow(gt_img)
        axes[gt_row, col].set_xticks([]); axes[gt_row, col].set_yticks([])
        axes[recon_row, col].imshow(hat_img)
        axes[recon_row, col].set_xticks([]); axes[recon_row, col].set_yticks([])
        lines = []
        if zsrc == "ztxt" and gt_captions is not None:
            wrapped_gt = "\n".join(textwrap.wrap(gt_captions[k], width=wrap_width)) or "(empty)"
            lines.append(f"GT cap:\n{wrapped_gt}")
        if decoded_captions is not None:
            wrapped_dec = "\n".join(textwrap.wrap(decoded_captions[k], width=wrap_width)) or "(empty)"
            lines.append(f"Decoded:\n{wrapped_dec}")
        caption_text = "\n\n".join(lines) if lines else ""
        if caption_text:
            axes[recon_row, col].set_xlabel(caption_text, fontsize=6.5)
    # 마지막 group에서 남는 빈 칸 숨기기
    last_group_filled = n - (n_groups - 1) * max_cols
    if last_group_filled < cols:
        for col in range(last_group_filled, cols):
            axes[2 * (n_groups - 1), col].axis("off")
            axes[2 * (n_groups - 1) + 1, col].axis("off")
    for g in range(n_groups):
        axes[2 * g, 0].set_ylabel("GT", fontsize=10)
        axes[2 * g + 1, 0].set_ylabel("Recon", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="학습 run dir (e.g. runs/imgdiff_..._20260511-153122)")
    ap.add_argument("--ckpt", default="final.pt",
                    help="ckpt filename under run-dir/checkpoints/ (default: final.pt). "
                         "epoch_001.pt 등 epoch별 ckpt도 가능.")
    ap.add_argument("--n", type=int, default=8, help="number of samples")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--z-source", default="random", choices=["random", "zimg", "ztxt", "centroid"],
                    help="random=학습 시 z_source와 동일 (50/50), zimg/ztxt=강제, centroid=평균")
    ap.add_argument("--steps", type=int, default=30, help="DDIM step")
    ap.add_argument("--cfg-scale", type=float, default=1.0, help="classifier-free guidance scale")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="output PNG path. default: <run-dir>/samples/infer_<ckpt>_<zsrc>.png")
    ap.add_argument("--no-caption", action="store_true",
                    help="caption stream 강제 OFF (학습된 ckpt가 M3 on이어도)")
    ap.add_argument("--use-gt-caption", action="store_true",
                    help="cached ĉ 대신 GT caption 사용 (학습 시 curriculum과 동일 behavior)")
    ap.add_argument("--all-combos", action="store_true",
                    help="z_source × cfg_scale 조합을 한 번에 생성. default 6 PNG "
                         "(zimg/ztxt × 1.0/3.0/7.5). --include-random 시 9 PNG. "
                         "samples/epoch_<ckpt>/recon_{zsrc}_cfg{scale}.png로 저장.")
    ap.add_argument("--include-random", action="store_true",
                    help="--all-combos에 random z_source 포함 (학습 default 동작 reproduction). "
                         "보통 평가/ablation에는 zimg vs ztxt 페어만으로 충분.")
    ap.add_argument("--seed", type=int, default=None,
                    help="val sample selection seed. 미지정 시 sequential (첫 n images). "
                         "지정 시 그 seed로 val image indices를 shuffle해 random pick.")
    args = ap.parse_args()

    run_dir = resolve(args.run_dir, HERE)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    img_cfg = cfg["image_decoder"]
    print(f"[load] epoch={ckpt['epoch']}, z_source={img_cfg['z_source']}, "
          f"text_decoder_run={img_cfg.get('text_decoder_run', '')!r}")

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    precision = img_cfg.get("precision", "bf16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    text_decoder_run = img_cfg.get("text_decoder_run", "") or ""
    caption_stream_on = bool(text_decoder_run) and not args.no_caption

    # Build model + load state
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
    print(f"[load] LoRA + z_proj state loaded")

    # Cache load (val split)
    cache_dir = resolve(cfg["cache_dir"], HERE)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)
    val_image_ids = idx["val"]["image_ids"]
    val_caption_image_idx = idx["val"]["caption_image_idx"]
    val_caption_texts = idx["val"]["caption_texts"]

    # Caption cache (optional)
    captions_zimg_val = captions_ztxt_val = None
    if caption_stream_on:
        prefix = f"captions_{text_decoder_run}"
        captions_zimg_val = torch.load(cache_dir / f"{prefix}_val_zimg.pt", weights_only=False)
        captions_ztxt_val = torch.load(cache_dir / f"{prefix}_val_ztxt.pt", weights_only=False)

    # Select N val images — sequential (default) or random (--seed)
    coco_val = resolve(cfg["coco_root"], HERE) / "images" / "val2017"
    transform = make_pixel_transform()

    # Map image_idx → first caption_idx (sequential lookup). 학습/추론 양쪽에서 같은
    # 캡션 인덱스를 쓰기 위해 image마다 "첫 번째 caption"을 convention으로 고정.
    first_cap_for_image: dict[int, int] = {}
    for cap_idx, im_idx in enumerate(val_caption_image_idx):
        if im_idx not in first_cap_for_image:
            first_cap_for_image[im_idx] = cap_idx

    available_image_indices = sorted(first_cap_for_image.keys())
    if args.seed is not None:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(available_image_indices), generator=g).tolist()
        selected_image_indices = [available_image_indices[p] for p in perm[:args.n]]
        print(f"[select] random (seed={args.seed}) → image indices: {selected_image_indices}")
    else:
        selected_image_indices = available_image_indices[:args.n]
        print(f"[select] sequential → image indices: {selected_image_indices}")

    xs, z_imgs, z_txts, gts, zimg_caps, ztxt_caps = [], [], [], [], [], []
    for i in selected_image_indices:
        j = first_cap_for_image[i]
        iid = val_image_ids[i]
        img = Image.open(coco_val / f"{iid:012d}.jpg").convert("RGB")
        xs.append(transform(img))
        z_imgs.append(z_img_val[i].float())
        z_txts.append(z_txt_val[j].float())
        gts.append(val_caption_texts[j])
        zimg_caps.append(captions_zimg_val[i] if caption_stream_on else "")
        ztxt_caps.append(captions_ztxt_val[j] if caption_stream_on else "")

    n_actual = len(xs)
    x_gt = torch.stack(xs).to(device).to(dtype)
    z_img_b = torch.stack(z_imgs).to(device)
    z_txt_b = torch.stack(z_txts).to(device)

    # DDIM scheduler (한 번만 로드)
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler.from_pretrained(model.sd_model_id, subfolder="scheduler")
    ddim.set_timesteps(args.steps)

    def build_z_and_caps(zsrc: str):
        """zsrc에 맞게 z tensor + caption list 반환."""
        if zsrc == "zimg":
            z_in = z_img_b
            caps = list(zimg_caps)
        elif zsrc == "ztxt":
            z_in = z_txt_b
            caps = list(ztxt_caps)
        elif zsrc == "centroid":
            z_in = torch.nn.functional.normalize((z_img_b + z_txt_b) / 2.0, dim=-1)
            caps = list(gts)  # centroid는 별도 cache 없음 → GT 사용
        else:  # random
            mask = (torch.rand(n_actual, 1, device=device) < 0.5)
            z_in = torch.where(mask, z_img_b, z_txt_b)
            caps = [zimg_caps[k] if mask[k].item() else ztxt_caps[k]
                    for k in range(n_actual)]
        # caption overrides
        if args.use_gt_caption:
            caps = list(gts)
        if args.no_caption or not caption_stream_on:
            caps = None
        return z_in, caps

    def run_inference(z_in: torch.Tensor, captions_in, cfg_scale: float):
        x_hats = []
        bs = args.batch_size
        for s in range(0, n_actual, bs):
            zb = z_in[s:s + bs]
            cb = captions_in[s:s + bs] if captions_in is not None else None
            with torch.no_grad():
                cond = model.build_condition(zb, cb)
                if cfg_scale != 1.0:
                    uncond = torch.zeros_like(cond)
                latent = torch.randn(zb.size(0), 4, 28, 28, device=device, dtype=dtype)
                for t in ddim.timesteps:
                    ts = torch.tensor([t.item()] * zb.size(0), device=device).long()
                    eps_c = model.unet(latent, ts, encoder_hidden_states=cond).sample
                    if cfg_scale != 1.0:
                        eps_u = model.unet(latent, ts, encoder_hidden_states=uncond).sample
                        eps = eps_u + cfg_scale * (eps_c - eps_u)
                    else:
                        eps = eps_c
                    latent = ddim.step(eps, t, latent).prev_sample
                x_hat = model.decode_latent_to_image(latent)
            x_hats.append(x_hat)
        return torch.cat(x_hats, dim=0)

    cap_tag = "capON" if caption_stream_on else "capOFF"
    ckpt_tag = args.ckpt.replace(".pt", "")

    if args.all_combos:
        # Default 6 PNG: zimg/ztxt × cfg 1.0/3.0/7.5. random은 inference 시점에 평가 가치
        # 적어 default에서 제외 (학습 default 동작 reproduction이 필요하면 --include-random).
        # --seed 지정 시 파일명에 _seed{N} suffix를 붙여 sequential 결과를 덮어쓰지 않음.
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        z_sources_to_run = ["zimg", "ztxt"]
        if args.include_random:
            z_sources_to_run = ["random"] + z_sources_to_run
        n_combos = len(z_sources_to_run) * 3
        out_dir = run_dir / "samples" / ckpt_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sample] all_combos mode — {n_combos} PNGs → {out_dir}")
        for zsrc in z_sources_to_run:
            for cfg_scale in (10, 3.0, 7.5):
                z_in, captions_in = build_z_and_caps(zsrc)
                x_hat = run_inference(z_in, captions_in, cfg_scale)
                # ztxt: z의 source인 GT caption(=gts) + decoded caption(=captions_in) 둘 다 표시
                # zimg/random: z의 source가 caption이 아니므로 decoded만 표시
                gt_caps_for_panel = gts if zsrc == "ztxt" else None
                out_path = out_dir / f"recon_{zsrc}_cfg{cfg_scale}{seed_suffix}.png"
                save_grid_with_captions(
                    x_gt, x_hat, gt_caps_for_panel, captions_in, zsrc, out_path,
                )
                print(f"  [save] {out_path.name}")
        print(f"[done] {n_combos} grids saved under {out_dir}")
        return

    # Single inference (기존 동작)
    z_in, captions_in = build_z_and_caps(args.z_source)
    print(f"[sample] N={n_actual}, z_source={args.z_source}, caption_stream={caption_stream_on}, "
          f"steps={args.steps}, cfg_scale={args.cfg_scale}")
    for k in range(min(3, n_actual)):
        print(f"  [{k}] GT: {gts[k]}")
        if captions_in is not None:
            print(f"      cap_in: {captions_in[k]}")

    x_hat_all = run_inference(z_in, captions_in, args.cfg_scale)
    if args.output:
        out_path = Path(args.output)
    else:
        sample_dir = run_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"infer_{ckpt_tag}_{args.z_source}_{cap_tag}_cfg{args.cfg_scale}.png"
    # ztxt zsrc일 때만 GT caption(=z source) 표시. 그 외(zimg, random, centroid)는 decoded만.
    gt_caps_for_panel = gts if args.z_source == "ztxt" else None
    save_grid_with_captions(
        x_gt, x_hat_all, gt_caps_for_panel, captions_in, args.z_source, out_path,
    )
    print(f"[save] {out_path} (top={n_actual} GT, bottom={n_actual} recon)")


if __name__ == "__main__":
    main()
