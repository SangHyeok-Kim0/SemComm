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
                            wrap_width: int = 34, max_cols: int = 7) -> None:
    """matplotlib으로 GT row + recon row + caption text를 합쳐 한 PNG로 저장.

    zsrc == "ztxt": GT caption(z_txt의 source)과 decoded caption(captions_ztxt[j]) 둘 다 표시.
    zsrc == "zimg" / others: z 자체에 source caption이 없으므로 decoded caption만 표시.
    decoded_captions가 None이면 caption 라벨 생략 (caption stream OFF 시).

    n > max_cols인 경우 (GT, Recon) 2-row 그룹을 여러 번 쌓아 아래로 wrap.
    """
    import numpy as np
    n = x_gt.size(0)
    n_groups = (n + max_cols - 1) // max_cols
    cols = min(n, max_cols)
    # Nested gridspec.
    # 핵심: set_box_aspect(1)로 axes를 강제 정사각형 → imshow 자동 padding 제거 →
    #       hspace/wspace 값이 실제 visible spacing에 직결됨.
    # figsize: 각 axes ≈ 2.4" 정사각 + caption 영역. group 높이 ≈ 2*2.4 + 0.7 ≈ 5.5".
    fig = plt.figure(figsize=(cols * 2.4, n_groups * 5.5))
    outer_gs = fig.add_gridspec(n_groups, 1, hspace=0.23)
    axes = np.empty((2 * n_groups, cols), dtype=object)
    for g in range(n_groups):
        inner_gs = outer_gs[g].subgridspec(2, cols, hspace=0.02, wspace=0.02)
        for r in range(2):
            for c in range(cols):
                ax = fig.add_subplot(inner_gs[r, c])
                ax.set_box_aspect(1)
                axes[2 * g + r, c] = ax
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
            lines.append(f"GT caption:\n{wrapped_gt}")
        if decoded_captions is not None:
            wrapped_dec = "\n".join(textwrap.wrap(decoded_captions[k], width=wrap_width)) or "(empty)"
            lines.append(f"Decoded caption:\n{wrapped_dec}")
        caption_text = "\n\n".join(lines) if lines else ""
        if caption_text:
            axes[recon_row, col].set_xlabel(caption_text, fontsize=7.5)
    # 마지막 group에서 남는 빈 칸 숨기기
    last_group_filled = n - (n_groups - 1) * max_cols
    if last_group_filled < cols:
        for col in range(last_group_filled, cols):
            axes[2 * (n_groups - 1), col].axis("off")
            axes[2 * (n_groups - 1) + 1, col].axis("off")
    for g in range(n_groups):
        axes[2 * g, 0].set_ylabel("GT", fontsize=10)
        axes[2 * g + 1, 0].set_ylabel("Recon", fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_grid_paired_comparison(
    x_gt: torch.Tensor,
    x_hat_zimg: torch.Tensor, x_hat_ztxt: torch.Tensor,
    gt_captions: list[str] | None,
    decoded_zimg: list[str] | None, decoded_ztxt: list[str] | None,
    out_path: Path, wrap_width: int = 36, max_cols: int = 6,
) -> None:
    """한 column = GT / zimg recon / ztxt recon (3 row). caption text 는 column 하단에
    GT caption + Decoded(zimg) + Decoded(ztxt) 셋을 함께 표시. zimg vs ztxt 의 cross-modal
    swap 효과를 한 PNG에서 직접 비교할 수 있게 만든 layout."""
    import numpy as np
    n = x_gt.size(0)
    n_groups = (n + max_cols - 1) // max_cols
    cols = min(n, max_cols)
    # 3 row × cols. group 높이 = 3*2.4 + caption 영역 ≈ 8.0".
    fig = plt.figure(figsize=(cols * 2.4, n_groups * 8.0))
    outer_gs = fig.add_gridspec(n_groups, 1, hspace=0.30)
    axes = np.empty((3 * n_groups, cols), dtype=object)
    for g in range(n_groups):
        inner_gs = outer_gs[g].subgridspec(3, cols, hspace=0.02, wspace=0.02)
        for r in range(3):
            for c in range(cols):
                ax = fig.add_subplot(inner_gs[r, c])
                ax.set_box_aspect(1)
                axes[3 * g + r, c] = ax
    for k in range(n):
        group = k // max_cols
        col = k % max_cols
        row_base = 3 * group
        imgs = [
            denorm(x_gt[k]).float().cpu().permute(1, 2, 0).numpy(),
            denorm(x_hat_zimg[k]).float().cpu().permute(1, 2, 0).numpy(),
            denorm(x_hat_ztxt[k]).float().cpu().permute(1, 2, 0).numpy(),
        ]
        for r_off, im in enumerate(imgs):
            axes[row_base + r_off, col].imshow(im)
            axes[row_base + r_off, col].set_xticks([])
            axes[row_base + r_off, col].set_yticks([])
        lines = []
        if gt_captions is not None:
            wrapped = "\n".join(textwrap.wrap(gt_captions[k], width=wrap_width)) or "(empty)"
            lines.append(f"GT caption:\n{wrapped}")
        if decoded_zimg is not None:
            wrapped = "\n".join(textwrap.wrap(decoded_zimg[k], width=wrap_width)) or "(empty)"
            lines.append(f"Decoded (zimg):\n{wrapped}")
        if decoded_ztxt is not None:
            wrapped = "\n".join(textwrap.wrap(decoded_ztxt[k], width=wrap_width)) or "(empty)"
            lines.append(f"Decoded (ztxt):\n{wrapped}")
        caption_text = "\n\n".join(lines) if lines else ""
        if caption_text:
            axes[row_base + 2, col].set_xlabel(caption_text, fontsize=7.0)
    last_group_filled = n - (n_groups - 1) * max_cols
    if last_group_filled < cols:
        for col in range(last_group_filled, cols):
            for r_off in range(3):
                axes[3 * (n_groups - 1) + r_off, col].axis("off")
    for g in range(n_groups):
        axes[3 * g, 0].set_ylabel("GT", fontsize=10)
        axes[3 * g + 1, 0].set_ylabel("Recon\n(zimg)", fontsize=10)
        axes[3 * g + 2, 0].set_ylabel("Recon\n(ztxt)", fontsize=10)
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
                    help="Paired comparison 모드 — cfg 1.5/3.0/7.0 각각에 대해 한 PNG 생성 (총 3개). "
                         "각 column 은 GT / zimg recon / ztxt recon 3 row 로 구성되어 "
                         "cross-modal swap 효과를 직접 비교 가능. "
                         "samples/<ckpt>/paired_cfg{scale}.png 로 저장.")
    ap.add_argument("--include-random", action="store_true",
                    help="(legacy, paired 모드에서는 무시됨) 이전 z_source 분리 모드에서 random 포함용.")
    ap.add_argument("--seed", type=int, default=None,
                    help="val sample selection seed. 미지정 시 sequential (첫 n images). "
                         "지정 시 그 seed로 val image indices를 shuffle해 random pick.")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="diffusion inference 시 초기 latent noise + random z mask seed. "
                         "미지정 시 매 호출마다 다른 결과. 지정 시 같은 noise-seed 면 항상 "
                         "동일 recon (--all-combos 의 각 combo는 동일 seed 로 reset).")
    ap.add_argument("--coco-image-ids", type=int, nargs="+", default=None,
                    help="COCO image_id 를 직접 지정 (e.g. --coco-image-ids 397133 458755 ...). "
                         "지정 시 --seed/--n 무시. cache/index.json 의 val.image_ids 안에 "
                         "존재해야 함. 특정 이미지로 정성평가 / 재현 시 사용.")
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
    if args.coco_image_ids is not None:
        # User 가 직접 COCO image_id 지정 → val_image_ids 에서 internal idx 로 역매핑.
        coco_id_to_idx = {iid: i for i, iid in enumerate(val_image_ids)}
        missing = [iid for iid in args.coco_image_ids if iid not in coco_id_to_idx]
        if missing:
            raise ValueError(f"COCO image_id 가 val cache 에 없음: {missing}")
        selected_image_indices = [coco_id_to_idx[iid] for iid in args.coco_image_ids]
        # caption_image_idx 에 등장하지 않는 image (caption 없는 이미지) 는 추론 불가.
        no_cap = [iid for iid, i in zip(args.coco_image_ids, selected_image_indices)
                  if i not in first_cap_for_image]
        if no_cap:
            raise ValueError(f"caption 없는 COCO image_id (caption_image_idx 미등록): {no_cap}")
        print(f"[select] manual (--coco-image-ids) → image indices: {selected_image_indices}")
    elif args.seed is not None:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(available_image_indices), generator=g).tolist()
        selected_image_indices = [available_image_indices[p] for p in perm[:args.n]]
        print(f"[select] random (seed={args.seed}) → image indices: {selected_image_indices}")
    else:
        selected_image_indices = available_image_indices[:args.n]
        print(f"[select] sequential → image indices: {selected_image_indices}")
    # COCO image_id (실제 파일명 번호) 도 같이 출력 — image_idx 는 내부 인덱스라 파일 찾기 불편.
    selected_coco_ids = [val_image_ids[i] for i in selected_image_indices]
    print(f"[select] → COCO image_ids: {selected_coco_ids}")

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
        """CFG batching 적용: cond + uncond 를 한 batch 로 합쳐 UNet 1번 호출 (호출 수 절반).
        BS 는 2배로 늘지만 GB10 128GB 유니파이드 메모리에서 충분히 처리 가능."""
        x_hats = []
        bs = args.batch_size
        use_cfg = cfg_scale != 1.0
        for s in range(0, n_actual, bs):
            zb = z_in[s:s + bs]
            cb = captions_in[s:s + bs] if captions_in is not None else None
            with torch.no_grad():
                cond = model.build_condition(zb, cb)
                ctx = torch.cat([torch.zeros_like(cond), cond], dim=0) if use_cfg else cond
                latent = torch.randn(zb.size(0), 4, 28, 28, device=device, dtype=dtype)
                for t in ddim.timesteps:
                    ts = torch.tensor([t.item()] * zb.size(0), device=device).long()
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
            x_hats.append(x_hat)
        return torch.cat(x_hats, dim=0)

    cap_tag = "capON" if caption_stream_on else "capOFF"
    ckpt_tag = args.ckpt.replace(".pt", "")

    def seed_noise_rng():
        """noise-seed 가 지정된 경우 torch global RNG 를 그 seed 로 reset.
        run_inference 의 torch.randn(latent) + build_z_and_caps 의 torch.rand(z mask) 둘 다 영향."""
        if args.noise_seed is not None:
            torch.manual_seed(args.noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.noise_seed)

    if args.all_combos:
        # Paired comparison mode: 한 column = GT / zimg recon / ztxt recon. cfg 별 1 PNG.
        # 같은 cfg 안에서 zimg ↔ ztxt 가 동일 noise-seed 로 비교되어 cross-modal swap 효과를
        # 직접 비교 가능. 총 3 PNG (cfg 1.0/3.0/7.5). --include-random 은 이 모드에서 무시.
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        if args.noise_seed is not None:
            seed_suffix += f"_nseed{args.noise_seed}"
        out_dir = run_dir / "samples" / ckpt_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_scales = (3.0, 5.0, 7.0)
        print(f"[sample] paired mode — {len(cfg_scales)} PNGs (cfg {cfg_scales}) → {out_dir}")
        z_imgs_in, caps_zimg = build_z_and_caps("zimg")
        z_txts_in, caps_ztxt = build_z_and_caps("ztxt")
        for cfg_scale in cfg_scales:
            seed_noise_rng()
            x_hat_zimg = run_inference(z_imgs_in, caps_zimg, cfg_scale)
            seed_noise_rng()  # ztxt도 동일 noise 로 reset → 공정 비교
            x_hat_ztxt = run_inference(z_txts_in, caps_ztxt, cfg_scale)
            out_path = out_dir / f"paired{seed_suffix}_cfg{cfg_scale}.png"
            save_grid_paired_comparison(
                x_gt, x_hat_zimg, x_hat_ztxt,
                gts,  # z_txt 의 source GT caption (항상 표시)
                caps_zimg if caption_stream_on else None,
                caps_ztxt if caption_stream_on else None,
                out_path,
            )
            print(f"  [save] {out_path.name}")
        print(f"[done] {len(cfg_scales)} paired grids saved under {out_dir}")
        return

    # Single inference (기존 동작)
    seed_noise_rng()
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
