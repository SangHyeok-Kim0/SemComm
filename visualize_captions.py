"""val 이미지 + GT 5개 + GEN 1개 시각화.

기본은 5종 한 번에 (image / text_{mean,random} / centroid_{mean,random}) + 합본 1장.
--z-source: image=z_img(cross-modal swap, SC 핵심), text=z_txt, centroid=둘의 평균.

Usage:
  python visualize_captions.py --run-dir runs/txt_modality_K10_ep10_...
  python visualize_captions.py --run-dir <dir> --n 10 --z-source image --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import TextDecoder  # noqa: E402


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def apply_awgn(z: torch.Tensor, snr_db: float | None,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """unit-norm z 위에 AWGN 추가 후 re-normalize. Phase A convention과 동일.
    SNR_dB = -10·log10(D·σ²) → σ = sqrt(10^(-snr/10) / D).
    snr_db=None / inf → noise=0, z 그대로.
    """
    if snr_db is None or snr_db == float("inf"):
        return z
    D = z.size(-1)
    sigma = (10.0 ** (-snr_db / 10.0) / D) ** 0.5
    n = torch.randn(z.shape, generator=generator, device=z.device,
                    dtype=z.dtype) * sigma
    return F.normalize(z + n, dim=-1)


def build_z(z_img_val: torch.Tensor, z_txt_val: torch.Tensor,
            per_img_cap_indices: list[list[int]], img_indices: list[int],
            z_source: str, cap_agg: str, rng: random.Random
            ) -> tuple[torch.Tensor, list[list[int]]]:
    """이미지당 한 개 z 만들기.

    한 이미지에 caption이 5개라 text/centroid는 caption-side z를
    어떻게 모을지 선택해야 함:
      cap_agg='mean'   : 5 z_txt 평균 후 unit-norm (이미지의 text-side prototype)
      cap_agg='random' : caption 1개 무작위 선택 (rng으로 결정론적; 선택된 caption만 표시)
    image 모드에서는 무시 (이미지당 z 하나; 5 caption 모두 표시).

    Returns:
        z         : (n, D) tensor
        shown_cap : 이미지별 표시할 caption index 리스트
                    — random은 1개, mean/image는 전체 5개 (rendering이 GT 표시에 사용)
    """
    out, shown = [], []
    for i in img_indices:
        z_v = z_img_val[i].float()
        cap_idx = per_img_cap_indices[i]
        if z_source == "image":
            z = z_v
            shown.append(cap_idx)
        else:
            if cap_agg == "random":
                j = rng.choice(cap_idx)
                z_t = z_txt_val[j].float()
                shown.append([j])
            else:  # mean
                z_t = F.normalize(z_txt_val[cap_idx].float().mean(dim=0), dim=-1)
                shown.append(cap_idx)
            z = z_t if z_source == "text" else F.normalize((z_v + z_t) / 2.0, dim=-1)
        out.append(z)
    return torch.stack(out), shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20326)
    ap.add_argument("--z-source", default="all",
                    choices=["all", "image", "text", "centroid"],
                    help="all=5종 한 번에 (기본). image=z_img (cross-modal swap), "
                         "text=z_txt, centroid=평균")
    ap.add_argument("--output", default=None,
                    help="PNG path. 단일 조합일 때만 적용. all 모드면 무시되고 "
                         "<run-dir>/results/visualization/visualize_<suffix>.png 5개로 저장.")
    ap.add_argument("--no-clean", action="store_true",
                    help="첫 문장 cut 끄기 — raw beam output 그대로 (max_new_tokens 끝까지)")
    ap.add_argument("--cap-agg", default="mean", choices=["mean", "random"],
                    help="text/centroid 시 이미지의 5 caption z를 어떻게 합칠지. "
                         "mean=평균(prototype, GT 5개 표시) / random=무작위 1개(선택된 GT만 표시)")
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="공유 cfg (cache_dir, coco_root) 읽기용. run-dir/config.json 우선.")
    ap.add_argument("--noise-snr", type=float, default=None,
                    help="AWGN inject SNR_dB. None=무노이즈. 예: 10, 0, -5. "
                         "Phase A 결과 ~+10 dB 위에선 거의 무손실, 0 부근부터 무너짐.")
    args = ap.parse_args()

    run_dir = resolve_path(args.run_dir, HERE)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg["text_decoder"]

    device = torch.device(f"cuda:{text_cfg['device_id']}" if torch.cuda.is_available() else "cpu")

    # 1. Model
    print("[model] loading ...")
    model = TextDecoder(
        lm_name=text_cfg["lm_name"],
        z_dim=1024,
        prefix_len=text_cfg["prefix_len"],
        mapper_type=text_cfg["mapper_type"],
        mapper_layers=text_cfg["mapper_layers"],
        mapper_heads=text_cfg["mapper_heads"],
        mapper_dropout=text_cfg["mapper_dropout"],
        freeze_lm=text_cfg["freeze_lm"],
        lm_dtype=text_cfg.get("lm_dtype", "auto"),
    ).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / args.ckpt, map_location=device,
                      weights_only=False)
    model.mapper.load_state_dict(ckpt["mapper"])
    model.eval()

    # 2. Cache + index
    cache_dir = resolve_path(cfg["cache_dir"], HERE)
    coco_root = resolve_path(cfg["coco_root"], HERE)
    val_img_dir = coco_root / "images" / "val2017"

    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)
    val = idx["val"]
    image_ids = val["image_ids"]
    cap_img_idx = val["caption_image_idx"]
    cap_texts = val["caption_texts"]

    per_img_cap_idx: list[list[int]] = [[] for _ in image_ids]
    for j, im_idx in enumerate(cap_img_idx):
        per_img_cap_idx[im_idx].append(j)

    # 3. Combos: 'all'은 5종, 아니면 단일 조합만.
    if args.z_source == "all":
        combos = [("image", "mean"),
                  ("text", "mean"), ("text", "random"),
                  ("centroid", "mean"), ("centroid", "random")]
    else:
        combos = [(args.z_source, args.cap_agg)]

    # all 모드면 5개 GEN을 한 PNG에 합칠 때 쓰일 누적 buffers.
    all_gens: dict[tuple, list[str]] = {}
    random_caps: dict[int, str] = {}     # image idx → random에서 뽑힌 caption text
    last_sample_indices: list[int] = []

    for z_source, cap_agg in combos:
        # 매 combo마다 rng 새로 시드 → image 선택과 random caption 선택 모두 일관
        rng = random.Random(args.seed)
        n = min(args.n, len(image_ids))
        sample_indices = rng.sample(range(len(image_ids)), n)
        last_sample_indices = sample_indices

        z, shown_caps = build_z(z_img_val, z_txt_val, per_img_cap_idx, sample_indices,
                                z_source, cap_agg=cap_agg, rng=rng)
        z = z.to(device)
        if args.noise_snr is not None:
            # noise rng도 결정론 — combo마다 같은 noise pattern (공정 비교).
            torch_rng = torch.Generator(device=device).manual_seed(args.seed)
            z = apply_awgn(z, args.noise_snr, generator=torch_rng)
        print(f"[gen] z_source={z_source}, cap_agg={cap_agg}, n={n}, "
              f"beam={text_cfg['beam_size']}, max_new={text_cfg['max_new_tokens']}, "
              f"snr_db={args.noise_snr}")
        gens = model.generate(z, beam_size=text_cfg["beam_size"],
                              max_new_tokens=text_cfg["max_new_tokens"],
                              clean=not args.no_clean)
        all_gens[(z_source, cap_agg)] = gens
        if cap_agg == "random":
            for r, sc in enumerate(shown_caps):
                random_caps[sample_indices[r]] = cap_texts[sc[0]]

        fig, axes = plt.subplots(n, 2, figsize=(20, 3.8 * n),
                                 gridspec_kw={"width_ratios": [1, 3]})
        if n == 1:
            axes = axes[None, :]
        for r in range(n):
            i = sample_indices[r]
            iid = image_ids[i]
            img = Image.open(val_img_dir / f"{iid:012d}.jpg").convert("RGB")
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"id={iid}", fontsize=12)
            axes[r, 0].axis("off")

            gt_caps = [cap_texts[j] for j in shown_caps[r]]
            gt_block = "\n".join(f"  • {c}" for c in gt_caps)
            gt_label = "GT" if len(gt_caps) > 1 else "GT (selected for z)"
            text = (f"{gt_label}:\n{gt_block}\n\n"
                    f"GEN ({z_source}):\n  → {gens[r]}")
            axes[r, 1].text(0.0, 0.5, text, va="center", fontsize=18,
                            family="monospace", wrap=True, linespacing=0.95,
                            transform=axes[r, 1].transAxes)
            axes[r, 1].axis("off")

        suffix = z_source if z_source == "image" else f"{z_source}_{cap_agg}"
        snr_tag = f"_snr{args.noise_snr:g}" if args.noise_snr is not None else ""
        out = (Path(args.output) if (args.output and len(combos) == 1)
               else run_dir / "results" / "visualization" / f"visualize_{suffix}{snr_tag}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] {out}")

    # all 모드 → 5개 GEN을 한 PNG로 합쳐 추가 저장. random에서 뽑힌 GT는 빨간색.
    if args.z_source == "all":
        snr_tag = f"_snr{args.noise_snr:g}" if args.noise_snr is not None else ""
        render_combined(run_dir, last_sample_indices, image_ids, val_img_dir,
                        per_img_cap_idx, cap_texts, all_gens, random_caps,
                        snr_tag=snr_tag, snr_db=args.noise_snr)


def render_combined(run_dir: Path, sample_indices: list[int],
                    image_ids: list[int], val_img_dir: Path,
                    per_img_cap_idx: list[list[int]], cap_texts: list[str],
                    all_gens: dict, random_caps: dict[int, str],
                    snr_tag: str = "", snr_db: float | None = None) -> None:
    """5종 GEN 결과를 한 PNG에. row마다 image + GT 5개 + GEN 5개 표시.
    random에서 z 만들 때 뽑힌 GT만 빨간 글씨.
    """
    n = len(sample_indices)
    combo_labels = [
        (("image", "mean"),       "image           "),
        (("text", "mean"),        "text_mean       "),
        (("text", "random"),      "text_random     "),
        (("centroid", "mean"),    "centroid_mean   "),
        (("centroid", "random"),  "centroid_random "),
    ]
    fig, axes = plt.subplots(n, 2, figsize=(22, 5.5 * n),
                             gridspec_kw={"width_ratios": [1, 4]})
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        i = sample_indices[r]
        iid = image_ids[i]
        img = Image.open(val_img_dir / f"{iid:012d}.jpg").convert("RGB")
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"id={iid}", fontsize=12)
        axes[r, 0].axis("off")

        # (text, color) 라인 시퀀스 — random 선택 GT만 red.
        lines: list[tuple[str, str]] = [("GT:", "black")]
        sel = random_caps.get(i)
        for j in per_img_cap_idx[i]:
            cap = cap_texts[j]
            lines.append((f"  • {cap}", "red" if cap == sel else "black"))
        lines.append(("", "black"))
        lines.append(("GEN:", "black"))
        for key, label in combo_labels:
            lines.append((f"  {label}: → {all_gens[key][r]}", "black"))

        ax = axes[r, 1]
        ax.axis("off")
        # 텍스트 블록 중심을 y=0.5로 정렬 → 왼쪽 이미지의 세로 중심과 일치.
        line_h = 0.05
        y_top = 0.5 + ((len(lines) - 1) / 2) * line_h
        for idx, (txt, color) in enumerate(lines):
            y = y_top - idx * line_h
            ax.text(0.0, y, txt, color=color, fontsize=18,
                    family="monospace", ha="left", va="center",
                    transform=ax.transAxes)

    if snr_db is not None:
        fig.suptitle(f"AWGN inject @ SNR = {snr_db:g} dB",
                     fontsize=20, y=0.995)
    out = run_dir / "results" / "visualization" / f"visualize_combined{snr_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
