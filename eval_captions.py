"""COCO val 캡션 정량 평가 — 학습된 text decoder의 5 z_source 시나리오 비교.

메트릭: BLEU-1..4 + ROUGE-L + CIDEr (pycocoevalcap, Python-only 경로 — Java 미사용).
시나리오: image (cross-modal swap), text_{mean,random}, centroid_{mean,random}.

Usage:
  python eval_captions.py --run-dir runs/txt_modality_K10_ep10_<ts> [--n 500]

산출:
  <run-dir>/results/metric/eval_metrics.json   (시나리오별 메트릭 + 모든 GT/GEN)
  stdout markdown 표 요약
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import TextDecoder  # noqa: E402
from visualize_captions import apply_awgn, build_z, resolve_path  # noqa: E402


COMBOS = [
    ("image",    "mean"),
    ("text",     "mean"),
    ("text",     "random"),
    ("centroid", "mean"),
    ("centroid", "random"),
]
SCENARIO_KEYS = [s if a == "mean" or s == "image" else f"{s}_{a}"
                 for s, a in []]  # placeholder; computed below per loop


def scenario_key(z_source: str, cap_agg: str) -> str:
    """image는 cap_agg 무관 → 'image'. 그 외는 '<source>_<agg>'."""
    return z_source if z_source == "image" else f"{z_source}_{cap_agg}"


# ---------------------------------------------------------------------------
# Asset loading

def load_run_assets(run_dir: Path, ckpt_name: str, device: torch.device,
                    cache_dir: Path, coco_root: Path):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg["text_decoder"]

    # 새 구조: checkpoints/models/<ckpt>, 옛 구조: checkpoints/<ckpt> 둘 다 지원.
    ckpt_path = run_dir / "checkpoints" / "models" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / ckpt_name
    print(f"[model] loading from {ckpt_path} ...")
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.mapper.load_state_dict(ckpt["mapper"])
    model.eval()

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

    return (model, text_cfg, cfg, z_img_val, z_txt_val, image_ids,
            per_img_cap_idx, cap_texts)


# ---------------------------------------------------------------------------
# Generation

@torch.no_grad()
def batched_generate(model: TextDecoder, z: torch.Tensor, batch_size: int,
                     beam_size: int, max_new_tokens: int, clean: bool,
                     desc: str) -> list[str]:
    out: list[str] = []
    for s in tqdm(range(0, z.size(0), batch_size), desc=desc):
        chunk = z[s:s + batch_size]
        out.extend(model.generate(chunk, beam_size=beam_size,
                                  max_new_tokens=max_new_tokens,
                                  clean=clean))
    return out


# ---------------------------------------------------------------------------
# Metrics

_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """PTBTokenizer (Java) 대체 — lowercase + non-word strip + whitespace 정리."""
    s = _NORM_RE.sub(" ", s.lower().strip())
    return _WS_RE.sub(" ", s).strip()


def build_gt_refs(per_img_cap_idx: list[list[int]], cap_texts: list[str],
                  sample_indices: list[int]) -> dict[int, list[str]]:
    """이미지마다 GT caption 5개 (또는 그 이하)를 normalize해서 모음.
    key는 sample 내 행 번호 r — gens dict와 같은 key를 쓰면 metric scorer가 매칭한다.
    """
    refs: dict[int, list[str]] = {}
    for r, i in enumerate(sample_indices):
        refs[r] = [normalize(cap_texts[j]) for j in per_img_cap_idx[i]]
    return refs


def build_gens_dict(gens: list[str]) -> dict[int, list[str]]:
    """행 번호 r → [normalize(gen)] 한 리스트."""
    return {r: [normalize(g)] for r, g in enumerate(gens)}


def compute_metrics(gts: dict[int, list[str]],
                    res: dict[int, list[str]]) -> dict[str, float]:
    """pycocoevalcap의 Java-free scorer만 사용 (Bleu / Rouge / Cider)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    out: dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts, res)
    for i, b in enumerate(bleu, start=1):
        out[f"BLEU-{i}"] = float(b)
    rouge, _ = Rouge().compute_score(gts, res)
    out["ROUGE-L"] = float(rouge)
    cider, _ = Cider().compute_score(gts, res)
    out["CIDEr"] = float(cider)
    return out


# ---------------------------------------------------------------------------
# CLIPScore (semantic) — standard CLIP image-text cosine.
# eval_image.py와 동일하게 raw cosine 사용 (clamp/scale 없음).
# 학습된 encoder가 아닌 별도 OpenAI CLIP을 metric으로 사용 → fair.

_CLIP_BUNDLE = None  # (model, preprocess, tokenizer, device)


def load_clip_metric(device: torch.device,
                     model_name: str = "ViT-B-32",
                     pretrained: str = "openai"):
    global _CLIP_BUNDLE
    if _CLIP_BUNDLE is not None:
        return _CLIP_BUNDLE
    import open_clip
    print(f"[clip] loading {model_name} ({pretrained}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    _CLIP_BUNDLE = (model, preprocess, tokenizer, device)
    return _CLIP_BUNDLE


@torch.no_grad()
def precompute_clip_img_feats(image_ids: list[int], sample_indices: list[int],
                              coco_root: Path, device: torch.device,
                              batch_size: int = 64) -> torch.Tensor:
    """sample_indices에 해당하는 COCO val image들을 standard CLIP encoder로 인코딩 (한 번만)."""
    model, preprocess, _, _ = load_clip_metric(device)
    val_dir = coco_root / "images" / "val2017"
    imgs = []
    for i in tqdm(sample_indices, desc="clip img enc"):
        iid = image_ids[i]
        img = Image.open(val_dir / f"{iid:012d}.jpg").convert("RGB")
        imgs.append(preprocess(img))
    feats = []
    for s in range(0, len(imgs), batch_size):
        batch = torch.stack(imgs[s:s + batch_size]).to(device)
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def compute_clip_score(img_feats: torch.Tensor, captions: list[str],
                       device: torch.device, batch_size: int = 256) -> float:
    """img_feats: (N, D) normalized. Generated caption별 cosine mean."""
    model, _, tokenizer, _ = load_clip_metric(device)
    txt_feats = []
    for s in range(0, len(captions), batch_size):
        tokens = tokenizer(captions[s:s + batch_size]).to(device)
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        txt_feats.append(f.float())
    txt_feats = torch.cat(txt_feats, dim=0)
    return (img_feats * txt_feats).sum(-1).mean().item()


# ---------------------------------------------------------------------------
# Output formatting

METRIC_ORDER = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr", "CLIPScore"]


def format_markdown_table(metrics_by_scenario: dict[str, dict[str, float]]) -> str:
    cols = ["scenario"] + METRIC_ORDER
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = [header, sep]
    for sc, m in metrics_by_scenario.items():
        cells = [sc] + [f"{m.get(k, float('nan')):.4f}" for k in METRIC_ORDER]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n", type=int, default=None,
                    help="evaluate on first N val images (default: all 5000)")
    ap.add_argument("--scenarios", nargs="+",
                    default=["text_random"],
                    choices=["image", "text_mean", "text_random",
                            "centroid_mean", "centroid_random"])
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-clean", action="store_true",
                    help="첫 문장 cut 끄기 (raw beam output)")
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="공유 cfg (cache_dir, coco_root)")
    ap.add_argument("--cache-dir", default=None,
                    help="override cfg['cache_dir']. text decoder가 학습된 cache와 동일해야 함.")
    ap.add_argument("--output", default=None,
                    help="JSON 경로 (default: <run-dir>/results/metric/eval_metrics[_snr<x>].json)")
    ap.add_argument("--noise-snr", type=float, nargs="+", default=None,
                    help="AWGN inject SNR_dB. 단일 또는 sweep. "
                         "예: --noise-snr 10 0 -10 -15. 미지정 → 무노이즈 1회.")
    ap.add_argument("--no-clip-score", action="store_true",
                    help="skip CLIPScore (semantic) computation")
    ap.add_argument("--clip-model", default="ViT-B-32",
                    help="open_clip model for CLIPScore metric (eval_image.py와 동일)")
    ap.add_argument("--clip-pretrained", default="openai",
                    help="open_clip pretrained tag")
    args = ap.parse_args()
    # noise zero(None) baseline은 항상 측정 — SNR sweep과 비교 기준.
    snr_list: list[float | None] = [None]
    if args.noise_snr:
        snr_list += [s for s in args.noise_snr if s is not None]

    run_dir = resolve_path(args.run_dir, HERE)
    with open(args.config) as f:
        import yaml
        shared_cfg = yaml.safe_load(f)
    cache_dir = resolve_path(args.cache_dir if args.cache_dir else shared_cfg["cache_dir"], HERE)
    print(f"[cache_dir] {cache_dir}")
    coco_root = resolve_path(shared_cfg["coco_root"], HERE)

    # Device — text_cfg에서 device_id 가져옴
    with open(run_dir / "config.json") as f:
        device_id = json.load(f)["text_decoder"]["device_id"]
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    (model, text_cfg, cfg, z_img_val, z_txt_val, image_ids,
     per_img_cap_idx, cap_texts) = load_run_assets(
        run_dir, args.ckpt, device, cache_dir, coco_root)

    # 평가할 이미지 인덱스 — 결정론적으로 첫 N개.
    n = args.n if args.n is not None else len(image_ids)
    n = min(n, len(image_ids))
    sample_indices = list(range(n))
    print(f"[data] evaluating on first {n} val images "
          f"(of {len(image_ids)} total)")

    gts = build_gt_refs(per_img_cap_idx, cap_texts, sample_indices)

    # CLIPScore용 image feature 사전 계산 (한 번만, SNR/scenario 무관).
    clip_img_feats = None
    if not args.no_clip_score:
        clip_img_feats = precompute_clip_img_feats(
            image_ids, sample_indices, coco_root, device)
        print(f"[clip] img feats: {tuple(clip_img_feats.shape)}")

    # SNR sweep: 각 SNR마다 5 시나리오 돌리고 별도 JSON 저장.
    # snr→scenario→metrics 누적 (sweep 끝나면 PNG plot 생성).
    sweep_results: dict[float | None, dict[str, dict[str, float]]] = {}
    for snr_db in snr_list:
        snr_tag = f"_snr{snr_db:g}" if snr_db is not None else ""
        print(f"\n=== SNR = {snr_db if snr_db is not None else 'no noise'} ===")
        metrics_by_scenario: dict[str, dict[str, float]] = {}
        predictions_by_scenario: dict[str, list[dict]] = {}

        t0_total = time.time()
        for z_source, cap_agg in COMBOS:
            sc_key = scenario_key(z_source, cap_agg)
            if sc_key not in args.scenarios:
                continue

            # rng 매번 새로 시드 → 결정론적 image 선택 + random caption 선택
            rng = random.Random(args.seed)
            z, _shown = build_z(z_img_val, z_txt_val, per_img_cap_idx,
                                sample_indices, z_source, cap_agg=cap_agg, rng=rng)
            z = z.to(device)
            if snr_db is not None:
                # noise rng도 결정론 — combo마다 같은 noise pattern (공정 비교).
                torch_rng = torch.Generator(device=device).manual_seed(args.seed)
                z = apply_awgn(z, snr_db, generator=torch_rng)

            t0 = time.time()
            gens = batched_generate(
                model, z, batch_size=args.batch_size,
                beam_size=text_cfg["beam_size"],
                max_new_tokens=text_cfg["max_new_tokens"],
                clean=not args.no_clean,
                desc=f"gen {sc_key}{snr_tag}",
            )
            gen_time = time.time() - t0

            res = build_gens_dict(gens)
            metrics = compute_metrics(gts, res)
            if clip_img_feats is not None:
                metrics["CLIPScore"] = compute_clip_score(
                    clip_img_feats, gens, device)
            metrics_by_scenario[sc_key] = metrics
            predictions_by_scenario[sc_key] = [
                {"image_idx": i, "image_id": image_ids[i],
                 "gt": [cap_texts[j] for j in per_img_cap_idx[i]],
                 "gen": gens[r]}
                for r, i in enumerate(sample_indices)
            ]
            print(f"  [{sc_key}] {gen_time:.1f}s | " +
                  " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        print(f"\n[total snr={snr_db}] {time.time() - t0_total:.1f}s")
        table = format_markdown_table(metrics_by_scenario)
        print("\n" + table + "\n")

        # JSON 저장 (SNR마다 별도 파일)
        if args.output and len(snr_list) == 1:
            out_path = Path(args.output)
        else:
            out_path = run_dir / "results" / "metric" / f"eval_metrics{snr_tag}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_dir": str(run_dir),
            "ckpt": args.ckpt,
            "n_images": n,
            "config": {
                "beam_size": text_cfg["beam_size"],
                "max_new_tokens": text_cfg["max_new_tokens"],
                "seed": args.seed,
                "clean": not args.no_clean,
                "batch_size": args.batch_size,
                "noise_snr_db": snr_db,
            },
            "metrics_by_scenario": metrics_by_scenario,
            "predictions_by_scenario": predictions_by_scenario,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[done] {out_path}")
        sweep_results[snr_db] = metrics_by_scenario

    # SNR sweep plot — 2개 이상 SNR 측정 시 자동 PNG 생성.
    numeric_snrs = sorted(s for s in sweep_results if s is not None)
    if len(numeric_snrs) >= 2:
        import matplotlib.pyplot as plt
        plot_dir = run_dir / "results" / "metric"
        plot_dir.mkdir(parents=True, exist_ok=True)
        # 1x4 subplot — 메트릭별로 독립 axis.
        panels = [("BLEU-1", "tab:blue"), ("BLEU-3", "tab:green"),
                  ("CIDEr", "crimson"), ("CLIPScore", "tab:purple")]
        baseline = sweep_results.get(None)
        for sc in args.scenarios:
            if not all(sc in sweep_results[s] for s in numeric_snrs):
                continue
            fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
            for ax, (metric, color) in zip(axes, panels):
                ys = [sweep_results[s][sc].get(metric, float("nan")) for s in numeric_snrs]
                if all(y != y for y in ys):  # all NaN → skip
                    ax.set_title(f"{metric} (N/A)")
                    continue
                ax.plot(numeric_snrs, ys, marker="o", color=color, linewidth=2)
                if baseline and sc in baseline and metric in baseline[sc]:
                    ax.axhline(y=baseline[sc][metric], color=color,
                               linestyle="--", alpha=0.5, linewidth=1,
                               label="no-noise")
                    ax.legend(loc="best", fontsize=8)
                ax.set_xlabel("SNR (dB)")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"SNR sweep — {sc} (n={n}, dashed = no-noise)")
            fig.tight_layout()
            out_png = plot_dir / f"snr_sweep_{sc}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] {out_png}")


if __name__ == "__main__":
    main()
