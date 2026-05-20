"""SNR sweep comparison plot with per-sample variance shading.

eval_metrics*.json에 저장된 predictions를 다시 읽어 per-sample metric을 재계산하고,
mean ± 1.96·SEM (95% CI of the mean) shaded band로 plot.

Usage:
  python plot_snr_compare_with_ci.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = [
    {
        "label": "CLIP-LT (gap≈0.45)",
        "color": "tab:blue",
        "marker": "o",
        "run_dir": HERE / "runs" / "txt_modality_bs64_lr2e-05_K10_std_clip_20260515-100435",
    },
    {
        "label": "Ours (gap≈0.037)",
        "color": "tab:red",
        "marker": "s",
        "run_dir": HERE / "runs" / "txt_modality_K10_ep10_20260506-135401",
    },
]


_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
def normalize(s: str) -> str:
    s = _NORM_RE.sub(" ", s.lower().strip())
    return _WS_RE.sub(" ", s).strip()


def per_sample_metrics(preds: list[dict]) -> dict[str, np.ndarray]:
    """Return per-image arrays for BLEU-1, BLEU-3, CIDEr."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    gts = {i: [normalize(g) for g in p["gt"]] for i, p in enumerate(preds)}
    res = {i: [normalize(p["gen"])] for i, p in enumerate(preds)}
    _, bleu_per = Bleu(4).compute_score(gts, res)
    _, cider_per = Cider().compute_score(gts, res)
    return {
        "BLEU-1": np.array(bleu_per[0]),
        "BLEU-3": np.array(bleu_per[2]),
        "CIDEr": np.asarray(cider_per),
    }


_CLIP = None
def clip_setup(device):
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _CLIP = (model, preprocess, tokenizer)
    return _CLIP


@torch.no_grad()
def encode_images(image_ids: list[int], coco_val: Path, device) -> torch.Tensor:
    model, preprocess, _ = clip_setup(device)
    imgs = []
    for iid in tqdm(image_ids, desc="clip img"):
        img = Image.open(coco_val / f"{iid:012d}.jpg").convert("RGB")
        imgs.append(preprocess(img))
    feats = []
    bs = 64
    for s in range(0, len(imgs), bs):
        batch = torch.stack(imgs[s:s + bs]).to(device)
        f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def per_sample_clip_score(img_feats: torch.Tensor, captions: list[str], device) -> np.ndarray:
    model, _, tokenizer = clip_setup(device)
    txt_feats = []
    bs = 256
    for s in range(0, len(captions), bs):
        tokens = tokenizer(captions[s:s + bs]).to(device)
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        txt_feats.append(f.float())
    txt_feats = torch.cat(txt_feats, dim=0)
    sims = (img_feats * txt_feats).sum(-1)
    return sims.cpu().numpy()


def collect_run(run_dir: Path, device) -> tuple[dict, dict]:
    """Returns ({snr: {metric: per_sample_array}}, baseline_dict)."""
    metric_dir = run_dir / "results" / "metric"
    by_snr: dict[float, dict[str, np.ndarray]] = {}
    baseline: dict[str, np.ndarray] = {}

    # Use first JSON to get image_ids (consistent across SNRs since seed/n fixed)
    first_json = next(metric_dir.glob("eval_metrics*.json"))
    preds_first = json.load(open(first_json))["predictions_by_scenario"]["text_random"]
    image_ids = [p["image_id"] for p in preds_first]

    # Encode images once (CLIP)
    coco_val = (HERE / "../ModalityGap/data/coco/images/val2017").resolve()
    print(f"  [clip] encoding {len(image_ids)} images from {coco_val}")
    img_feats = encode_images(image_ids, coco_val, device)

    for f in sorted(metric_dir.glob("eval_metrics*.json")):
        d = json.load(open(f))
        preds = d["predictions_by_scenario"].get("text_random")
        if preds is None:
            continue
        snr = d["config"].get("noise_snr_db")
        print(f"  [{f.name}] snr={snr} computing per-sample metrics ...")
        m = per_sample_metrics(preds)
        gens = [p["gen"] for p in preds]
        m["CLIPScore"] = per_sample_clip_score(img_feats, gens, device)
        if snr is None:
            baseline = m
        else:
            by_snr[float(snr)] = m
    return by_snr, baseline


METRICS = ["BLEU-1", "BLEU-3", "CIDEr", "CLIPScore"]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = []
    for r in RUNS:
        print(f"\n=== {r['label']} ===")
        by_snr, baseline = collect_run(r["run_dir"], device)
        data.append({**r, "by_snr": by_snr, "baseline": baseline})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    for ax, metric in zip(axes, METRICS):
        for d in data:
            snrs = sorted(d["by_snr"].keys())
            means = np.array([d["by_snr"][s][metric].mean() for s in snrs])
            sems = np.array([d["by_snr"][s][metric].std(ddof=1) /
                             np.sqrt(len(d["by_snr"][s][metric])) for s in snrs])
            lo, hi = means - 1.96 * sems, means + 1.96 * sems
            ax.plot(snrs, means, marker=d["marker"], color=d["color"],
                    linewidth=2, markersize=7, label=d["label"])
            ax.fill_between(snrs, lo, hi, color=d["color"], alpha=0.18)
            if metric in d["baseline"]:
                bmean = d["baseline"][metric].mean()
                ax.axhline(y=bmean, color=d["color"], linestyle="--",
                           alpha=0.5, linewidth=1)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("SNR Sweep — mean ± 95% CI (1.96·SEM, n=2000), dashed = no-noise baseline")
    fig.tight_layout()
    out = HERE / "snr_sweep_compare_ci.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[plot] {out}")

    # Numeric summary
    print("\n=== Numeric summary (mean ± 1.96·SEM) ===")
    snrs_all = sorted({s for d in data for s in d["by_snr"]})
    for metric in METRICS:
        print(f"\n[{metric}]")
        header = f'{"SNR":>5} | ' + " | ".join(f"{d['label']:>25}" for d in data) + " | Δ(Ours−LT)"
        print(header)
        for s in snrs_all:
            cells = []
            vals = []
            for d in data:
                if s in d["by_snr"]:
                    arr = d["by_snr"][s][metric]
                    m = arr.mean()
                    e = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
                    cells.append(f"{m:>.4f} ± {e:.4f}")
                    vals.append(m)
                else:
                    cells.append("            N/A         ")
                    vals.append(float("nan"))
            diff = vals[1] - vals[0] if len(vals) == 2 else float("nan")
            print(f"{s:>5.0f} | " + " | ".join(cells) + f" | {diff:+.4f}")


if __name__ == "__main__":
    main()
