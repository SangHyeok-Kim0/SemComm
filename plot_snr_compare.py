"""Combined SNR sweep comparison plot (CLIP-LT vs Ours).

두 text decoder run의 eval_metrics_snr*.json을 읽어 4-panel (BLEU-1, BLEU-3, CIDEr, CLIPScore)
SNR sweep 비교 plot 생성.

Usage:
  python plot_snr_compare.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

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

METRICS = ["BLEU-1", "BLEU-3", "CIDEr", "CLIPScore"]


def collect(run_dir: Path) -> tuple[dict[float, dict[str, float]], dict[str, float]]:
    """returns ({snr: metrics_dict}, baseline_metrics)."""
    metric_dir = run_dir / "results" / "metric"
    by_snr: dict[float, dict[str, float]] = {}
    baseline: dict[str, float] = {}
    for f in metric_dir.glob("eval_metrics*.json"):
        d = json.load(open(f))
        # text_random scenario only
        m = d["metrics_by_scenario"].get("text_random")
        if m is None:
            continue
        snr = d["config"].get("noise_snr_db")
        if snr is None:
            baseline = m
        else:
            by_snr[float(snr)] = m
    return by_snr, baseline


def main():
    data = []
    for r in RUNS:
        by_snr, baseline = collect(r["run_dir"])
        data.append({**r, "by_snr": by_snr, "baseline": baseline})
        snrs = sorted(by_snr.keys())
        print(f"[{r['label']}] SNRs: {snrs}")
        print(f"  baseline: {baseline}")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    for ax, metric in zip(axes, METRICS):
        for d in data:
            snrs = sorted(d["by_snr"].keys())
            ys = [d["by_snr"][s].get(metric, float("nan")) for s in snrs]
            ax.plot(snrs, ys, marker=d["marker"], color=d["color"],
                    linewidth=2, markersize=8, label=d["label"])
            if metric in d["baseline"]:
                ax.axhline(y=d["baseline"][metric], color=d["color"],
                           linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("SNR Sweep Comparison — text_random (epoch=10, n=2000, dashed = no-noise baseline)")
    fig.tight_layout()
    out = HERE / "snr_sweep_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
