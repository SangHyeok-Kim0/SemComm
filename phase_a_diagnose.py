"""Phase A — Decoder-swap 가능성 진단

ModalityGap Run 1의 final_full.pt 임베딩만으로 (학습 없이) 다음을 산출:
  1) Modality classifier sanity (linear LR, RBF-SVM, 5-NN)
  2) Pair geometry 통계 + 분포 히스토그램
  3) AWGN σ 스윕에서 KNN / V-Measure / CosTP / gap 변화

산출물: Code/SemComm/results/phase_a/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODGAP = HERE.parent / "ModalityGap"
sys.path.insert(0, str(MODGAP))
from ModalityGap.metrics import linear_modality_classifier_acc  # noqa: E402

DEFAULT_RUN = MODGAP / "runs" / "bs128_lr1e-4_ep100_Tfix_fp16_a50-50_b20-50_20260505-034844"
RESULTS = HERE / "results" / "phase_a"


# ---------------------------------------------------------------------------

def to_unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def coco_single_object_labels(ids_img, coco_root: Path):
    """Mirror metrics.compute_clustering_metrics's filter: keep pairs whose
    image has exactly one distinct category. Returns (keep_idx, labels, n_classes).
    """
    from pycocotools.coco import COCO

    inst_path = coco_root / "annotations/instances_val2017.json"
    coco = COCO(str(inst_path))
    keep, raw = [], []
    for i, sid in enumerate(ids_img):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=int(sid)))
        cats = {a["category_id"] for a in anns}
        if len(cats) == 1:
            keep.append(i)
            raw.append(next(iter(cats)))
    uniq = sorted(set(raw))
    remap = {c: j for j, c in enumerate(uniq)}
    labels = torch.tensor([remap[c] for c in raw], dtype=torch.long)
    return torch.tensor(keep, dtype=torch.long), labels, len(uniq)


# ---------- 1. Modality classifier ----------

def diagnose_modality_classifier(img_e: torch.Tensor, txt_e: torch.Tensor) -> dict:
    """0.5 ≈ modality-agnostic (어떤 분류기도 modality 못 맞춤).
    1.0 ≈ 임베딩만 봐도 modality가 식별됨 (gap 큼). decoder-swap에 직격."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    out = {"linear_LR_acc": linear_modality_classifier_acc(img_e, txt_e)}

    X = np.concatenate([img_e.numpy(), txt_e.numpy()], axis=0)
    y = np.concatenate([np.zeros(len(img_e)), np.ones(len(txt_e))])

    # SVM/KNN은 풀배치가 무거워 subsample (각 modality에서 같은 수만큼).
    n_each = min(2000, len(img_e))
    rng = np.random.default_rng(0)
    idx_i = rng.choice(len(img_e), n_each, replace=False)
    idx_t = rng.choice(len(txt_e), n_each, replace=False) + len(img_e)
    sel = np.concatenate([idx_i, idx_t])
    Xs, ys = X[sel], y[sel]

    out["rbf_svm_acc"] = float(cross_val_score(SVC(kernel="rbf"), Xs, ys, cv=5).mean())
    out["knn5_acc"] = float(cross_val_score(KNeighborsClassifier(5), Xs, ys, cv=5).mean())
    out["n_subsample_per_modality"] = n_each
    return out


# ---------- 2. Pair geometry ----------

def diagnose_pair_geometry(img_e: torch.Tensor, txt_e: torch.Tensor) -> dict:
    pair_dist = (img_e - txt_e).norm(dim=-1)
    pair_cos = (img_e * txt_e).sum(-1)
    return {
        "pair_dist_mean": float(pair_dist.mean()),
        "pair_dist_std": float(pair_dist.std(unbiased=False)),
        "pair_dist_min": float(pair_dist.min()),
        "pair_dist_max": float(pair_dist.max()),
        "pair_cos_mean": float(pair_cos.mean()),
        "pair_cos_std": float(pair_cos.std(unbiased=False)),
        "pair_cos_min": float(pair_cos.min()),
        "pair_cos_max": float(pair_cos.max()),
        "_pair_dist": pair_dist.numpy(),
        "_pair_cos": pair_cos.numpy(),
    }


# ---------- 3. AWGN sweep ----------

def _knn_acc(Xtr, ytr, Xte, yte, k=5):
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    return accuracy_score(yte, knn.predict(Xte))


def _vmeas(X, y, n_clusters):
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    return v_measure_score(y, km.fit_predict(X))


def _add_awgn_renorm(z: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    if sigma == 0.0:
        return z
    n = torch.randn(z.shape, generator=gen) * sigma
    return to_unit(z + n)


def diagnose_awgn_sweep(img_e, txt_e, keep_idx, labels, n_classes, sigmas, seed=0):
    D = img_e.size(-1)
    rng = np.random.default_rng(42)
    perm = rng.permutation(2 * len(keep_idx))  # split is fixed across sigmas

    rows = []
    for sigma in sigmas:
        g = torch.Generator().manual_seed(seed)
        zi = _add_awgn_renorm(img_e, sigma, g)
        zt = _add_awgn_renorm(txt_e, sigma, g)

        zi_k = zi[keep_idx]
        zt_k = zt[keep_idx]

        cos_tp = float((zi_k * zt_k).sum(-1).mean())
        gap = float((zi.mean(0) - zt.mean(0)).norm())

        # Stack like compute_clustering_metrics: text first, then image.
        X = torch.vstack([zt_k, zi_k]).numpy()
        y = torch.cat([labels, labels]).numpy()
        ntr = int(0.8 * len(X))
        tr, te = perm[:ntr], perm[ntr:]
        knn = _knn_acc(X[tr], y[tr], X[te], y[te])
        vm = _vmeas(X, y, n_classes)

        # SNR per-dim: signal power = 1/D (unit-norm 가정), noise power = σ².
        snr_db = -float("inf") if sigma == 0 else float(-10 * np.log10(D * sigma ** 2))
        rows.append({
            "sigma": float(sigma),
            "snr_db": snr_db,
            "gap": gap,
            "cos_tp": cos_tp,
            "knn_acc": float(knn),
            "v_measure": float(vm),
        })
    return rows


# ---------- plotting ----------

def plot(geo, awgn_rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(geo["_pair_dist"], bins=60, color="steelblue", alpha=0.85)
    axes[0].set_xlabel(r"$\|z_i^{img} - z_i^{txt}\|_2$")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Pair Euclidean dist  (mean={geo['pair_dist_mean']:.3f}, "
                      f"std={geo['pair_dist_std']:.3f})")
    axes[1].hist(geo["_pair_cos"], bins=60, color="indianred", alpha=0.85)
    axes[1].set_xlabel(r"$\cos(z_i^{img}, z_i^{txt})$")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Pair cosine sim  (mean={geo['pair_cos_mean']:.3f}, "
                      f"std={geo['pair_cos_std']:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "pair_geometry_hist.png", dpi=130)
    plt.close(fig)

    sigmas = [r["sigma"] for r in awgn_rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(sigmas, [r["knn_acc"] for r in awgn_rows], "o-", color="C0", label="KNN acc")
    ax1.plot(sigmas, [r["v_measure"] for r in awgn_rows], "s-", color="C1", label="V-Measure")
    ax1.plot(sigmas, [r["cos_tp"] for r in awgn_rows], "^-", color="C2", label="CosTP")
    ax1.set_xlabel(r"AWGN $\sigma$ (per-dim, on unit-norm embedding)")
    ax1.set_ylabel("metric")
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left")
    ax1.set_title("Channel-noise robustness — Run 1 latent space")

    ax2 = ax1.twinx()
    ax2.plot(sigmas, [r["gap"] for r in awgn_rows], "x--", color="gray",
             label="gap (centroid dist)")
    ax2.set_ylabel("gap", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    fig.savefig(out_dir / "awgn_sweep.png", dpi=130)
    plt.close(fig)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(DEFAULT_RUN))
    p.add_argument("--coco-root", default=str(MODGAP / "data" / "coco"))
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)

    print(f"[load] {run_dir / 'embeddings/final_full.pt'}")
    snap = torch.load(run_dir / "embeddings" / "final_full.pt",
                      map_location="cpu", weights_only=False)
    img_e = to_unit(snap["image_embeds"])
    txt_e = to_unit(snap["text_embeds"])
    ids_img = list(snap["ids_img"])
    print(f"  N pairs={len(img_e)}  D={img_e.size(-1)}")

    # 1) modality classifier
    print("\n[1/3] Modality classifier sanity")
    clf = diagnose_modality_classifier(img_e, txt_e)
    print(json.dumps(clf, indent=2))
    (RESULTS / "modality_classifier.json").write_text(json.dumps(clf, indent=2))

    # 2) pair geometry
    print("\n[2/3] Pair geometry")
    geo = diagnose_pair_geometry(img_e, txt_e)
    summary = {k: v for k, v in geo.items() if not k.startswith("_")}
    print(json.dumps(summary, indent=2))
    (RESULTS / "pair_geometry.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS / "pair_dist_values.npy", geo["_pair_dist"])
    np.save(RESULTS / "pair_cos_values.npy", geo["_pair_cos"])

    # COCO single-object 라벨 (KNN/V-Measure용)
    print("\n[label] COCO val2017 single-object filter")
    keep_idx, labels, n_classes = coco_single_object_labels(ids_img, Path(args.coco_root))
    print(f"  kept {len(keep_idx)}/{len(ids_img)} pairs, {n_classes} classes")

    # 3) AWGN sweep
    print("\n[3/3] AWGN σ sweep")
    rows = diagnose_awgn_sweep(img_e, txt_e, keep_idx, labels, n_classes,
                               args.sigmas, seed=args.seed)
    print(f"  {'sigma':>7s} {'SNR_dB':>8s} {'gap':>6s} {'CosTP':>6s} {'KNN':>6s} {'V':>6s}")
    for r in rows:
        snr_str = f"{r['snr_db']:>+8.1f}" if r["snr_db"] != float("-inf") else "    -inf"
        print(f"  σ={r['sigma']:<6.3f}{snr_str}  {r['gap']:>5.3f}  "
              f"{r['cos_tp']:>5.3f}  {r['knn_acc']:>5.3f}  {r['v_measure']:>5.3f}")

    with open(RESULTS / "awgn_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot(geo, rows, RESULTS)
    print(f"\n[done] artifacts in {RESULTS}/")


if __name__ == "__main__":
    main()
