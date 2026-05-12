"""eval_image.py가 만든 metrics_*.json 여러 개를 읽어서 표로 비교 print +
paper-friendly format (CSV / Markdown / LaTeX) 으로도 저장.

Usage:
  # stdout 만
  python Code/SemComm/compare_image_decoders.py \
      Code/SemComm/runs/img_..._<ts>/eval/metrics_n1000_seed42.json \
      Code/SemComm/runs/imgdiff_..._<ts>/eval/metrics_n1000_seed42_cfg3.0_nseed7.json

  # CSV + Markdown + LaTeX 으로 저장
  python Code/SemComm/compare_image_decoders.py \
      <paths...> --output-dir Code/SemComm/eval_results --tag baseline_vs_diffusion

산출 (--output-dir 지정 시):
  <output-dir>/<tag>_per_zsource.csv    — z_source 별 metric (각 row = 1 모델 × 1 z_source)
  <output-dir>/<tag>_delta.csv          — Δ(ztxt−zimg) per 모델
  <output-dir>/<tag>.md                 — Markdown 표 (README 복붙용)
  <output-dir>/<tag>.tex                — LaTeX booktabs 표 (paper 복붙용)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_ORDER = ["fid", "lpips", "clip_score", "psnr", "ssim"]
# higher_is_better
HIB = {"fid": False, "lpips": False, "clip_score": True, "psnr": True, "ssim": True}
ARROW = {k: "↑" if HIB[k] else "↓" for k in METRIC_ORDER}


def label_for(payload: dict) -> str:
    d = payload["decoder_type"]
    if d == "convt":
        return f"ConvT ({Path(payload['ckpt']).parent.parent.name})"
    return f"Diffusion cfg{payload['cfg_scale']} ({Path(payload['ckpt']).parent.parent.name})"


def short_label(payload: dict) -> str:
    """CSV/LaTeX 용 짧은 라벨."""
    d = payload["decoder_type"]
    if d == "convt":
        return "ConvT"
    return f"Diffusion(cfg={payload['cfg_scale']})"


def print_table(cells: list[list[str]], headers: list[str]) -> None:
    rows = [headers] + cells
    col_widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    sep = "  ".join("-" * w for w in col_widths)
    line = lambda r: "  ".join(s.ljust(w) for s, w in zip(r, col_widths))
    print(line(headers)); print(sep)
    for r in cells:
        print(line(r))


# ---------------------------------------------------------------------------
# Format writers

def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def write_markdown(path: Path, sections: list[tuple[str, list[str], list[list[str]]]],
                   note: str | None = None) -> None:
    """sections = [(section_title, headers, rows), ...]"""
    with open(path, "w") as f:
        for title, headers, rows in sections:
            f.write(f"## {title}\n\n")
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join("---" for _ in headers) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
            f.write("\n")
        if note:
            f.write(f"_{note}_\n")


def write_latex(path: Path, sections: list[tuple[str, list[str], list[list[str]]]],
                note: str | None = None) -> None:
    """booktabs 스타일. 사용 시 \\usepackage{booktabs} 필요."""
    with open(path, "w") as f:
        for title, headers, rows in sections:
            n_col = len(headers)
            col_spec = "l" + "c" * (n_col - 1)
            f.write("\\begin{table}[t]\n  \\centering\n")
            f.write(f"  \\caption{{{title}}}\n")
            f.write(f"  \\begin{{tabular}}{{{col_spec}}}\n  \\toprule\n  ")
            f.write(" & ".join(headers) + " \\\\\n  \\midrule\n  ")
            for r in rows:
                f.write(" & ".join(r) + " \\\\\n  ")
            f.write("\\bottomrule\n  \\end{tabular}\n\\end{table}\n\n")
        if note:
            f.write(f"% {note}\n")


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_files", nargs="+", help="metrics_*.json paths")
    ap.add_argument("--output-dir", default=None,
                    help="저장 디렉터리. 지정 시 CSV/MD/LaTeX 모두 출력. 미지정 시 stdout 만.")
    ap.add_argument("--tag", default="compare",
                    help="저장 파일명 prefix (default: 'compare')")
    args = ap.parse_args()

    payloads = []
    for p in args.metrics_files:
        with open(p) as f:
            payloads.append(json.load(f))

    # --- Per z_source table ---
    headers_zsrc = ["model", "z_src"] + [f"{k} {ARROW[k]}" for k in METRIC_ORDER]
    rows_zsrc_display = []   # stdout / md 용 (긴 라벨)
    rows_zsrc_csv = []        # CSV 용 (짧은 라벨 + 추가 메타)
    csv_headers_zsrc = ["model", "decoder_type", "cfg_scale", "noise_seed",
                        "ckpt", "n", "seed", "z_src"] + METRIC_ORDER

    for p in payloads:
        lbl_full = label_for(p)
        lbl_short = short_label(p)
        for zsrc in ("zimg", "ztxt"):
            if zsrc not in p["results"]:
                continue
            m = p["results"][zsrc]
            display_row = [lbl_full, zsrc] + [
                f"{m[k]:.3f}" if k in m else "-" for k in METRIC_ORDER
            ]
            rows_zsrc_display.append(display_row)
            csv_row = [
                lbl_short, p["decoder_type"],
                str(p.get("cfg_scale") if p.get("cfg_scale") is not None else ""),
                str(p.get("noise_seed") if p.get("noise_seed") is not None else ""),
                p["ckpt"], str(p["n"]), str(p["seed"]), zsrc,
            ] + [f"{m[k]:.6f}" if k in m else "" for k in METRIC_ORDER]
            rows_zsrc_csv.append(csv_row)

    print("\n=== Per z_source metrics ===")
    print_table(rows_zsrc_display, headers_zsrc)

    # --- Delta table ---
    headers_delta = ["model"] + [f"Δ{k}" for k in METRIC_ORDER]
    rows_delta_display = []
    rows_delta_csv = []
    csv_headers_delta = ["model", "decoder_type", "cfg_scale", "ckpt"] + [f"delta_{k}" for k in METRIC_ORDER]

    for p in payloads:
        if "delta_ztxt_minus_zimg" not in p["results"]:
            continue
        d = p["results"]["delta_ztxt_minus_zimg"]
        rows_delta_display.append(
            [label_for(p)] + [f"{d[k]:+.3f}" if k in d else "-" for k in METRIC_ORDER]
        )
        rows_delta_csv.append([
            short_label(p), p["decoder_type"],
            str(p.get("cfg_scale") if p.get("cfg_scale") is not None else ""),
            p["ckpt"],
        ] + [f"{d[k]:+.6f}" if k in d else "" for k in METRIC_ORDER])

    print("\n=== Δ (ztxt − zimg) — modality-agnostic 지표: |Δ| 작을수록 좋음 ===")
    print_table(rows_delta_display, headers_delta)

    note = "fid/lpips ↓ better, clip_score/psnr/ssim ↑ better. |Δ| ↓ better (modality-agnostic)."
    print(f"\n* {note}")

    # --- Save files ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_csv(out_dir / f"{args.tag}_per_zsource.csv", csv_headers_zsrc, rows_zsrc_csv)
        write_csv(out_dir / f"{args.tag}_delta.csv", csv_headers_delta, rows_delta_csv)

        sections = [
            ("Per z_source metrics", headers_zsrc, rows_zsrc_display),
            ("Δ (ztxt − zimg) — modality-agnostic", headers_delta, rows_delta_display),
        ]
        write_markdown(out_dir / f"{args.tag}.md", sections, note=note)

        # LaTeX 은 arrow / Δ 기호 escape 필요 (Δ → $\Delta$)
        sections_latex = []
        for title, headers, rows in sections:
            headers_tex = [h.replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
                            .replace("Δ", "$\\Delta$").replace("_", "\\_")
                           for h in headers]
            rows_tex = [[c.replace("_", "\\_") for c in r] for r in rows]
            sections_latex.append((title.replace("Δ", "$\\Delta$"), headers_tex, rows_tex))
        write_latex(out_dir / f"{args.tag}.tex", sections_latex, note=note)

        print(f"\n[save] {out_dir}/")
        for ext in ("_per_zsource.csv", "_delta.csv", ".md", ".tex"):
            print(f"  {args.tag}{ext}")


if __name__ == "__main__":
    main()
