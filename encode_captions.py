"""Pre-generate captions from a trained TextDecoder for image_decoder M3 stream.

핵심: image_decoder 학습 시 매 step `ĉ = TextDecoder(z)`를 호출하면 너무 느리다.
text decoder는 frozen이라 결과가 deterministic이므로 사전 생성 후 cache로 저장한다.

Outputs (under cache/):
  captions_{text_decoder_run}_train_zimg.pt   list[str], len=n_images   (118287)
  captions_{text_decoder_run}_train_ztxt.pt   list[str], len=n_captions (591753)
  captions_{text_decoder_run}_val_zimg.pt     list[str], len=5000
  captions_{text_decoder_run}_val_ztxt.pt     list[str], len=25014

Indexing rule (image_decoder 학습 시):
  z_source random_img_txt 선택이 z_img이면 → captions_zimg[caption_image_idx[j]]
  z_source random_img_txt 선택이 z_txt이면 → captions_ztxt[j]

파일명 prefix가 text_decoder_run을 포함해 어떤 text decoder의 caption인지 즉시 식별 가능.

Usage:
  python encode_captions.py --text-decoder-run txt_random_img_txt_bs64_lr2e-05_K10_20260511-135112
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import TextDecoder  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


@torch.no_grad()
def gen_all(model: TextDecoder, z_all: torch.Tensor, device: torch.device,
            batch_size: int, beam_size: int, max_new_tokens: int,
            label: str) -> list[str]:
    captions: list[str] = []
    for i in tqdm(range(0, z_all.size(0), batch_size), desc=label):
        zb = z_all[i:i + batch_size].float().to(device)
        caps = model.generate(zb, beam_size=beam_size, max_new_tokens=max_new_tokens)
        captions.extend(caps)
    return captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--text-decoder-run", required=True,
                    help="text decoder run name under runs/ (e.g. txt_random_img_txt_..._)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="generate batch size (Qwen3-0.6B on GB10 handles 128 comfortably)")
    ap.add_argument("--beam-size", type=int, default=5,
                    help="1=greedy (fast). Caption diversity matters less for conditioning than speed.")
    ap.add_argument("--max-new-tokens", type=int, default=20,
                    help="COCO caption 95th-percentile ~25 token, 20 covers most")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if cache file already exists")
    ap.add_argument("--cache-dir", default=None,
                    help="override cfg['cache_dir']. Stage 2 (Standard CLIP) 시 사용.")
    ap.add_argument("--zkinds", nargs="+", default=["zimg", "ztxt"],
                    choices=["zimg", "ztxt"],
                    help="which z-kind(s) to generate captions for. "
                         "ztxt-only 학습이면 'ztxt'만 지정해 시간 절약.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        f"cuda:{cfg['text_decoder']['device_id']}" if torch.cuda.is_available() else "cpu"
    )

    cache_dir = resolve_path(args.cache_dir if args.cache_dir else cfg["cache_dir"], HERE)
    print(f"[cache_dir] {cache_dir}")
    runs_root = resolve_path(cfg["runs_root"], HERE)
    run_dir = runs_root / args.text_decoder_run
    # 새 구조: checkpoints/models/final.pt, 옛 구조: checkpoints/final.pt 둘 다 지원.
    ckpt_path = run_dir / "checkpoints" / "models" / "final.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"text decoder final.pt not found: {ckpt_path}")

    # Load ckpt and inspect saved config
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_td = ckpt["config"]["text_decoder"]
    print(f"[load] {ckpt_path}")
    print(f"[load] z_source={saved_td['z_source']}, epoch={ckpt['epoch']}")

    # Load cached embeddings (z_dim inferred here)
    print("[cache] loading embeddings ...")
    z_img_train = torch.load(cache_dir / "z_img_train.pt", weights_only=False)
    z_txt_train = torch.load(cache_dir / "z_txt_train.pt", weights_only=False)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    z_dim = z_img_train.size(-1)
    print(f"[cache] z_dim={z_dim}, train_img={z_img_train.shape}, train_txt={z_txt_train.shape}")

    # Rebuild TextDecoder with saved hyperparams + load mapper
    print(f"[model] loading {saved_td['lm_name']} ...")
    model = TextDecoder(
        lm_name=saved_td["lm_name"],
        z_dim=z_dim,
        prefix_len=saved_td["prefix_len"],
        mapper_type=saved_td["mapper_type"],
        mapper_layers=saved_td["mapper_layers"],
        mapper_heads=saved_td["mapper_heads"],
        mapper_dropout=saved_td["mapper_dropout"],
        freeze_lm=saved_td["freeze_lm"],
        lm_dtype=saved_td.get("lm_dtype", "auto"),
    ).to(device)
    model.mapper.load_state_dict(ckpt["mapper"])
    model.eval()

    prefix = f"captions_{args.text_decoder_run}"
    all_targets = [
        ("train z_img", z_img_train, "train", "zimg"),
        ("train z_txt", z_txt_train, "train", "ztxt"),
        ("val z_img",   z_img_val,   "val",   "zimg"),
        ("val z_txt",   z_txt_val,   "val",   "ztxt"),
    ]
    targets = [t for t in all_targets if t[3] in args.zkinds]
    print(f"[zkinds] generating: {args.zkinds}")
    for label, z_all, split, zkind in targets:
        out_path = cache_dir / f"{prefix}_{split}_{zkind}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name} exists ({z_all.size(0)} samples) — use --overwrite to regenerate")
            continue
        caps = gen_all(model, z_all, device,
                       batch_size=args.batch_size,
                       beam_size=args.beam_size,
                       max_new_tokens=args.max_new_tokens,
                       label=label)
        torch.save(caps, out_path)
        print(f"[save] {out_path.name} ({len(caps)} captions)")
        # Quick sample print for inspection
        for s in caps[:3]:
            print(f"   • {s}")

    print("[done] caption cache generated.")


if __name__ == "__main__":
    main()
