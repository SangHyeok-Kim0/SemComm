"""Train ClipCap-style TextDecoder (frozen Qwen + Transformer mapper) on cached COCO embeddings.

  z source policy:
    centroid: mu = normalize((z_img[i] + z_txt[j]) / 2) for caption j with parent image i
    modality: z = z_txt[j]   (image-text 짝의 caption 측 임베딩만 사용)

LM is frozen (cfg.text_decoder.freeze_lm). Only the mapper updates.

Run artifacts: runs/<run_name>/{config.json, checkpoints/*.pt, samples/, metrics.json}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import TextDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def build_run_name(d_cfg: dict, suffix: str) -> str:
    if d_cfg.get("run_name") and d_cfg["run_name"] not in ("auto", "null", ""):
        return d_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (f"{suffix}_{d_cfg['z_source']}_bs{d_cfg['batch_size']}"
            f"_lr{d_cfg['learning_rate']}_K{d_cfg['prefix_len']}_{ts}")


# ---------------------------------------------------------------------------
# Dataset (per-caption sampling)

class TextDecoderDataset(Dataset):
    """Yield (z, caption_text). One sample per caption (train2017 ~590k samples)."""

    def __init__(self, caption_image_idx: list[int], caption_texts: list[str],
                 z_img: torch.Tensor, z_txt: torch.Tensor, z_source: str):
        assert z_source in ("centroid", "modality")
        assert len(caption_image_idx) == len(caption_texts) == z_txt.size(0)
        self.caption_image_idx = caption_image_idx
        self.caption_texts = caption_texts
        self.z_img = z_img
        self.z_txt = z_txt
        self.z_source = z_source

    def __len__(self) -> int:
        return len(self.caption_texts)

    def __getitem__(self, j: int):
        text = self.caption_texts[j]
        z_t = self.z_txt[j].float()
        if self.z_source == "modality":
            z = z_t
        else:  # centroid
            im_idx = self.caption_image_idx[j]
            z_v = self.z_img[im_idx].float()
            z = (z_v + z_t) / 2.0
            z = F.normalize(z, dim=-1)
        return z, text


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        zs = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        return zs, enc.input_ids, enc.attention_mask
    return collate


# ---------------------------------------------------------------------------
# Eval

@torch.no_grad()
def evaluate(model: TextDecoder, loader: DataLoader, device: torch.device,
             use_amp: bool, amp_dtype: torch.dtype,
             max_batches: int | None = None) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for bi, (z, input_ids, attn) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        z = z.to(device); input_ids = input_ids.to(device); attn = attn.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = model(z, input_ids, attn)
        total_loss += loss.item()
        n_batches += 1
    return {"val/ce": total_loss / max(n_batches, 1)}


@torch.no_grad()
def sample_generate(model: TextDecoder, loader: DataLoader, device: torch.device,
                    n_samples: int, beam_size: int, max_new_tokens: int) -> list[tuple[str, str]]:
    """Return list of (gt_caption, generated_caption) for first n_samples val items."""
    model.eval()
    out: list[tuple[str, str]] = []
    z_buf, gt_buf = [], []
    for z, input_ids, attn in loader:
        for k in range(z.size(0)):
            if len(z_buf) >= n_samples:
                break
            z_buf.append(z[k])
            gt_buf.append(model.tokenizer.decode(input_ids[k][attn[k].bool()],
                                                  skip_special_tokens=True))
        if len(z_buf) >= n_samples:
            break
    z_t = torch.stack(z_buf).to(device)
    gens = model.generate(z_t, beam_size=beam_size, max_new_tokens=max_new_tokens)
    for gt, gen in zip(gt_buf, gens):
        out.append((gt, gen))
    return out


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--max-train-captions", type=int, default=None,
                    help="cap train caption count for sanity dry-run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override text_decoder.batch_size")
    ap.add_argument("--z-source", default=None, choices=["centroid", "modality"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    text_cfg = cfg["text_decoder"]
    if args.epochs is not None:
        text_cfg["epochs"] = args.epochs
    if args.z_source is not None:
        text_cfg["z_source"] = args.z_source
    if args.batch_size is not None:
        text_cfg["batch_size"] = args.batch_size

    random.seed(text_cfg["seed"])
    torch.manual_seed(text_cfg["seed"])
    device = torch.device(f"cuda:{text_cfg['device_id']}" if torch.cuda.is_available() else "cpu")

    cache_dir = resolve_path(cfg["cache_dir"], HERE)
    runs_root = resolve_path(cfg["runs_root"], HERE)
    run_name = build_run_name(text_cfg, suffix="txt")
    run_dir = runs_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[run] {run_dir}")

    # WandB
    wb_cfg = cfg.get("wandb", {})
    wb_mode = "online" if wb_cfg.get("enabled", False) else "disabled"
    wandb.init(
        project=wb_cfg.get("project", "SemCom"),
        entity=wb_cfg.get("entity") or None,
        name=run_name,
        config=cfg,
        mode=wb_mode,
        dir=str(run_dir),
    )

    # Cache
    print("[cache] loading ...")
    z_img_train = torch.load(cache_dir / "z_img_train.pt", weights_only=False)
    z_txt_train = torch.load(cache_dir / "z_txt_train.pt", weights_only=False)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)

    # Optional subset
    if args.max_train_captions is not None:
        n_cap = min(args.max_train_captions, idx["train"]["n_captions"])
        # Keep only captions whose parent image is still cached
        cap_img_idx = idx["train"]["caption_image_idx"][:n_cap]
        cap_texts = idx["train"]["caption_texts"][:n_cap]
        z_txt_train = z_txt_train[:n_cap]
        idx["train"]["caption_image_idx"] = cap_img_idx
        idx["train"]["caption_texts"] = cap_texts
        print(f"[sanity] capped train captions to {n_cap}")

    # Model (loads tokenizer + LM)
    print(f"[model] loading {text_cfg['lm_name']} ...")
    model = TextDecoder(
        lm_name=text_cfg["lm_name"],
        z_dim=z_img_train.size(-1),
        prefix_len=text_cfg["prefix_len"],
        mapper_type=text_cfg["mapper_type"],
        mapper_layers=text_cfg["mapper_layers"],
        mapper_heads=text_cfg["mapper_heads"],
        mapper_dropout=text_cfg["mapper_dropout"],
        freeze_lm=text_cfg["freeze_lm"],
        lm_dtype=text_cfg.get("lm_dtype", "auto"),
    ).to(device)

    n_train = sum(p.numel() for p in model.mapper.parameters())
    n_lm = sum(p.numel() for p in model.lm.parameters())
    print(f"[model] mapper trainable={n_train/1e6:.2f}M, LM frozen={n_lm/1e6:.2f}M, "
          f"d_model={model.lm_dim}")

    train_ds = TextDecoderDataset(
        idx["train"]["caption_image_idx"],
        idx["train"]["caption_texts"],
        z_img_train, z_txt_train, text_cfg["z_source"],
    )
    val_ds = TextDecoderDataset(
        idx["val"]["caption_image_idx"],
        idx["val"]["caption_texts"],
        z_img_val, z_txt_val, text_cfg["z_source"],
    )
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    collate = make_collate(model.tokenizer, text_cfg["caption_max_tokens"])
    train_loader = DataLoader(train_ds, batch_size=text_cfg["batch_size"], shuffle=True,
                              num_workers=text_cfg["num_workers"], drop_last=True,
                              collate_fn=collate, pin_memory=True,
                              persistent_workers=text_cfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=text_cfg["batch_size"], shuffle=False,
                            num_workers=text_cfg["num_workers"], drop_last=False,
                            collate_fn=collate, pin_memory=True,
                            persistent_workers=text_cfg["num_workers"] > 0)

    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=text_cfg["learning_rate"],
                                  weight_decay=text_cfg["weight_decay"])

    # Mixed precision selection (mirrors train_image_decoder.py:262-271 pattern).
    precision = text_cfg.get("precision", "bf16")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown text_decoder.precision: {precision!r} "
                         f"(expected fp32 | fp16 | bf16)")
    use_amp = (precision != "fp32") and device.type == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[precision]
    # bf16 has fp32-equivalent dynamic range → no GradScaler needed; only fp16 risks underflow.
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and use_amp))
    print(f"[precision] {precision}{' (AMP)' if use_amp else ' (no AMP)'}")

    metrics_log: list[dict] = []
    global_step = 0
    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------
    # Train
    for epoch in range(text_cfg["epochs"]):
        model.train()
        running_loss, running_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"ep{epoch+1}/{text_cfg['epochs']}")
        for z, input_ids, attn in pbar:
            z = z.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(z, input_ids, attn)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * z.size(0)
            running_n += z.size(0)
            global_step += 1
            pbar.set_postfix(ce=f"{loss.item():.4f}")

            # epoch scalar는 reference용 — 매끄러운 1.0 → 2.0 → ... 진행 (wrap 없음).
            step_in_epoch = (global_step - 1) % steps_per_epoch + 1
            wandb.log({
                "loss/ce": loss.item(),
                "optim/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + step_in_epoch / steps_per_epoch,
            }, step=global_step)

        train_avg = running_loss / max(running_n, 1)

        # Val CE (subset for speed if sanity)
        val_metrics = evaluate(model, val_loader, device,
                               use_amp=use_amp, amp_dtype=amp_dtype,
                               max_batches=20 if args.max_train_captions else None)

        # Sample 8 generations for inspection
        samples = sample_generate(model, val_loader, device, n_samples=8,
                                  beam_size=text_cfg["beam_size"],
                                  max_new_tokens=text_cfg["max_new_tokens"])
        with open(sample_dir / f"epoch_{epoch+1:03d}.txt", "w") as f:
            for gt, gen in samples:
                f.write(f"GT : {gt}\nGEN: {gen}\n---\n")

        epoch_log = {"epoch": epoch + 1, "train/ce": train_avg, **val_metrics}
        metrics_log.append(epoch_log)
        print(f"  [epoch {epoch+1}] " + " ".join(f"{k}={v:.4f}" for k, v in epoch_log.items() if k != "epoch"))
        for gt, gen in samples[:2]:
            print(f"   GT : {gt}\n   GEN: {gen}")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        # Per-epoch wandb scalars
        wandb_epoch = {"train_epoch/ce": train_avg, **val_metrics, "epoch": epoch + 1}
        wandb.log(wandb_epoch, step=global_step)

        # Per-epoch sample table (GT vs GEN)
        log_samples = ((epoch + 1) %
                       max(int(cfg.get("wandb", {}).get("log_samples_every_n_epochs", 1)), 1) == 0)
        if log_samples:
            tbl = wandb.Table(columns=["epoch", "gt", "gen"])
            for gt, gen in samples:
                tbl.add_data(epoch + 1, gt, gen)
            wandb.log({"samples/captions": tbl}, step=global_step)

        if (epoch + 1) % text_cfg["save_every_n_epochs"] == 0 or (epoch + 1) == text_cfg["epochs"]:
            torch.save({"mapper": model.mapper.state_dict(),
                        "epoch": epoch + 1, "config": cfg},
                       ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    torch.save({"mapper": model.mapper.state_dict(),
                "epoch": text_cfg["epochs"], "config": cfg},
               ckpt_dir / "final.pt")
    wandb.finish()
    print(f"\n[done] saved to {run_dir}")


if __name__ == "__main__":
    main()
