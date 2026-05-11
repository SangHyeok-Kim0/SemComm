"""Train the ConvTranspose ImageDecoder on cached COCO embeddings.

  z source policy:
    centroid: mu = normalize((z_img + z_txt) / 2)   ← cfg.image_decoder.z_source = 'centroid'
    modality: z = z_img                             ← cfg.image_decoder.z_source = 'modality'

  Pixel space is [-1, 1] (NOT CLIP normalization), so we re-load originals via
  PIL and apply T.Normalize(0.5, 0.5). The encoder cache used CLIP mean/std at
  encode time — that step is upstream-only.

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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import ImageDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers

def load_config(base_path: str, decoder_path: str | None = None) -> dict:
    """Load base config (config.yaml) and optionally deep-merge a decoder-specific config
    (e.g. config_convt.yaml or config_diffusion.yaml) into it."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if decoder_path:
        with open(decoder_path) as f:
            override = yaml.safe_load(f) or {}
        for k, v in override.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def make_pixel_transform() -> T.Compose:
    """[-1, 1] pixel-space (Tanh-compatible). NOT CLIP normalization."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),                                      # [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),     # [-1, 1]
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def build_run_name(d_cfg: dict, suffix: str) -> str:
    if d_cfg.get("run_name") and d_cfg["run_name"] not in ("auto", "null", ""):
        return d_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (f"{suffix}_{d_cfg['z_source']}_bs{d_cfg['batch_size']}"
            f"_lr{d_cfg['learning_rate']}_{ts}")


# ---------------------------------------------------------------------------
# Dataset

class ImageDecoderDataset(Dataset):
    """Yield (z, x_pixel) pairs.

    z source: 'centroid' samples one of an image's captions per __getitem__ →
    centroid varies across epochs even for the same image. 'modality' returns
    z_img directly (same z every epoch for a given image).
    """

    def __init__(self, image_dir: Path, image_ids: list[int],
                 caption_image_idx: list[int],
                 z_img: torch.Tensor, z_txt: torch.Tensor,
                 transform: T.Compose, z_source: str):
        assert z_source in ("centroid", "modality")
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform
        self.z_img = z_img        # (N_img, D) fp16, unit-norm
        self.z_txt = z_txt        # (N_cap, D) fp16, unit-norm
        self.z_source = z_source

        # Captions grouped by parent image index.
        per_img: list[list[int]] = [[] for _ in image_ids]
        for cap_j, im_idx in enumerate(caption_image_idx):
            per_img[im_idx].append(cap_j)
        # Some COCO images may have 0 captions; filter to keep dataset clean.
        self.valid = [i for i, lst in enumerate(per_img) if len(lst) > 0]
        self.captions_for_image = per_img

    def __len__(self) -> int:
        return len(self.valid)

    def _file(self, iid: int) -> Path:
        return self.image_dir / f"{iid:012d}.jpg"

    def __getitem__(self, k: int):
        i = self.valid[k]
        iid = self.image_ids[i]
        img = Image.open(self._file(iid)).convert("RGB")
        x = self.transform(img)

        z_v = self.z_img[i].float()
        if self.z_source == "modality":
            z = z_v
        else:  # centroid
            cap_j = random.choice(self.captions_for_image[i])
            z_t = self.z_txt[cap_j].float()
            z = (z_v + z_t) / 2.0
            z = F.normalize(z, dim=-1)
        return z, x


# ---------------------------------------------------------------------------
# Eval

@torch.no_grad()
def evaluate(model: ImageDecoder, loader: DataLoader, device: torch.device,
             lpips_fn=None, max_batches: int | None = None) -> dict:
    model.eval()
    mse_sum, l1_sum, lpips_sum = 0.0, 0.0, 0.0
    n_pixels, n_images = 0, 0
    for bi, (z, x) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        z = z.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        x_hat = model(z)
        mse_sum += F.mse_loss(x_hat, x, reduction="sum").item()
        l1_sum += F.l1_loss(x_hat, x, reduction="sum").item()
        if lpips_fn is not None:
            lpips_sum += lpips_fn(x_hat, x).sum().item()
        n_pixels += x.numel()
        n_images += x.size(0)
    out = {
        "val/mse": mse_sum / max(n_pixels, 1),
        "val/l1": l1_sum / max(n_pixels, 1),
    }
    if lpips_fn is not None:
        out["val/lpips"] = lpips_sum / max(n_images, 1)
    return out


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="base config (encoder/data/wandb/text_decoder)")
    ap.add_argument("--decoder-config", default=str(HERE / "config_convt.yaml"),
                    help="image_decoder config (ConvT v1 default)")
    ap.add_argument("--max-train-images", type=int, default=None,
                    help="cap train images for sanity dry-run")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config.epochs (sanity)")
    ap.add_argument("--z-source", default=None, choices=["centroid", "modality"],
                    help="override config.z_source")
    args = ap.parse_args()

    cfg = load_config(args.config, args.decoder_config)
    img_cfg = cfg["image_decoder"]
    if args.epochs is not None:
        img_cfg["epochs"] = args.epochs
    if args.z_source is not None:
        img_cfg["z_source"] = args.z_source

    random.seed(img_cfg["seed"])
    torch.manual_seed(img_cfg["seed"])

    device = torch.device(f"cuda:{img_cfg['device_id']}" if torch.cuda.is_available() else "cpu")
    coco_root = resolve_path(cfg["coco_root"], HERE)
    cache_dir = resolve_path(cfg["cache_dir"], HERE)
    runs_root = resolve_path(cfg["runs_root"], HERE)

    run_name = build_run_name(img_cfg, suffix="img")
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

    # Cache load
    print("[cache] loading ...")
    z_img_train = torch.load(cache_dir / "z_img_train.pt", weights_only=False)
    z_txt_train = torch.load(cache_dir / "z_txt_train.pt", weights_only=False)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)

    # Optional subset for sanity
    if args.max_train_images is not None:
        n = min(args.max_train_images, idx["train"]["n_images"])
        kept_img_ids = idx["train"]["image_ids"][:n]
        # Filter captions whose image_idx < n
        kept_cap_idx = [cap_im for cap_im in idx["train"]["caption_image_idx"] if cap_im < n]
        z_img_train = z_img_train[:n]
        z_txt_train = z_txt_train[: len(kept_cap_idx)]
        idx["train"]["image_ids"] = kept_img_ids
        idx["train"]["caption_image_idx"] = kept_cap_idx
        print(f"[sanity] capped train images to {n}, captions to {len(kept_cap_idx)}")

    transform = make_pixel_transform()
    train_ds = ImageDecoderDataset(
        coco_root / "images" / "train2017",
        idx["train"]["image_ids"],
        idx["train"]["caption_image_idx"],
        z_img_train, z_txt_train,
        transform, img_cfg["z_source"],
    )
    val_ds = ImageDecoderDataset(
        coco_root / "images" / "val2017",
        idx["val"]["image_ids"],
        idx["val"]["caption_image_idx"],
        z_img_val, z_txt_val,
        transform, img_cfg["z_source"],
    )
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=img_cfg["batch_size"], shuffle=True,
                              num_workers=img_cfg["num_workers"], drop_last=True,
                              pin_memory=True, persistent_workers=img_cfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=img_cfg["batch_size"], shuffle=False,
                            num_workers=img_cfg["num_workers"], drop_last=False,
                            pin_memory=True, persistent_workers=img_cfg["num_workers"] > 0)

    # Model
    model = ImageDecoder(z_dim=z_img_train.size(-1), hidden_init=img_cfg["hidden_init"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ImageDecoder params={n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=img_cfg["learning_rate"],
                                  weight_decay=img_cfg["weight_decay"])

    # Mixed precision selection (mirrors Code/ModalityGap/main.py:482-493 pattern).
    precision = img_cfg.get("precision", "bf16")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown image_decoder.precision: {precision!r} "
                         f"(expected fp32 | fp16 | bf16)")
    use_amp = (precision != "fp32") and device.type == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[precision]
    # bf16 has fp32-equivalent dynamic range → no GradScaler needed; only fp16 risks underflow.
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and use_amp))
    print(f"[precision] {precision}{' (AMP)' if use_amp else ' (no AMP)'}")

    # LPIPS (optional)
    lpips_fn = None
    if img_cfg["lpips_weight"] > 0:
        import lpips                                                   # noqa: I001
        lpips_fn = lpips.LPIPS(net=img_cfg["lpips_net"]).to(device)
        for p in lpips_fn.parameters():
            p.requires_grad = False
        lpips_fn.eval()
        print(f"[loss] LPIPS({img_cfg['lpips_net']}) enabled, w={img_cfg['lpips_weight']}")

    metrics_log: list[dict] = []
    global_step = 0
    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------
    # Train
    for epoch in range(img_cfg["epochs"]):
        model.train()
        running = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "lpips": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"ep{epoch+1}/{img_cfg['epochs']}")
        for z, x in pbar:
            z = z.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                x_hat = model(z)
                mse = F.mse_loss(x_hat, x)
                l1 = F.l1_loss(x_hat, x) if img_cfg["l1_weight"] > 0 else torch.zeros((), device=device)
                lp = lpips_fn(x_hat, x).mean() if lpips_fn is not None else torch.zeros((), device=device)
                loss = (img_cfg["mse_weight"] * mse
                        + img_cfg["l1_weight"] * l1
                        + img_cfg["lpips_weight"] * lp)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            B = x.size(0)
            running["loss"] += loss.item() * B
            running["mse"] += mse.item() * B
            running["l1"] += l1.item() * B
            running["lpips"] += lp.item() * B
            running["n"] += B
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{mse.item():.4f}")

            # epoch scalar는 reference용 — 매끄러운 1.0 → 2.0 → ... 진행 (wrap 없음).
            step_in_epoch = (global_step - 1) % steps_per_epoch + 1
            wandb.log({
                "loss/total": loss.item(),
                "loss/mse":   mse.item(),
                "loss/l1":    l1.item(),
                "loss/lpips": lp.item(),
                "optim/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + step_in_epoch / steps_per_epoch,
            }, step=global_step)

        N = max(running["n"], 1)
        train_avg = {k: v / N for k, v in running.items() if k != "n"}

        # Val
        val_metrics = evaluate(model, val_loader, device, lpips_fn,
                               max_batches=4 if args.max_train_images else None)
        epoch_log = {"epoch": epoch + 1,
                     **{f"train/{k}": v for k, v in train_avg.items()},
                     **val_metrics}
        metrics_log.append(epoch_log)
        print(f"  [epoch {epoch+1}] " + " ".join(f"{k}={v:.4f}" for k, v in epoch_log.items() if k != "epoch"))

        # Save metrics + sample
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        # Per-epoch wandb scalars
        wandb_epoch = {f"train_epoch/{k}": v for k, v in train_avg.items()}
        wandb_epoch.update(val_metrics)
        wandb_epoch["epoch"] = epoch + 1
        wandb.log(wandb_epoch, step=global_step)

        # Sample 16 val images
        log_samples = ((epoch + 1) %
                       max(int(cfg.get("wandb", {}).get("log_samples_every_n_epochs", 1)), 1) == 0)
        with torch.no_grad():
            model.eval()
            z_s, x_s = next(iter(val_loader))
            z_s = z_s[:16].to(device); x_s = x_s[:16].to(device)
            x_hat_s = model(z_s)
            grid = torch.cat([denorm(x_s), denorm(x_hat_s)], dim=0)
            sample_path = sample_dir / f"epoch_{epoch+1:03d}.png"
            vutils.save_image(grid, sample_path, nrow=16, padding=2)
            if log_samples:
                wandb.log({"samples/val_grid": wandb.Image(
                    str(sample_path),
                    caption=f"epoch {epoch+1}: top=GT, bottom=recon (16 val pairs)",
                )}, step=global_step)

        # Checkpoint
        if (epoch + 1) % img_cfg["save_every_n_epochs"] == 0 or (epoch + 1) == img_cfg["epochs"]:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg},
                       ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    torch.save({"model": model.state_dict(), "epoch": img_cfg["epochs"], "config": cfg},
               ckpt_dir / "final.pt")
    wandb.finish()
    print(f"\n[done] saved to {run_dir}")


if __name__ == "__main__":
    main()
