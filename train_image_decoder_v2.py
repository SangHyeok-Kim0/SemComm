"""Train DiffusionImageDecoder (SD 1.5 VAE + LoRA UNet + dual conditioning) on COCO.

핵심 차이 vs train_image_decoder.py (v1, ConvT):
  - Model: DiffusionImageDecoder (frozen VAE/text encoder, LoRA UNet, projection MLP).
  - Loss: ε-prediction MSE in VAE latent space (model.training_step()이 처리).
  - z_source random_img_txt: 매 image에서 5 captions 중 random k 선택 후 50/50로 z_img[i] vs z_txt[k].
  - Caption stream (M3): text_decoder_run이 비어있지 않으면 cache/captions_{run}_*.pt 로드해서
    encoder_hidden_states에 추가 conditioning. 처음 caption_curriculum_epochs는 GT caption 사용.
  - Trainable: UNet LoRA + z_proj MLP만.

z source policy (v2):
  centroid:       mu = normalize((z_img + z_txt) / 2)
  modality:       z = z_img
  random_img_txt: per-sample 50/50 — z = z_img[i] or z = z_txt[k] (random k of image's 5 captions)

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
import torchvision.transforms as T
import torchvision.utils as vutils
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import DiffusionImageDecoder  # noqa: E402


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
    """[-1, 1] pixel-space (SD VAE expects this range)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),                                       # [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),      # [-1, 1]
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def build_run_name(d_cfg: dict, suffix: str) -> str:
    if d_cfg.get("run_name") and d_cfg["run_name"] not in ("auto", "null", ""):
        return d_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cap_tag = "capON" if d_cfg.get("text_decoder_run") else "capOFF"
    return (f"imgdiff_{d_cfg['z_source']}_{cap_tag}_bs{d_cfg['batch_size']}"
            f"_lr{d_cfg['learning_rate']}_r{d_cfg.get('lora_rank', 8)}_{ts}")


# ---------------------------------------------------------------------------
# Dataset (image-wise iteration)

class ImageDecoderDatasetV2(Dataset):
    """Yield dict per sample: {x, z, gt_caption, cached_caption_or_empty, z_kind}.

    image-wise iteration이라 한 epoch=n_images (118k train). 매 sample마다 random.choice로
    image의 5 captions 중 k를 뽑아 사용 → 매 epoch 다른 k가 나와 caption diversity 자동 보장.

    Caption stream:
      - text_decoder_run이 비어있으면 cached_*가 모두 빈 문자열 → caption stream OFF.
      - 학습 loop이 epoch < curriculum_epochs면 gt_caption 사용, else cached caption 사용
        (z_kind=="zimg"이면 captions_zimg_cache, "ztxt"이면 captions_ztxt_cache).
    """

    def __init__(self, image_dir: Path, image_ids: list[int],
                 caption_image_idx: list[int], caption_texts: list[str],
                 z_img: torch.Tensor, z_txt: torch.Tensor,
                 transform: T.Compose, z_source: str,
                 captions_zimg_cache: list[str] | None = None,
                 captions_ztxt_cache: list[str] | None = None):
        assert z_source in ("centroid", "modality", "random_img_txt")
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.caption_texts = caption_texts
        self.transform = transform
        self.z_img = z_img        # (N_img, D)
        self.z_txt = z_txt        # (N_cap, D)
        self.z_source = z_source
        self.captions_zimg_cache = captions_zimg_cache
        self.captions_ztxt_cache = captions_ztxt_cache

        # Group captions by parent image index.
        per_img: list[list[int]] = [[] for _ in image_ids]
        for cap_j, im_idx in enumerate(caption_image_idx):
            per_img[im_idx].append(cap_j)
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

        cap_indices = self.captions_for_image[i]
        # torch.randint: DataLoader가 worker마다 seed 자동 분기 (random/numpy는 명시 필요).
        cap_pos = torch.randint(len(cap_indices), (1,)).item()
        cap_j = cap_indices[cap_pos]
        z_v = self.z_img[i].float()
        z_t = self.z_txt[cap_j].float()

        if self.z_source == "modality":
            z = z_v
            z_kind = "zimg"
        elif self.z_source == "random_img_txt":
            if torch.rand(1).item() < 0.5:
                z = z_v
                z_kind = "zimg"
            else:
                z = z_t
                z_kind = "ztxt"
        else:  # centroid
            z = F.normalize((z_v + z_t) / 2.0, dim=-1)
            z_kind = "centroid"

        gt_caption = self.caption_texts[cap_j]
        if z_kind == "zimg" and self.captions_zimg_cache is not None:
            cached_caption = self.captions_zimg_cache[i]
        elif z_kind == "ztxt" and self.captions_ztxt_cache is not None:
            cached_caption = self.captions_ztxt_cache[cap_j]
        else:
            # centroid path uses no separate cache; loop will fall back to gt_caption.
            cached_caption = ""

        return {"x": x, "z": z, "gt_caption": gt_caption,
                "cached_caption": cached_caption, "z_kind": z_kind}


def collate(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
        "gt_caption": [b["gt_caption"] for b in batch],
        "cached_caption": [b["cached_caption"] for b in batch],
        "z_kind": [b["z_kind"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Caption stream selector

def select_captions(batch: dict, *, use_cached: bool, caption_stream_on: bool) -> list[str] | None:
    """결정: 어떤 caption을 model에 넣을지.

    - caption stream off (text_decoder_run=""): None (z만 conditioning).
    - on + curriculum 단계: GT caption.
    - on + cached 단계: cached captions; cached가 빈 문자열인 sample은 GT로 fallback.
    """
    if not caption_stream_on:
        return None
    if not use_cached:
        return batch["gt_caption"]
    out = []
    for cached, gt in zip(batch["cached_caption"], batch["gt_caption"]):
        out.append(cached if cached else gt)
    return out


# ---------------------------------------------------------------------------
# DDIM inference helper for samples

@torch.no_grad()
def sample_images(model: DiffusionImageDecoder, scheduler, z: torch.Tensor,
                  captions: list[str] | None, n_steps: int,
                  cfg_scale: float = 1.0) -> torch.Tensor:
    """Return decoded images in [-1, 1]. cfg_scale=1.0 = no CFG (just conditioned)."""
    device = z.device
    B = z.size(0)
    scheduler.set_timesteps(n_steps)

    cond = model.build_condition(z, captions)
    if cfg_scale != 1.0:
        uncond = torch.zeros_like(cond)

    latent = torch.randn(B, 4, 28, 28, device=device, dtype=model.dtype)
    for t in scheduler.timesteps:
        ts = torch.tensor([t.item()] * B, device=device).long()
        eps_c = model.unet(latent, ts, encoder_hidden_states=cond).sample
        if cfg_scale != 1.0:
            eps_u = model.unet(latent, ts, encoder_hidden_states=uncond).sample
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        else:
            eps = eps_c
        latent = scheduler.step(eps, t, latent).prev_sample
    return model.decode_latent_to_image(latent)


# ---------------------------------------------------------------------------
# Eval (latent ε-MSE — 학습과 동일 surrogate metric)

@torch.no_grad()
def evaluate(model: DiffusionImageDecoder, loader: DataLoader, device: torch.device,
             *, use_cached: bool, caption_stream_on: bool,
             max_batches: int | None = None) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        z = batch["z"].to(device, non_blocking=True)
        captions = select_captions(batch, use_cached=use_cached,
                                   caption_stream_on=caption_stream_on)
        loss = model.training_step(x, z, captions, cond_drop_prob=0.0)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return {"val/eps_mse": total_loss / max(n, 1)}


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="base config (encoder/data/wandb/text_decoder)")
    ap.add_argument("--decoder-config", default=str(HERE / "config_diffusion.yaml"),
                    help="image_decoder config (SD-LoRA v2 default)")
    ap.add_argument("--max-train-images", type=int, default=None,
                    help="cap train images for sanity dry-run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--z-source", default=None,
                    choices=["centroid", "modality", "random_img_txt"])
    ap.add_argument("--text-decoder-run", default=None,
                    help="override image_decoder.text_decoder_run (caption stream selector)")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override image_decoder.batch_size (SD UNet needs much smaller batch than ConvT)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.decoder_config)
    img_cfg = cfg["image_decoder"]
    if args.epochs is not None:
        img_cfg["epochs"] = args.epochs
    if args.z_source is not None:
        img_cfg["z_source"] = args.z_source
    if args.text_decoder_run is not None:
        img_cfg["text_decoder_run"] = args.text_decoder_run
    if args.lora_rank is not None:
        img_cfg["lora_rank"] = args.lora_rank
    if args.batch_size is not None:
        img_cfg["batch_size"] = args.batch_size

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

    # ----- Cache load -----
    print("[cache] loading embeddings ...")
    z_img_train = torch.load(cache_dir / "z_img_train.pt", weights_only=False)
    z_txt_train = torch.load(cache_dir / "z_txt_train.pt", weights_only=False)
    z_img_val = torch.load(cache_dir / "z_img_val.pt", weights_only=False)
    z_txt_val = torch.load(cache_dir / "z_txt_val.pt", weights_only=False)
    with open(cache_dir / "index.json") as f:
        idx = json.load(f)

    # Caption stream selector
    text_decoder_run = img_cfg.get("text_decoder_run", "") or ""
    caption_stream_on = bool(text_decoder_run)
    captions_zimg_train = captions_ztxt_train = None
    captions_zimg_val = captions_ztxt_val = None
    if caption_stream_on:
        print(f"[caption] stream ON — loading cache for text_decoder_run={text_decoder_run!r}")
        prefix = f"captions_{text_decoder_run}"
        for name in ["train_zimg", "train_ztxt", "val_zimg", "val_ztxt"]:
            p = cache_dir / f"{prefix}_{name}.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"caption cache missing: {p}\n"
                    f"Run encode_captions.py --text-decoder-run {text_decoder_run} first.")
        captions_zimg_train = torch.load(cache_dir / f"{prefix}_train_zimg.pt", weights_only=False)
        captions_ztxt_train = torch.load(cache_dir / f"{prefix}_train_ztxt.pt", weights_only=False)
        captions_zimg_val   = torch.load(cache_dir / f"{prefix}_val_zimg.pt", weights_only=False)
        captions_ztxt_val   = torch.load(cache_dir / f"{prefix}_val_ztxt.pt", weights_only=False)
        print(f"[caption] loaded: zimg_train={len(captions_zimg_train)}, "
              f"ztxt_train={len(captions_ztxt_train)}")
    else:
        print("[caption] stream OFF (text_decoder_run empty) — z만 conditioning")

    # Optional sanity subset
    if args.max_train_images is not None:
        n = min(args.max_train_images, idx["train"]["n_images"])
        kept_img_ids = idx["train"]["image_ids"][:n]
        kept_cap_idx = [c for c in idx["train"]["caption_image_idx"] if c < n]
        kept_cap_texts = idx["train"]["caption_texts"][: len(kept_cap_idx)]
        z_img_train = z_img_train[:n]
        z_txt_train = z_txt_train[: len(kept_cap_idx)]
        idx["train"]["image_ids"] = kept_img_ids
        idx["train"]["caption_image_idx"] = kept_cap_idx
        idx["train"]["caption_texts"] = kept_cap_texts
        if captions_zimg_train is not None:
            captions_zimg_train = captions_zimg_train[:n]
            captions_ztxt_train = captions_ztxt_train[: len(kept_cap_idx)]
        print(f"[sanity] capped train images to {n}, captions to {len(kept_cap_idx)}")

    transform = make_pixel_transform()
    train_ds = ImageDecoderDatasetV2(
        coco_root / "images" / "train2017",
        idx["train"]["image_ids"],
        idx["train"]["caption_image_idx"],
        idx["train"]["caption_texts"],
        z_img_train, z_txt_train,
        transform, img_cfg["z_source"],
        captions_zimg_cache=captions_zimg_train,
        captions_ztxt_cache=captions_ztxt_train,
    )
    val_ds = ImageDecoderDatasetV2(
        coco_root / "images" / "val2017",
        idx["val"]["image_ids"],
        idx["val"]["caption_image_idx"],
        idx["val"]["caption_texts"],
        z_img_val, z_txt_val,
        transform, img_cfg["z_source"],
        captions_zimg_cache=captions_zimg_val,
        captions_ztxt_cache=captions_ztxt_val,
    )
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=img_cfg["batch_size"], shuffle=True,
                              num_workers=img_cfg["num_workers"], drop_last=True,
                              pin_memory=True, collate_fn=collate,
                              persistent_workers=img_cfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=img_cfg["batch_size"], shuffle=False,
                            num_workers=img_cfg["num_workers"], drop_last=False,
                            pin_memory=True, collate_fn=collate,
                            persistent_workers=img_cfg["num_workers"] > 0)

    # ----- Model -----
    precision = img_cfg.get("precision", "bf16")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown image_decoder.precision: {precision!r}")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[precision]

    lora_rank = img_cfg.get("lora_rank", 8)
    n_z_tokens = img_cfg.get("n_z_tokens", 10)
    print(f"[model] DiffusionImageDecoder LoRA rank={lora_rank}, n_z_tokens={n_z_tokens}, dtype={precision}")
    model = DiffusionImageDecoder(
        z_dim=z_img_train.size(-1),
        n_z_tokens=n_z_tokens,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        caption_conditioning=caption_stream_on,
        dtype=dtype,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={n_trainable/1e6:.2f}M / total={n_total/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=img_cfg["learning_rate"],
        weight_decay=img_cfg["weight_decay"],
    )

    # DDIM scheduler (for sample inference)
    from diffusers import DDIMScheduler  # noqa: E402
    ddim = DDIMScheduler.from_pretrained(model.sd_model_id, subfolder="scheduler")

    curriculum_epochs = int(img_cfg.get("caption_curriculum_epochs", 0))
    cond_drop_prob = float(img_cfg.get("cond_drop_prob", 0.1))
    cfg_scale_sample = float(img_cfg.get("cfg_scale_sample", 1.0))
    sample_steps = int(img_cfg.get("sample_steps", 30))
    print(f"[train] caption_curriculum_epochs={curriculum_epochs}, "
          f"cond_drop_prob={cond_drop_prob}, sample_steps={sample_steps}, "
          f"cfg_scale_sample={cfg_scale_sample}")

    metrics_log: list[dict] = []
    global_step = 0
    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------
    # Train
    for epoch in range(img_cfg["epochs"]):
        use_cached_this_epoch = caption_stream_on and (epoch >= curriculum_epochs)
        cap_mode = "cached_ĉ" if use_cached_this_epoch else ("gt" if caption_stream_on else "off")
        model.train()
        running_loss, running_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"ep{epoch+1}/{img_cfg['epochs']} cap={cap_mode}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            captions = select_captions(batch, use_cached=use_cached_this_epoch,
                                       caption_stream_on=caption_stream_on)

            loss = model.training_step(x, z, captions, cond_drop_prob=cond_drop_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_n += x.size(0)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            step_in_epoch = (global_step - 1) % steps_per_epoch + 1
            wandb.log({
                "loss/eps_mse": loss.item(),
                "optim/learning_rate": optimizer.param_groups[0]["lr"],
                "caption_mode": 1 if use_cached_this_epoch else (0 if caption_stream_on else -1),
                "epoch": epoch + step_in_epoch / steps_per_epoch,
            }, step=global_step)

        train_avg = running_loss / max(running_n, 1)
        val_metrics = evaluate(model, val_loader, device,
                               use_cached=use_cached_this_epoch,
                               caption_stream_on=caption_stream_on,
                               max_batches=4 if args.max_train_images else 50)
        epoch_log = {"epoch": epoch + 1, "train/eps_mse": train_avg, **val_metrics,
                     "caption_mode": cap_mode}
        metrics_log.append(epoch_log)
        print(f"  [epoch {epoch+1}] " + " ".join(
            f"{k}={v if isinstance(v, str) else f'{v:.4f}'}"
            for k, v in epoch_log.items() if k != "epoch"))

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        wb_epoch = {"train_epoch/eps_mse": train_avg, **val_metrics, "epoch": epoch + 1}
        wandb.log(wb_epoch, step=global_step)

        # Sample 8 val images (slow — 30-step DDIM); log every N epochs
        log_samples = ((epoch + 1) %
                       max(int(wb_cfg.get("log_samples_every_n_epochs", 1)), 1) == 0)
        if log_samples:
            model.eval()
            with torch.no_grad():
                v_batch = next(iter(val_loader))
                z_s = v_batch["z"][:8].to(device)
                x_s = v_batch["x"][:8].to(device)
                cap_s = select_captions(
                    {k: v[:8] if isinstance(v, list) else v for k, v in v_batch.items()},
                    use_cached=use_cached_this_epoch,
                    caption_stream_on=caption_stream_on,
                )
                x_hat = sample_images(model, ddim, z_s, cap_s,
                                      n_steps=sample_steps,
                                      cfg_scale=cfg_scale_sample)
                grid = torch.cat([denorm(x_s).float().cpu(),
                                  denorm(x_hat).float().cpu()], dim=0)
                sample_path = sample_dir / f"epoch_{epoch+1:03d}.png"
                vutils.save_image(grid, sample_path, nrow=8, padding=2)
                wandb.log({"samples/val_grid": wandb.Image(
                    str(sample_path),
                    caption=f"epoch {epoch+1}: top=GT, bottom=recon (cap={cap_mode})",
                )}, step=global_step)

        # Checkpoint — LoRA + z_proj only (frozen modules는 sd_model_id로 재로드)
        if (epoch + 1) % img_cfg["save_every_n_epochs"] == 0 or (epoch + 1) == img_cfg["epochs"]:
            from peft import get_peft_model_state_dict
            ckpt = {
                "lora": get_peft_model_state_dict(model.unet),
                "z_proj": model.z_proj.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    # Final ckpt
    from peft import get_peft_model_state_dict
    torch.save({
        "lora": get_peft_model_state_dict(model.unet),
        "z_proj": model.z_proj.state_dict(),
        "epoch": img_cfg["epochs"],
        "config": cfg,
    }, ckpt_dir / "final.pt")
    wandb.finish()
    print(f"\n[done] saved to {run_dir}")


if __name__ == "__main__":
    main()
