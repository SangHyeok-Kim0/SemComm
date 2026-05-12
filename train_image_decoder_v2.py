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


def build_run_name(d_cfg: dict, suffix: str, name_suffix: str = "") -> str:
    if d_cfg.get("run_name") and d_cfg["run_name"] not in ("auto", "null", ""):
        return d_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cap_tag = "capON" if d_cfg.get("text_decoder_run") else "capOFF"
    extra = f"_{name_suffix}" if name_suffix else ""
    return (f"imgdiff_{d_cfg['z_source']}_{cap_tag}_bs{d_cfg['batch_size']}"
            f"_lr{d_cfg['learning_rate']}_r{d_cfg.get('lora_rank', 8)}{extra}_{ts}")


# ---------------------------------------------------------------------------
# Dataset (image-wise iteration)

class ImageDecoderDatasetV2(Dataset):
    """Yield dict per sample: {x, z, gt_caption, cached_caption_or_empty, z_kind}.

    iteration_unit:
      - "image" (default, 118k/epoch train): 매 sample마다 image i의 5 captions 중 random k를
        뽑아 사용 → 매 epoch 다른 k가 나와 caption diversity 자동 보장. wall-clock ~51분/epoch.
      - "caption" (590k/epoch train): 매 sample이 특정 caption j와 그 부모 image. 한 image가
        epoch당 5번 (5 captions로) 노출 → caption diversity 명시적, image load도 5번. wall-clock
        ~4.2h/epoch (5배).

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
                 captions_ztxt_cache: list[str] | None = None,
                 iteration_unit: str = "image"):
        assert z_source in ("centroid", "modality", "random_img_txt")
        assert iteration_unit in ("image", "caption")
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.caption_image_idx = caption_image_idx
        self.caption_texts = caption_texts
        self.transform = transform
        self.z_img = z_img        # (N_img, D)
        self.z_txt = z_txt        # (N_cap, D)
        self.z_source = z_source
        self.captions_zimg_cache = captions_zimg_cache
        self.captions_ztxt_cache = captions_ztxt_cache
        self.iteration_unit = iteration_unit

        # Group captions by parent image index (image-wise iteration용; caption-wise도
        # 검증용으로 동일하게 생성하지만 사용 안 함).
        per_img: list[list[int]] = [[] for _ in image_ids]
        for cap_j, im_idx in enumerate(caption_image_idx):
            per_img[im_idx].append(cap_j)
        self.valid = [i for i, lst in enumerate(per_img) if len(lst) > 0]
        self.captions_for_image = per_img

    def __len__(self) -> int:
        if self.iteration_unit == "image":
            return len(self.valid)
        return len(self.caption_texts)

    def _file(self, iid: int) -> Path:
        return self.image_dir / f"{iid:012d}.jpg"

    def __getitem__(self, k: int):
        if self.iteration_unit == "image":
            i = self.valid[k]
            cap_indices = self.captions_for_image[i]
            # torch.randint: DataLoader가 worker마다 seed 자동 분기.
            cap_pos = torch.randint(len(cap_indices), (1,)).item()
            cap_j = cap_indices[cap_pos]
        else:  # caption-wise: k가 caption index 그대로
            cap_j = k
            i = self.caption_image_idx[cap_j]

        iid = self.image_ids[i]
        img = Image.open(self._file(iid)).convert("RGB")
        x = self.transform(img)

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
def fetch_val_samples_all_z(val_ds: "ImageDecoderDatasetV2", n: int,
                            device: torch.device, dtype: torch.dtype) -> dict:
    """Fetch n val samples returning x_gt + z_img + z_txt + cached caps for both modes.

    학습 시 random sampling과 달리 inference에서는 동일 image에 대한 zimg/ztxt를 모두 보고
    싶기 때문에 별도 fetch helper로 분리. caption-wise iteration도 image-wise iteration도
    image i의 첫 caption(captions_for_image[i][0])을 selection convention으로 사용.
    """
    xs, z_imgs, z_txts, gt_caps, zimg_caps, ztxt_caps = [], [], [], [], [], []
    n_actual = min(n, len(val_ds.valid))
    for k in range(n_actual):
        i = val_ds.valid[k]
        iid = val_ds.image_ids[i]
        img = Image.open(val_ds._file(iid)).convert("RGB")
        xs.append(val_ds.transform(img))
        cap_j = val_ds.captions_for_image[i][0]
        z_imgs.append(val_ds.z_img[i].float())
        z_txts.append(val_ds.z_txt[cap_j].float())
        gt_caps.append(val_ds.caption_texts[cap_j])
        zimg_caps.append(val_ds.captions_zimg_cache[i] if val_ds.captions_zimg_cache else "")
        ztxt_caps.append(val_ds.captions_ztxt_cache[cap_j] if val_ds.captions_ztxt_cache else "")
    return {
        "x_gt": torch.stack(xs).to(device).to(dtype),
        "z_img": torch.stack(z_imgs).to(device),
        "z_txt": torch.stack(z_txts).to(device),
        "gt_caps": gt_caps,
        "zimg_caps": zimg_caps,
        "ztxt_caps": ztxt_caps,
    }


@torch.no_grad()
def generate_epoch_inference_grids(model: DiffusionImageDecoder, scheduler,
                                    val_ds: "ImageDecoderDatasetV2", epoch: int,
                                    sample_dir: Path, device: torch.device,
                                    dtype: torch.dtype, sample_steps: int = 30,
                                    n: int = 8, use_cached_caption: bool = True,
                                    caption_stream_on: bool = True) -> Path:
    """Save 9 combos (3 z_source × 3 cfg_scale) per-epoch under sample_dir/epoch_NNN/.

    z_source: random (per-sample 50/50 zimg vs ztxt), zimg (image-side), ztxt (text-side)
    cfg_scale: 1.0 (no guidance), 3.0 (SD standard), 7.5 (SD web demo default)

    각 PNG는 (top n=GT, bottom n=복원) grid. 학습 process와 동일 흐름으로 caption 결정:
    epoch < curriculum_epochs이면 GT caption, else cached ĉ (zimg_caps or ztxt_caps).
    """
    epoch_dir = sample_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    s = fetch_val_samples_all_z(val_ds, n=n, device=device, dtype=dtype)
    n_actual = s["z_img"].size(0)

    # Default 6 combos: zimg/ztxt × cfg 1.0/3.0/7.5. random은 평균/혼합 효과만 보여
    # 평가 가치 적음 — 학습 default 동작 reproduction이 필요하면 wandb 대표 sample을 별도로 생성.
    z_sources = [
        ("zimg", s["z_img"], s["zimg_caps"]),
        ("ztxt", s["z_txt"], s["ztxt_caps"]),
    ]
    cfg_scales = [1.0, 3.0, 7.5]

    for zsrc_name, z_in, cached_caps in z_sources:
        for cfg_scale in cfg_scales:
            if not caption_stream_on:
                captions_for_inf = None
            elif use_cached_caption:
                captions_for_inf = [c if c else g
                                    for c, g in zip(cached_caps, s["gt_caps"])]
            else:
                captions_for_inf = s["gt_caps"]
            x_hat = sample_images(model, scheduler, z_in, captions_for_inf,
                                  n_steps=sample_steps, cfg_scale=cfg_scale)
            grid = torch.cat([denorm(s["x_gt"]).float().cpu(),
                              denorm(x_hat).float().cpu()], dim=0)
            out_path = epoch_dir / f"recon_{zsrc_name}_cfg{cfg_scale}.png"
            vutils.save_image(grid, out_path, nrow=n_actual, padding=2)
    return epoch_dir


@torch.no_grad()
def sample_images(model: DiffusionImageDecoder, scheduler, z: torch.Tensor,
                  captions: list[str] | None, n_steps: int,
                  cfg_scale: float = 1.0) -> torch.Tensor:
    """Return decoded images in [-1, 1]. cfg_scale=1.0 = no CFG (just conditioned)."""
    device = z.device
    B = z.size(0)
    scheduler.set_timesteps(n_steps)

    # CFG batching: cond + uncond 를 한 batch 로 합쳐 UNet 1번 호출. UNet call 수 절반.
    use_cfg = cfg_scale != 1.0
    cond = model.build_condition(z, captions)
    ctx = torch.cat([torch.zeros_like(cond), cond], dim=0) if use_cfg else cond

    latent = torch.randn(B, 4, 28, 28, device=device, dtype=model.dtype)
    for t in scheduler.timesteps:
        ts = torch.tensor([t.item()] * B, device=device).long()
        if use_cfg:
            latent_2x = torch.cat([latent, latent], dim=0)
            ts_2x = torch.cat([ts, ts], dim=0)
            eps_2x = model.unet(latent_2x, ts_2x, encoder_hidden_states=ctx).sample
            eps_u, eps_c = eps_2x.chunk(2, dim=0)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        else:
            eps = model.unet(latent, ts, encoder_hidden_states=ctx).sample
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
    ap.add_argument("--iteration-unit", default=None, choices=["image", "caption"],
                    help="dataset iteration unit. image=118k/ep (default), caption=590k/ep (5x time)")
    ap.add_argument("--cache-dir", default=None,
                    help="override cfg['cache_dir']. Stage 2 (Standard CLIP) 시 사용.")
    ap.add_argument("--run-name-suffix", default="",
                    help="run_name 에 붙일 suffix (예: 'std_clip'). encoder 종류 추적용.")
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
    if args.iteration_unit is not None:
        img_cfg["iteration_unit"] = args.iteration_unit

    random.seed(img_cfg["seed"])
    torch.manual_seed(img_cfg["seed"])

    device = torch.device(f"cuda:{img_cfg['device_id']}" if torch.cuda.is_available() else "cpu")
    coco_root = resolve_path(cfg["coco_root"], HERE)
    cache_dir = resolve_path(args.cache_dir if args.cache_dir else cfg["cache_dir"], HERE)
    runs_root = resolve_path(cfg["runs_root"], HERE)
    print(f"[cache_dir] {cache_dir}")

    run_name = build_run_name(img_cfg, suffix="img", name_suffix=args.run_name_suffix)
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
    iteration_unit = img_cfg.get("iteration_unit", "image")
    train_ds = ImageDecoderDatasetV2(
        coco_root / "images" / "train2017",
        idx["train"]["image_ids"],
        idx["train"]["caption_image_idx"],
        idx["train"]["caption_texts"],
        z_img_train, z_txt_train,
        transform, img_cfg["z_source"],
        captions_zimg_cache=captions_zimg_train,
        captions_ztxt_cache=captions_ztxt_train,
        iteration_unit=iteration_unit,
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
        iteration_unit=iteration_unit,
    )
    print(f"[data] iteration_unit={iteration_unit}, train={len(train_ds)}, val={len(val_ds)}")

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

        # Per-epoch 9 inference grids (3 z_source × 3 cfg). 학습 throughput ~2% overhead.
        # 파일: samples/epoch_NNN/recon_{random,zimg,ztxt}_cfg{1.0,3.0,7.5}.png
        model.eval()
        epoch_sample_dir = generate_epoch_inference_grids(
            model, ddim, val_ds, epoch + 1, sample_dir, device, dtype,
            sample_steps=sample_steps, n=8,
            use_cached_caption=use_cached_this_epoch,
            caption_stream_on=caption_stream_on,
        )

        # Wandb upload — 매 epoch 6장 다 올리면 dashboard 폭주. 대표 2장만 upload:
        # zimg+cfg3.0 (in-modal baseline) + ztxt+cfg3.0 (cross-modal swap, contribution 핵심).
        log_samples = ((epoch + 1) %
                       max(int(wb_cfg.get("log_samples_every_n_epochs", 1)), 1) == 0)
        if log_samples:
            for tag in ("zimg", "ztxt"):
                rep_path = epoch_sample_dir / f"recon_{tag}_cfg3.0.png"
                if rep_path.exists():
                    wandb.log({f"samples/{tag}_cfg3.0": wandb.Image(
                        str(rep_path),
                        caption=f"epoch {epoch+1}: {tag}+cfg3.0 (cap={cap_mode})",
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
