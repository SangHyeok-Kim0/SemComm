"""Train ClipCap-style TextDecoder (frozen Qwen + Transformer mapper) on cached COCO embeddings.

  z source policy:
    centroid:       mu = normalize((z_img[i] + z_txt[j]) / 2) for caption j with parent image i
    modality:       z = z_txt[j]   (image-text 짝의 caption 측 임베딩만 사용)
    random_img_txt: per-sample 50/50 — z = z_img[i] or z = z_txt[j].
                    GT는 항상 caption_texts[j]. caption-wise iteration이라
                    한 image의 5 captions 각각이 별도 sample로 노출되어 image-side
                    선택 시 자연스럽게 one-to-many supervision이 발생.

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


def build_run_name(d_cfg: dict, suffix: str, name_suffix: str = "") -> str:
    if d_cfg.get("run_name") and d_cfg["run_name"] not in ("auto", "null", ""):
        return d_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    extra = f"_{name_suffix}" if name_suffix else ""
    return (f"{suffix}_{d_cfg['z_source']}_bs{d_cfg['batch_size']}"
            f"_lr{d_cfg['learning_rate']}_K{d_cfg['prefix_len']}{extra}_{ts}")


# ---------------------------------------------------------------------------
# Dataset (per-caption sampling)

class TextDecoderDataset(Dataset):
    """Yield (z, caption_text). One sample per caption (train2017 ~590k samples)."""

    def __init__(self, caption_image_idx: list[int], caption_texts: list[str],
                 z_img: torch.Tensor, z_txt: torch.Tensor, z_source: str):
        assert z_source in ("centroid", "modality", "random_img_txt")
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
        elif self.z_source == "random_img_txt":
            # torch.rand: DataLoader worker마다 자동으로 seed 분리됨 (random/numpy는 명시 필요).
            if torch.rand(1).item() < 0.5:
                im_idx = self.caption_image_idx[j]
                z = self.z_img[im_idx].float()
            else:
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
def compute_caption_metrics(model: TextDecoder,
                            z_in: torch.Tensor,
                            references: list[list[str]],
                            beam_size: int, max_new_tokens: int,
                            batch_size: int = 32) -> dict[str, float]:
    """val z (input) + multi-reference GT captions → BLEU-1..4 / ROUGE-L / CIDEr.

    Args:
        z_in: (N, D) val sample 의 z (zimg 또는 ztxt).
        references: 길이 N. 각 원소 = 해당 sample 의 모든 GT captions (multi-ref).
        beam_size, max_new_tokens: generation 설정.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    model.eval()
    gens: list[str] = []
    for s in range(0, z_in.size(0), batch_size):
        gens.extend(model.generate(z_in[s:s + batch_size],
                                    beam_size=beam_size, max_new_tokens=max_new_tokens))
    # pycocoevalcap 입력 포맷
    gts_dict = {i: [r.lower().strip() for r in refs] for i, refs in enumerate(references)}
    res_dict = {i: [g.lower().strip() or "(empty)"] for i, g in enumerate(gens)}
    out: dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts_dict, res_dict)
    for i, b in enumerate(bleu, start=1):
        out[f"BLEU-{i}"] = float(b)
    rouge, _ = Rouge().compute_score(gts_dict, res_dict)
    out["ROUGE-L"] = float(rouge)
    cider, _ = Cider().compute_score(gts_dict, res_dict)
    out["CIDEr"] = float(cider)
    return out


def prepare_eval_subset(z_img_val: torch.Tensor, z_txt_val: torch.Tensor,
                        val_caption_image_idx: list[int], val_caption_texts: list[str],
                        n_samples: int, seed: int = 42):
    """eval subset 구성: image-wise 로 n_samples 개 image 선택, 각 image 의 5 captions 를 multi-reference 로 묶음.

    Returns:
        z_img_sub: (n, D), each row = z_img[selected_image_i]
        z_txt_sub: (n, D), each row = z_txt[first caption of selected_image_i]
        refs: list of length n, refs[k] = [caption_1, caption_2, ...] (multi-ref for image k)
    """
    # image_idx → all caption_idxs
    img_to_caps: dict[int, list[int]] = {}
    for cap_idx, im_idx in enumerate(val_caption_image_idx):
        img_to_caps.setdefault(im_idx, []).append(cap_idx)
    available = sorted(img_to_caps.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(available), generator=g).tolist()
    selected = [available[p] for p in perm[:n_samples]]

    z_img_sub = torch.stack([z_img_val[i].float() for i in selected])
    z_txt_sub = torch.stack([z_txt_val[img_to_caps[i][0]].float() for i in selected])
    refs = [[val_caption_texts[j] for j in img_to_caps[i]] for i in selected]
    return z_img_sub, z_txt_sub, refs


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
    ap.add_argument("--z-source", default=None,
                    choices=["centroid", "modality", "random_img_txt"])
    ap.add_argument("--resume", default=None,
                    help="기존 run 디렉터리 경로. 지정 시 final.pt mapper를 로드해 "
                         "ckpt 저장 시점 epoch 이후로 --epochs 만큼 추가 학습. "
                         "config.json/기존 metrics/checkpoints/samples 모두 보존. "
                         "Optimizer state는 ckpt에 없어 AdamW 모멘트는 0부터 재시작.")
    ap.add_argument("--cache-dir", default=None,
                    help="override cfg['cache_dir']. Stage 2 (Standard CLIP) 시 사용.")
    ap.add_argument("--run-name-suffix", default="",
                    help="run_name 에 붙일 suffix (예: 'std_clip'). cache 종류 추적용.")
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

    cache_dir = resolve_path(args.cache_dir if args.cache_dir else cfg["cache_dir"], HERE)
    runs_root = resolve_path(cfg["runs_root"], HERE)
    print(f"[cache_dir] {cache_dir}")

    # Resume: 기존 run_dir 재사용, config/metrics/checkpoints 모두 보존.
    start_epoch = 0
    resume_ckpt = None
    metrics_log: list[dict] = []
    if args.resume:
        run_dir = resolve_path(args.resume, HERE)
        ckpt_dir = run_dir / "checkpoints"
        model_dir = ckpt_dir / "models"
        eval_metrics_dir = ckpt_dir / "eval_metrics"
        # Backward compat: 옛 run 은 checkpoints/epoch_NNN.pt + final.pt 였음.
        # 새 구조 (models/) 가 없으면 fallback 으로 ckpt_dir 직접 사용.
        final_pt_candidates = [model_dir / "final.pt", ckpt_dir / "final.pt"]
        final_pt = next((p for p in final_pt_candidates if p.exists()), None)
        if final_pt is None:
            raise FileNotFoundError(f"resume: final.pt 없음 (tried {final_pt_candidates})")
        resume_ckpt = torch.load(final_pt, map_location="cpu", weights_only=False)
        start_epoch = int(resume_ckpt["epoch"])
        # 새 구조로 마이그레이션 (없으면 생성)
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_dir.mkdir(parents=True, exist_ok=True)
        backup_path = model_dir / f"final_epoch_{start_epoch:03d}.pt"
        if not backup_path.exists():
            import shutil
            shutil.copy2(final_pt, backup_path)
            print(f"[resume] backed up final.pt → models/{backup_path.name}")
        # 기존 metrics 로드 — 새 epoch 결과는 append.
        metrics_json = run_dir / "metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                metrics_log = json.load(f)
        run_name = run_dir.name + "_resume"
        print(f"[resume] from epoch {start_epoch}, will train "
              f"{text_cfg['epochs']} more epoch(s) → target epoch "
              f"{start_epoch + text_cfg['epochs']}")
    else:
        run_name = build_run_name(text_cfg, suffix="txt", name_suffix=args.run_name_suffix)
        run_dir = runs_root / run_name
        ckpt_dir = run_dir / "checkpoints"
        model_dir = ckpt_dir / "models"
        eval_metrics_dir = ckpt_dir / "eval_metrics"
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[run] {run_dir}")
    print(f"[dirs] models → {model_dir}, eval_metrics → {eval_metrics_dir}")

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

    # BLEU/CIDEr eval subset (image-wise, multi-ref) — 매 평가 epoch 에서 재사용
    eval_n = int(text_cfg.get("eval_metrics_n_samples", 500))
    z_img_eval, z_txt_eval, eval_refs = prepare_eval_subset(
        z_img_val, z_txt_val,
        idx["val"]["caption_image_idx"], idx["val"]["caption_texts"],
        n_samples=eval_n, seed=text_cfg.get("seed", 42),
    )
    z_img_eval = z_img_eval.to(device)
    z_txt_eval = z_txt_eval.to(device)
    print(f"[eval] BLEU/CIDEr subset: {eval_n} images (multi-ref)")

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

    if resume_ckpt is not None:
        model.mapper.load_state_dict(resume_ckpt["mapper"])
        print(f"[resume] mapper loaded from final.pt (saved epoch={start_epoch})")

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

    global_step = 0
    steps_per_epoch = len(train_loader)
    # 비-resume 시 start_epoch=0이라 total_epochs == text_cfg["epochs"] (기존 동작과 동일).
    total_epochs = start_epoch + text_cfg["epochs"]

    # ------------------------------------------------------------------
    # Train
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss, running_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"ep{epoch+1}/{total_epochs}")
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

        epoch_log = {"epoch": epoch + 1, "train/ce": train_avg, **val_metrics}
        metrics_log.append(epoch_log)
        print(f"  [epoch {epoch+1}] " + " ".join(f"{k}={v:.4f}" for k, v in epoch_log.items() if k != "epoch"))

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        # Per-epoch wandb scalars (val CE 만, BLEU 는 평가 epoch 에서 추가)
        wandb_epoch = {"train_epoch/ce": train_avg, **val_metrics, "epoch": epoch + 1}

        # 지정된 epoch 마다 BLEU/CIDEr 측정 — z_img / z_txt 분리 + 저장
        eval_every = int(text_cfg.get("eval_metrics_every_n_epochs", text_cfg.get("save_every_n_epochs", 1)))
        do_eval_metrics = (epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs
        if do_eval_metrics:
            print(f"  [eval-metrics] computing BLEU/CIDEr on {eval_n} samples ...")
            beam = int(text_cfg.get("beam_size", 1))
            mnt = int(text_cfg.get("max_new_tokens", 20))
            metrics_zimg = compute_caption_metrics(model, z_img_eval, eval_refs, beam, mnt)
            metrics_ztxt = compute_caption_metrics(model, z_txt_eval, eval_refs, beam, mnt)
            eval_payload = {
                "epoch": epoch + 1,
                "n_samples": eval_n,
                "zimg": metrics_zimg,
                "ztxt": metrics_ztxt,
            }
            with open(eval_metrics_dir / f"epoch_{epoch+1:03d}.json", "w") as f:
                json.dump(eval_payload, f, indent=2)
            # 출력 표
            print(f"    zimg: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_zimg.items()))
            print(f"    ztxt: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics_ztxt.items()))
            # wandb log — eval/<metric>_<modality> 형식. BLEU 는 1, 3 만 (2,4 는 JSON 에는 저장됨).
            wandb_metric_keys = ["BLEU-1", "BLEU-3", "ROUGE-L", "CIDEr"]
            for k in wandb_metric_keys:
                if k in metrics_zimg:
                    wandb_epoch[f"eval/{k}_zimg"] = metrics_zimg[k]
                if k in metrics_ztxt:
                    wandb_epoch[f"eval/{k}_ztxt"] = metrics_ztxt[k]

        wandb.log(wandb_epoch, step=global_step)

        if (epoch + 1) % text_cfg["save_every_n_epochs"] == 0 or (epoch + 1) == total_epochs:
            torch.save({"mapper": model.mapper.state_dict(),
                        "epoch": epoch + 1, "config": cfg},
                       model_dir / f"epoch_{epoch+1:03d}.pt")

    torch.save({"mapper": model.mapper.state_dict(),
                "epoch": total_epochs, "config": cfg},
               model_dir / "final.pt")

    wandb.finish()
    print(f"\n[done] saved to {run_dir}")


if __name__ == "__main__":
    main()
