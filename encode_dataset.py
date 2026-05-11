"""Cache image/text embeddings for COCO train2017 + val2017 using the frozen Run 1 encoder.

Outputs under cfg['cache_dir']:
  z_img_{train,val}.pt    (N_img, D) fp16, unit-norm
  z_txt_{train,val}.pt    (N_cap, D) fp16, unit-norm
  index.json              per-split metadata (image_ids, caption_image_idx, caption_texts)

Encoder is loaded once. Run 1 was trained inside DataParallel, so 'module.' prefix is
stripped before load. CLIP mean/std normalization mirrors Code/ModalityGap/data.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
MODGAP = HERE.parent / "ModalityGap"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Helpers

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def load_encoder(cfg: dict, device: torch.device):
    """Load Run 1's open_clip model, strip DataParallel prefix, freeze."""
    model_name = cfg["encoder_model"]
    ckpt_path = (
        MODGAP / "runs" / cfg["encoder_run"] / "checkpoints" / cfg["encoder_ckpt_filename"]
    )

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=None, device=device
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"[encoder] {model_name} loaded from {ckpt_path}")
    return model, tokenizer


def make_image_transform() -> T.Compose:
    # Mirrors Code/ModalityGap/data.py:test_transform — must match training-time preprocessing.
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ---------------------------------------------------------------------------
# COCO image side

class CocoImagesOnly(Dataset):
    """Yield (CLIP-normalized image, COCO image_id) for image encoding.
    Skips caption loading at __getitem__ time to keep image-pass batched & fast.
    """
    def __init__(self, image_dir: str, ann_file: str, transform: T.Compose,
                 max_images: int | None = None):
        # CocoCaptions parses the JSON; we use it for both image listing and (later)
        # caption lookup via self.coco.coco.{getAnnIds, loadAnns}.
        self.coco = dset.CocoCaptions(root=image_dir, annFile=ann_file)
        self.transform = transform
        n = len(self.coco) if max_images is None else min(max_images, len(self.coco))
        self.indices = list(range(n))
        self.image_ids = [int(self.coco.ids[i]) for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        idx = self.indices[k]
        img, _captions_unused = self.coco[idx]
        return self.transform(img), int(self.coco.ids[idx])


# ---------------------------------------------------------------------------
# Encoding loops

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _resolve_amp(precision: str, device: torch.device) -> tuple[bool, torch.dtype]:
    if precision not in _DTYPE:
        raise ValueError(f"Unknown encode.precision: {precision!r} (expected fp32 | fp16 | bf16)")
    use_amp = (precision != "fp32") and device.type == "cuda"
    return use_amp, _DTYPE[precision]


@torch.no_grad()
def encode_images(model, dataset: CocoImagesOnly, batch_size: int, num_workers: int,
                  device: torch.device, precision: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    embeds, image_ids = [], []
    use_amp, amp_dtype = _resolve_amp(precision, device)
    for imgs, ids in tqdm(loader, desc="img enc"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            z = model.encode_image(imgs)
        z = z / z.norm(dim=-1, keepdim=True)
        # Cache는 항상 fp16 — storage 절약, 다운스트림 변환 단순화.
        embeds.append(z.detach().to(torch.float16).cpu())
        image_ids.extend(int(i) for i in ids)
    return torch.cat(embeds, 0), image_ids


@torch.no_grad()
def encode_captions(model, tokenizer, captions: list[str], batch_size: int,
                    device: torch.device, precision: str):
    embeds = []
    use_amp, amp_dtype = _resolve_amp(precision, device)
    for i in tqdm(range(0, len(captions), batch_size), desc="txt enc"):
        chunk = captions[i:i + batch_size]
        tokens = tokenizer(chunk).to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            z = model.encode_text(tokens)
        z = z / z.norm(dim=-1, keepdim=True)
        embeds.append(z.detach().to(torch.float16).cpu())
    return torch.cat(embeds, 0)


def collect_captions(coco: dset.CocoCaptions, image_ids: list[int]) -> tuple[list[str], list[int]]:
    """For each image_id (in order), enumerate all of its captions.
    Returns:
        cap_texts: flat list of caption strings.
        cap_img_idx: same length; cap_img_idx[j] = index of parent image in image_ids.
    """
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    cap_texts, cap_img_idx = [], []
    for iid in image_ids:
        ann_ids = coco.coco.getAnnIds(imgIds=iid)
        anns = coco.coco.loadAnns(ann_ids)
        for ann in anns:
            cap_texts.append(ann["caption"])
            cap_img_idx.append(id_to_idx[iid])
    return cap_texts, cap_img_idx


# ---------------------------------------------------------------------------
# Per-split orchestration

def process_split(name: str, image_dir: str, ann_file: str, model, tokenizer,
                  cfg: dict, device: torch.device, cache_dir: Path,
                  max_images: int | None):
    print(f"\n[split:{name}] image_dir={image_dir}")
    img_ds = CocoImagesOnly(image_dir, ann_file, make_image_transform(), max_images=max_images)

    z_img, image_ids = encode_images(
        model, img_ds, cfg["batch_size"], cfg["num_workers"], device, cfg["precision"]
    )
    assert image_ids == img_ds.image_ids, "image_id ordering mismatch"
    print(f"  z_img: {tuple(z_img.shape)}, ids: {len(image_ids)}")

    cap_texts, cap_img_idx = collect_captions(img_ds.coco, image_ids)
    print(f"  captions: {len(cap_texts)}")

    z_txt = encode_captions(
        model, tokenizer, cap_texts, cfg["batch_size"], device, cfg["precision"]
    )
    print(f"  z_txt: {tuple(z_txt.shape)}")

    torch.save(z_img, cache_dir / f"z_img_{name}.pt")
    torch.save(z_txt, cache_dir / f"z_txt_{name}.pt")
    return {
        "image_ids": image_ids,
        "caption_image_idx": cap_img_idx,
        "caption_texts": cap_texts,
        "n_images": len(image_ids),
        "n_captions": len(cap_texts),
        "embed_dim": int(z_img.size(-1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--max-images", type=int, default=None,
                    help="limit images per split (smoke testing)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    enc_cfg = cfg["encode"]
    device = torch.device(f"cuda:{enc_cfg['device_id']}" if torch.cuda.is_available() else "cpu")

    coco_root = resolve_path(cfg["coco_root"], HERE)
    cache_dir = resolve_path(cfg["cache_dir"], HERE)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": (
            coco_root / "images" / "train2017",
            coco_root / "annotations" / "captions_train2017.json",
        ),
        "val": (
            coco_root / "images" / "val2017",
            coco_root / "annotations" / "captions_val2017.json",
        ),
    }

    model, tokenizer = load_encoder(cfg, device)

    # Merge with existing index.json so re-running one split doesn't wipe the other.
    index_path = cache_dir / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

    for sp in args.splits:
        img_dir, ann = split_paths[sp]
        index[sp] = process_split(
            sp, str(img_dir), str(ann), model, tokenizer, enc_cfg, device,
            cache_dir, max_images=args.max_images,
        )

    with open(index_path, "w") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"\n[done] cache at {cache_dir}/")


if __name__ == "__main__":
    main()
