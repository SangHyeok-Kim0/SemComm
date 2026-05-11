# SemComm — Semantic Communication on top of ModalityGap

ModalityGap (ICLR 2026) 학습 결과 위에서 Semantic Communication을 실험하는 코드.

기준 체크포인트: `Code/ModalityGap/runs/bs128_lr1e-4_ep100_Tfix_fp16_a50-50_b20-50_20260505-034844/`
인코더 정책: **Freeze**. 디코더/채널 모듈만 학습.

## 파이프라인 개요

```
[image] ──► (frozen RN50 image enc) ──┐
                                       │ z (1024-D, unit-norm)
                                       ├──► z̃ ──► [image dec] ──► reconstructed image
[text]  ──► (frozen RN50 text  enc) ──┘   (channel)  [text dec ] ──► reconstructed caption
```

송신측은 단일 z (image 또는 text)를 보내고 수신측은 modality별 디코더 스왑으로 복원. 본 저장소는 **frozen encoder + modality-agnostic decoder 학습 + cross-modal decoding 평가**를 단계적으로 진행한다.

## Config 구조 (3 파일 분리, 2026-05-11)

```
config.yaml             — 공통 (encoder/data 경로/wandb/encode/text_decoder)
config_convt.yaml       — Image decoder v1 (ConvTranspose) hparam
config_diffusion.yaml   — Image decoder v2 (Stable Diffusion 1.5 + LoRA) hparam
```

image decoder 학습 스크립트는 `--config config.yaml --decoder-config <convt|diffusion>.yaml`을 받아 두 dict를 deep-merge한다. 각 script의 default는 자신과 맞는 decoder-config:
- `train_image_decoder.py` → `config_convt.yaml`
- `train_image_decoder_v2.py` → `config_diffusion.yaml`

`train_text_decoder.py` / `encode_captions.py` / `encode_dataset.py`는 `config.yaml` 한 파일만 읽음.

## Phase A — Decoder-swap 가능성 진단 (학습 없음)

```bash
python Code/SemComm/phase_a_diagnose.py
```

산출: `results/phase_a/` (json + csv + png). 자세한 내용은 `phase_a_diagnose.py` 참조.

## Phase B — Decoder 학습

### Step 1 — 임베딩 캐시

```bash
# 전체 (train ~118k images / ~590k captions, val ~5k / ~25k)
python Code/SemComm/encode_dataset.py

# 샘플 (smoke test)
python Code/SemComm/encode_dataset.py --max-images 256 --splits train val
```

산출:
- `cache/z_img_{train,val}.pt`  — (N_img, 1024) fp16, unit-norm
- `cache/z_txt_{train,val}.pt`  — (N_cap, 1024) fp16, unit-norm
- `cache/index.json`            — image_ids, caption_image_idx, caption_texts

### Step 2 — Text Decoder (Qwen3-0.6B-Base + Transformer mapper)

```bash
# Default config.yaml (text_decoder.z_source=random_img_txt, 15 epoch)
python Code/SemComm/train_text_decoder.py

# Sanity
python Code/SemComm/train_text_decoder.py \
    --max-train-captions 5000 --epochs 2 --batch-size 32 --z-source centroid

# Resume
python Code/SemComm/train_text_decoder.py \
  --resume runs/txt_random_img_txt_..._<ts> --epochs 5
```

z_source 옵션:
- `centroid`: z = normalize((z_img[i] + z_txt[j]) / 2)
- `modality`: z = z_txt[j] (caption-side only)
- `random_img_txt`: per-sample 50/50 z_img[i] vs z_txt[j] **(proposed, modality-agnostic)**

LM(Qwen3-0.6B) frozen, mapper(111M) + projection만 학습. 최초 실행 시 ~1.2GB 가중치 자동 다운로드. EOS-aware 변종은 `train_text_decoder_eos.py`.

### Step 3 — Caption Cache (M3 stream 준비, v2 학습 전 필수)

frozen text decoder의 `generate()`를 사전 호출해 image decoder의 caption conditioning에 쓸 captions를 cache.

```bash
python Code/SemComm/encode_captions.py \
    --text-decoder-run txt_random_img_txt_..._<ts>
```

산출 (~40MB 합계, ~25분 on GB10):
- `cache/captions_{text_decoder_run}_train_zimg.pt`  — list[str] × 118k
- `cache/captions_{text_decoder_run}_train_ztxt.pt`  — list[str] × 591k
- `cache/captions_{text_decoder_run}_val_zimg.pt`    — list[str] × 5k
- `cache/captions_{text_decoder_run}_val_ztxt.pt`    — list[str] × 25k

cache 파일명에 text decoder run name이 박혀 있어 어떤 decoder의 caption인지 즉시 식별. cross-encoder mismatch (Run 1 z를 Standard CLIP decoder에 잘못 넣는 등) 방지.

### Step 4a — Image Decoder v1 (ConvTranspose 베이스라인, [-1, 1] 픽셀)

```bash
# 풀 학습 (config_convt.yaml 기준)
python Code/SemComm/train_image_decoder.py

# 옵션
python Code/SemComm/train_image_decoder.py \
    --max-train-images 1000 --epochs 3 --z-source modality
```

Loss: `mse_w·MSE + l1_w·L1 + lpips_w·LPIPS` ([config_convt.yaml](config_convt.yaml)으로 가중치 조절, `lpips_w=0`이면 LPIPS 미로드).

### Step 4b — Image Decoder v2 (SD 1.5 + LoRA, proposed)

dual conditioning: z (1024-D → K virtual tokens) + caption ĉ (SD CLIP-L → 77 tokens) → UNet cross-attn.

```bash
# 풀 학습 (config_diffusion.yaml의 text_decoder_run 등록 필수)
python Code/SemComm/train_image_decoder_v2.py

# CLI override (config 안 건드리고 즉시 실행)
python Code/SemComm/train_image_decoder_v2.py \
    --z-source random_img_txt \
    --text-decoder-run txt_random_img_txt_..._<ts> \
    --batch-size 16 --epochs 10

# Sanity
python Code/SemComm/train_image_decoder_v2.py \
    --max-train-images 20 --epochs 1 --batch-size 4 --text-decoder-run ""
```

z_source 옵션:
- `centroid`: normalize((z_img + z_txt) / 2)
- `modality`: z_img (image-side만)
- `random_img_txt`: per-image 5 captions 중 random k → 50/50로 z_img[i] vs z_txt[k] **(proposed)**

Caption stream (M3):
- `text_decoder_run` 비어있으면 caption stream OFF (z만 conditioning, M3 off cell)
- 비어있지 않으면 cache/captions_{run}_*.pt 로드 (Step 3에서 생성된 것)
- `caption_curriculum_epochs=5`: 처음 5 epoch는 GT caption 사용, 6 epoch부터 cached ĉ로 swap (학습-추론 distribution 차이 완화)

Trainable: UNet cross-attn LoRA (rank 8) + projection MLP = 약 19M params. VAE/text encoder/base UNet 모두 frozen.
Loss: ε-prediction MSE in VAE latent (28×28×4). pixel-level metric은 미사용.

### Step 5 — 시각화 & 평가

```bash
# Caption 시각화 (이미지 | GT 5개 | GEN 1개)
python Code/SemComm/visualize_captions.py --run-dir runs/txt_..._<ts> --n 10

# Caption 정량 평가 (BLEU / ROUGE-L / CIDEr, 5 시나리오)
pip install pycocoevalcap   # 1회
python Code/SemComm/eval_captions.py --run-dir runs/txt_..._<ts>
```

자세한 옵션은 `visualize_captions.py --help` / `eval_captions.py --help`.

이미지 디코더 평가는 v2 학습 run의 `samples/epoch_*.png` (매 epoch GT+복원 grid) 와 wandb의 `samples/val_grid`로 시각 검증. 정량 평가 script는 향후 추가 (FID/CLIP-score/LPIPS, 5k val 기준).

### 산출물 (각 학습 run)

```
runs/<run_name>/
├── config.json                # 학습 시작 시 cfg dump (text_decoder_run 등 dependency 기록)
├── checkpoints/
│   ├── epoch_NNN.pt
│   └── final.pt
├── samples/                   # image: PNG (위=GT, 아래=복원), text: GT/GEN .txt
├── wandb/                     # wandb 로컬 캐시
└── metrics.json               # epoch별 train/val 손실
```

v2 image decoder의 ckpt는 LoRA state + z_proj만 저장 (frozen 모듈은 sd_model_id로 재로드).

## Dependency Tracking (image decoder ↔ text decoder)

v2 image decoder의 M3 caption stream이 text decoder의 generate 출력에 의존하므로 **어떤 text decoder ckpt로 학습됐는지** 명시적 추적:

1. **Config 명시**: `image_decoder.text_decoder_run` 필드 ([config_diffusion.yaml](config_diffusion.yaml))
2. **Run config.json 자동 기록**: 학습 시작 시 cfg 전체가 `run_dir/config.json`에 dump되어 사후 추적
3. **Cache 파일명 prefix**: `cache/captions_{text_decoder_run}_*.pt` — mismatch 즉시 발견

**Cross-encoder swap 금지**: 같은 encoder weights로 만든 z embedding ↔ text decoder ckpt만 매칭. 예: Run 1 encoder의 z를 Standard CLIP text decoder에 넣는 등의 mismatch는 임베딩 공간이 달라 의미 없음 — 파일명/config로 사전 차단.

## 핵심 hyperparameters

### 공통 ([config.yaml](config.yaml))
| 키 | 의미 | 기본 |
|---|---|---:|
| `text_decoder.z_source` | `centroid` / `modality` / `random_img_txt` | `random_img_txt` |
| `text_decoder.prefix_len` | K (mapper 출력 토큰 수) | `10` |
| `text_decoder.mapper_type` | `transformer` (frozen LM 권장) / `mlp` | `transformer` |
| `text_decoder.beam_size` | 추론 beam | `5` |

### v1 ConvT ([config_convt.yaml](config_convt.yaml))
| 키 | 의미 | 기본 |
|---|---|---:|
| `image_decoder.z_source` | `centroid` / `modality` / `random_img_txt` | `modality` |
| `image_decoder.batch_size` | data loader bottleneck 고려 | `256` |
| `image_decoder.lpips_weight` | `0.0`이면 LPIPS 모듈 미로드 | `5.0` |
| `image_decoder.lpips_net` | `alex` / `vgg` / `squeeze` | `alex` |

### v2 SD-LoRA ([config_diffusion.yaml](config_diffusion.yaml))
| 키 | 의미 | 기본 |
|---|---|---:|
| `image_decoder.z_source` | proposed: `random_img_txt` | `random_img_txt` |
| `image_decoder.batch_size` | SD UNet 메모리 — 16~32 권장 | `16` |
| `image_decoder.epochs` | LoRA 수렴 50k step 정도 | `10` |
| `image_decoder.text_decoder_run` | M3 caption stream의 text decoder run name | `""` (M3 off) |
| `image_decoder.caption_curriculum_epochs` | 처음 N epoch GT, 이후 cached ĉ | `5` |
| `image_decoder.lora_rank` | UNet cross-attn LoRA rank | `8` |
| `image_decoder.n_z_tokens` | z → K virtual tokens | `10` |
| `image_decoder.cond_drop_prob` | classifier-free guidance 학습 drop | `0.1` |
| `image_decoder.sample_steps` | val sample DDIM step | `30` |
| `image_decoder.cfg_scale_sample` | inference CFG scale (1.0=guidance 없음) | `1.0` |

## 전체 학습 파이프라인 (Stage 1, proposed method)

```bash
# 0. (1회) embeddings cache
python encode_dataset.py

# 1. text decoder (random_img_txt, ~1.5h)
python train_text_decoder.py
# → 새 run dir 이름 메모: runs/txt_random_img_txt_..._<ts>

# 2. caption cache (~25분)
python encode_captions.py --text-decoder-run txt_random_img_txt_..._<ts>

# 3. config_diffusion.yaml의 text_decoder_run 채우기 (또는 CLI로)

# 4. image decoder v2 학습 (4 cells, 각 ~16-20h)
# Cell 1 (proposed): M1 on + M3 on
python train_image_decoder_v2.py --z-source random_img_txt --text-decoder-run txt_random_img_txt_..._<ts>
# Cell 2 (M3 ablation): M1 on, caption stream off
python train_image_decoder_v2.py --z-source random_img_txt --text-decoder-run ""
# Cell 3 (M1 ablation): modality only, M3 on
python train_image_decoder_v2.py --z-source modality --text-decoder-run txt_random_img_txt_..._<ts>
# Cell 4 (bottom baseline): modality + M3 off
python train_image_decoder_v2.py --z-source modality --text-decoder-run ""
```

Stage 2 (Standard CLIP baseline 비교)는 Stage 1의 image decoder가 선명한 cross-modal reconstruction을 보인 후 진행.
