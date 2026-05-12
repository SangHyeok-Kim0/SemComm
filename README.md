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

**매 epoch 자동 6-inference grid**: 학습 중 매 epoch 끝마다 `samples/epoch_NNN/` 폴더가 생성되고 그 안에 **2 z_source × 3 cfg = 6 PNG**가 저장됨 — `recon_{zimg,ztxt}_cfg{1.0,3.0,7.5}.png`. 학습 throughput ~1.5% overhead (epoch당 ~50초 추가).
- 학습 과정 추적 (epoch별 reconstruction quality 변화)
- **zimg vs ztxt 페어가 cross-modal swap (modality-agnostic) 검증의 핵심** — random은 평균 효과라 평가 가치 적어 제외 (학습 default 동작 reproduction이 필요하면 `infer_image_decoder_v2.py --include-random`)
- wandb dashboard에는 대표 2장(`zimg+cfg3.0`, `ztxt+cfg3.0`) `log_samples_every_n_epochs` 주기로 업로드

### Step 4c — v2 학습 중간 / 종료 후 inference (`infer_image_decoder_v2.py`)

매 epoch 저장된 `epoch_NNN.pt` ckpt로 즉시 sample image 생성. 학습 process와 GPU 공유 가능 (~5GB 추가). 학습 끝 기다리지 않고 진행 상황을 시각적으로 추적하거나, z_source/CFG ablation을 빠르게 시도하는 도구. 모든 명령은 **workspace 디렉터리**(`/workspace`)에서 실행.

```bash
# 추천: 6 combos 한 번에 (zimg/ztxt × cfg 1.0/3.0/7.5 = 6 PNG) — train 매 epoch 자동 출력과 동일 구조
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/imgdiff_random_img_txt_capON_bs16_lr0.0001_r8_20260511-153122 \
    --ckpt epoch_004.pt --all-combos --n 8 --seed 26
# 산출: samples/epoch_003/recon_{zimg,ztxt}_cfg{1.0,3.0,7.5}_seed42.png (6개)

# random z_source 포함 시 9 PNG (학습 default 동작 reproduction용)
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/... --ckpt epoch_003.pt --all-combos --include-random --n 8
```

옵션별 사용 예시 (모두 workspace에서 실행):

```bash
# Single inference — z_source 1개 × cfg 1개
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/imgdiff_..._<ts> --ckpt epoch_005.pt \
    --z-source zimg --cfg-scale 3.0 --n 8

# Random sample selection (재현 가능한 random seed 지정)
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/imgdiff_..._<ts> --ckpt epoch_005.pt \
    --all-combos --n 8 --seed 42

# Caption stream 끄기 (M3 ablation)
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/imgdiff_..._<ts> --ckpt epoch_010.pt --no-caption

# GT caption 사용 (curriculum 단계의 동작 재현)
python Code/SemComm/infer_image_decoder_v2.py \
    --run-dir runs/imgdiff_..._<ts> --ckpt epoch_005.pt --use-gt-caption
```

CLI 옵션 요약:
| 옵션 | 의미 | default |
|---|---|---|
| `--ckpt` | run dir의 checkpoints/ 아래 파일명 | `final.pt` |
| `--n` | 생성할 sample 개수 | 8 |
| `--batch-size` | DDIM 배치 크기 | 8 |
| `--z-source` | `random` (50/50) / `zimg` / `ztxt` / `centroid` | `random` |
| `--steps` | DDIM step 수 | 30 |
| `--cfg-scale` | classifier-free guidance scale | 1.0 |
| `--all-combos` | 6 PNG (zimg/ztxt × cfg) 한 번에. `--include-random`이면 9 PNG | — |
| `--include-random` | `--all-combos`에 random z_source 추가 (학습 default 동작 reproduction) | — |
| `--seed` | val image random 선택 시드 (미지정 시 sequential 첫 n개) | None (sequential) |
| `--no-caption` | caption stream 강제 OFF | — |
| `--use-gt-caption` | cached ĉ 대신 GT caption 사용 | — |
| `--output` | output PNG 경로 override (single inference만) | auto |

Sample selection 동작:
- **`--seed` 미지정**: sorted val image indices의 첫 n개 (예: index 0, 1, 2, ..., n-1). 매번 같은 GT 8장.
- **`--seed N` 지정**: `torch.randperm(N_val_images, seed=N)`으로 shuffle 후 첫 n개. 같은 seed면 재현 가능, 다른 seed면 다른 sample.
- 각 image의 첫 caption(`caption_image_idx[j]==i` 만족하는 첫 j)을 convention으로 사용. z_txt 인덱스 j와 captions_ztxt[j], z_img 인덱스 i와 captions_zimg[i]가 정확히 매칭.

파일명에 `z_source / cap on-off / cfg_scale`가 모두 박혀 있어 여러 조합 돌려도 덮어쓰기 없음. ckpt도 다르면 `samples/epoch_001/`, `samples/epoch_005/` 분리 저장 → epoch별 quality 변화 시각 추적 가능.

추천 사용 시점:
- epoch 1 ckpt 생성 직후 → `--all-combos`로 9 combo quick check
- caption curriculum 끝(epoch 6+) → cached ĉ로 swap된 후 quality 변화 확인
- 학습 종료 후 → `--seed N`로 여러 sample set에서 cross-modal swap 일관성 정량 비교

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
│   ├── epoch_NNN.pt           # save_every_n_epochs 주기로 저장
│   └── final.pt               # 학습 종료 시
├── samples/
│   ├── epoch_001/             # v2: 매 epoch별 폴더, 9 PNG (3 z_source × 3 cfg)
│   │   ├── recon_random_cfg1.0.png
│   │   ├── recon_random_cfg3.0.png
│   │   ├── recon_random_cfg7.5.png
│   │   ├── recon_zimg_cfg{1.0,3.0,7.5}.png
│   │   └── recon_ztxt_cfg{1.0,3.0,7.5}.png
│   ├── epoch_002/ ...
│   └── infer_*.png            # infer_image_decoder_v2.py로 수동 생성한 PNG
├── wandb/                     # wandb 로컬 캐시
└── metrics.json               # epoch별 train/val 손실
```

v1(ConvT)의 `samples/`는 단일 `epoch_NNN.png` (15 sample grid). v2는 위 9-combo 폴더 구조.
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
