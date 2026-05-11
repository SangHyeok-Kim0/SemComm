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

송신측은 단일 z (image / text / centroid 중 하나)를 보내고, 수신측은 modality별 디코더 스왑으로 원하는 modality를 복원. 본 저장소는 **frozen encoder + per-modality decoder 학습 + cross-modal decoding 평가**를 단계적으로 진행한다.

## Phase A — Decoder-swap 가능성 진단 (학습 없음)

`final_full.pt`의 image/text 임베딩(4992 × 1024)으로:

1. **Modality classifier sanity** — linear LR / RBF-SVM / 5-NN. 0.5에 가까울수록 modality-agnostic.
2. **Pair geometry** — pair-wise Euclidean distance & cosine similarity의 평균/분산/분포.
3. **AWGN σ 스윕** — 임베딩에 가우시안 노이즈 주입 후 re-normalize → KNN / V-Measure / CosTP / gap 변화.

```bash
python Code/SemComm/phase_a_diagnose.py
```

산출: `results/phase_a/` (json + csv + png)

## Phase B — Decoder 학습 (현재)

### Step 1 — 임베딩 캐시

frozen Run 1 encoder로 COCO train2017 + val2017의 image/caption 임베딩을 1회 계산하여 `cache/`에 저장.

```bash
# 전체 (train ~118k images / ~590k captions, val ~5k / ~25k)
python Code/SemComm/encode_dataset.py

# 샘플 (smoke test)
python Code/SemComm/encode_dataset.py --max-images 256 --splits train val
```

산출:
- `cache/z_img_{train,val}.pt`  — (N_img, 1024) fp16, unit-norm
- `cache/z_txt_{train,val}.pt`  — (N_cap, 1024) fp16, unit-norm
- `cache/index.json`             — image_ids, caption_image_idx, caption_texts

### Step 2 — Image Decoder (ConvTranspose, [-1, 1] 픽셀 공간)

```bash
# 풀 학습 (config.yaml 기준)
python Code/SemComm/train_image_decoder.py

# 옵션
python Code/SemComm/train_image_decoder.py \
    --max-train-images 1000 \
    --epochs 3 \
    --z-source modality
```

Loss: `mse_w·MSE + l1_w·L1 + lpips_w·LPIPS` (config로 가중치 조절, `lpips_w=0`이면 LPIPS 모듈 미로드).

### Step 3 — Text Decoder (Qwen3-0.6B-Base + Transformer mapper)

```bash
python Code/SemComm/train_text_decoder.py

python Code/SemComm/train_text_decoder.py \
    --max-train-captions 5000 \
    --epochs 2 \
    --batch-size 32 \
    --z-source centroid
```

LM은 frozen, mapper(111M)만 학습. 최초 실행 시 ~1.2GB 가중치 자동 다운로드.

> EOS-aware 변종: `train_text_decoder_eos.py`. caption 끝에 EOS를 붙여 학습해 자연 종료를 학습. 기존 `train_text_decoder.py`는 EOS 미부착 — `_  _  _` / `<br />` 같은 LM pretraining junk가 생성 끝에 따라옴 (post-process로 우회).

### Step 4 — Caption 시각화 (이미지 | GT 5개 | GEN 1개)

```bash
python Code/SemComm/visualize_captions.py --run-dir runs/txt_modality_K10_ep10_<ts> --n 10
```

기본은 5종류 (`visualize_image`, `visualize_{text,centroid}_{mean,random}`)가 한 번에 생성. 단일 조합만 만들려면 `--z-source / --cap-agg`로 명시.

옵션:
- `--z-source all` (기본, 5종 한 번에) / `image` (cross-modal swap, z_img) / `text` (z_txt) / `centroid`
- `--cap-agg mean` (기본) / `random` — text/centroid에서 이미지당 5 caption z를 어떻게 합칠지. mean = 5 z_txt 평균 후 unit-norm (image의 text-side prototype, GT 5개 모두 표시). random = `--seed`로 결정된 1개만 선택 (선택된 GT 1개만 표시). **image 모드에는 영향 없음**.
- `--seed 42` (기본). 같은 seed면 매번 같은 이미지가 뽑힘 — 다른 샘플 보려면 값 변경. all 모드도 5개 PNG 모두 같은 이미지셋을 공유.
- `--ckpt epoch_005.pt` (기본 `final.pt`)
- `--no-clean` — 첫 문장 cut 끄기 (`models.TextDecoder.generate`의 `_clean_caption` 우회 → raw beam output)
- `--n 10` 샘플 수, `--output <path>` 저장 위치 (단일 조합일 때만 적용)

산출: `<run-dir>/samples/visualize_<suffix>.png`. suffix는 image 모드만 `image`, text/centroid는 `<z-source>_<cap-agg>` (e.g. `text_mean`, `centroid_random`).

`all` 모드(기본)에선 추가로 `visualize_combined.png` 1장 생성 — row마다 image + GT 5개 + GEN 5개를 한 panel에 모아 직접 비교 가능. random에서 z 만들 때 뽑힌 GT caption은 **빨간 글씨**로 표시.

### Step 5 — Caption 정량 평가 (BLEU / ROUGE-L / CIDEr)

```bash
pip install pycocoevalcap                                                      # 1회 (Java 미사용 경로)

# Full val 5000 이미지 × 5 z 시나리오 (~16분, GB10)
python Code/SemComm/eval_captions.py --run-dir runs/txt_modality_K10_ep10_<ts>

# Sanity (--n 32, ~10초)
python Code/SemComm/eval_captions.py --run-dir <run> --n 32 --batch-size 16
```

5 시나리오 (`image`=cross-modal swap, `text_{mean,random}`=in-modal, `centroid_{mean,random}`=평균) 각각 generate → BLEU-1~4 / ROUGE-L / CIDEr 산출. METEOR/SPICE는 Java 의존이라 skip (논문 비교는 BLEU-4 + CIDEr 가장 많이 인용).

옵션:
- `--n 500` 등으로 subset 평가
- `--scenarios image text_mean` 등으로 일부만
- `--batch-size 64` (메모리 여유 시 늘리기)
- `--no-clean` — `_clean_caption` 우회 비교
- `--noise-snr 20 10 0 -5` — AWGN inject SNR_dB sweep. 단일 또는 여러 값. 미지정 → 무노이즈. SNR마다 별도 JSON `eval_metrics_snr<x>.json`. Phase A 결과 기준 운영점은 +10 dB 이상.

`visualize_captions.py`도 `--noise-snr <단일 값>` 지원 — 파일명 suffix `_snr<x>` 추가, combined PNG 상단에 "AWGN inject @ SNR = x dB" title.

산출:
- `<run-dir>/eval_metrics.json` — 시나리오별 메트릭 + 모든 GT/GEN 페어
- stdout — markdown 표

해석 가이드:
- `text_mean ≥ text_random` (mean이 prototype이라 더 안정)
- `text_mean ≫ image` 일수록 modality gap이 cross-modal에 손실 큼; 비슷할수록 swap이 잘 작동함을 의미 (→ ModalityGap encoder의 효용 정량화)
- `centroid_*`가 `text_*`보다 떨어지면 z_img 노이즈 영향 발생 — gap 분석 단서

### 산출물 (각 학습 run)

```
runs/<run_name>/
├── config.json
├── checkpoints/{epoch_NNN.pt, final.pt}
├── samples/                        # image: PNG (위=GT, 아래=복원), text: GT/GEN 짝
├── wandb/                          # wandb 로컬 캐시 (cfg.wandb.enabled=true 일 때)
└── metrics.json                    # epoch별 train/val 손실
```

WandB 프로젝트는 `cfg.wandb.project` (기본 `SemCom`). 학습 중 step마다 `loss/*`, `optim/*`, epoch마다 `train_epoch/*`, `val/*`, 그리고 디코딩 샘플 이미지/캡션 테이블이 자동 업로드. 끄려면 `cfg.wandb.enabled: false`.

## 핵심 hyperparameters (config.yaml)

| 키 | 의미 | 기본 |
|---|---|---:|
| `z_source` | `centroid` (mu = norm((z_img+z_txt)/2)) / `modality` | `centroid` |
| `image_decoder.lpips_weight` | `0.0` 면 LPIPS 미로드 | `0.5` |
| `image_decoder.lpips_net` | `vgg` / `alex` / `squeeze` | `vgg` |
| `text_decoder.prefix_len` | K (mapper 출력 토큰 수) | `10` |
| `text_decoder.mapper_type` | `transformer` (frozen LM 권장) / `mlp` | `transformer` |
| `text_decoder.mapper_layers` / `mapper_heads` | mapper Transformer 깊이/헤드 | `8` / `8` |
| `text_decoder.beam_size` | 추론 beam | `5` |
| `text_decoder.max_new_tokens` | 추론 길이 한도 | `30` |
| `text_decoder.lm_dtype` | `auto` / `float32` / `bfloat16` | `auto` |
| `text_decoder.batch_size` | LM 메모리 부담으로 별도 | `64` |

전부 config.yaml에서 자유 조정.

## Phase C/D/E (예정)

- **C — Cross-modal swap 평가**: val 4992 페어에 대해 송신 z(img/txt/centroid) × 수신 decoder(img/txt) 모든 조합 — image 메트릭(MSE/LPIPS), text 메트릭(BLEU/CIDEr).
- **D — 채널 모델**: AWGN σ 스윕 (Phase A 운영점 활용).
- **E — Baseline 비교**: standard CLIP 인코더(추가 학습 필요)에서 동일 디코더 학습 → "gap이 크면 swap이 깨진다"를 정량화.
