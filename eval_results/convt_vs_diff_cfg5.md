## Per z_source metrics

| model | z_src | fid ↓ | lpips ↓ | clip_score ↑ | psnr ↑ | ssim ↑ |
|---|---|---|---|---|---|---|
| ConvT (img_centroid_bs256_ep30_20260506-130321) | zimg | 400.009 | 0.750 | 0.188 | 12.233 | 0.153 |
| ConvT (img_centroid_bs256_ep30_20260506-130321) | ztxt | 401.247 | 0.756 | 0.189 | 11.838 | 0.150 |
| Diffusion cfg5.0 (imgdiff_random_img_txt_capON_bs16_lr0.0001_r8_20260511-153122) | zimg | 58.773 | 0.695 | 0.269 | 9.331 | 0.205 |
| Diffusion cfg5.0 (imgdiff_random_img_txt_capON_bs16_lr0.0001_r8_20260511-153122) | ztxt | 60.977 | 0.725 | 0.291 | 8.937 | 0.189 |

## Δ (ztxt − zimg) — modality-agnostic

| model | Δfid | Δlpips | Δclip_score | Δpsnr | Δssim |
|---|---|---|---|---|---|
| ConvT (img_centroid_bs256_ep30_20260506-130321) | +1.238 | +0.006 | +0.001 | -0.395 | -0.004 |
| Diffusion cfg5.0 (imgdiff_random_img_txt_capON_bs16_lr0.0001_r8_20260511-153122) | +2.204 | +0.030 | +0.022 | -0.394 | -0.016 |

_fid/lpips ↓ better, clip_score/psnr/ssim ↑ better. |Δ| ↓ better (modality-agnostic)._
