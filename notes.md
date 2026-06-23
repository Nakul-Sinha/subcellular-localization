# Notes: Multi-Label Subcellular Localization (Microscopy + Function Text)

Working log of challenge facts, validation, and submissions.

## Challenge facts
- **Task:** 16-class multi-label subcellular localization; independent sigmoid prob per class.
- **Inputs:** 2-channel IF image `.npy` (2,320,320) uint8 [protein, nucleus] + location-stripped UniProt function text.
- **Metric:** prior-anchored macro-F1 skill score. `RawMacroF1` = mean per-class F1 (thr 0.5) over classes with ≥1 gold positive; `LocSkill = clip((RawMacroF1 − F1_base)/(1 − F1_base), 0, 1)`. Direction: maximize.
- **Anchor:** `F1_base ≈ 0.043` (predict-only-nucleoplasm, the only class with prevalence ≥ 0.5). Computed locally on train.
- **Data:** 6,304 train / 762 test. Soft labels in [0,1] (privacy noise). Prevalence 0.52→0.02. Mean 1.93 locations/protein; 70% multi-localize.
- **Generalization:** test families all unseen; `family_group_id` (1,186 families) → leave-family-out CV. Within-family label agreement ≈ 0.877 (strong in-distribution shortcut to avoid).
- **Output:** `submission.csv` = `id` + 16 `loc_<class>` probs, one row per test protein.

## Approach (see approach.md for full detail)
Two-encoder vision+text fusion: ConvNeXt-V2 (2-ch stem) + PubMedBERT (mean-pool, weak prior w/ modality dropout) → late concat MLP → 16 sigmoids + aux unimodal heads. Trained on soft labels with BCE+pos_weight under StratifiedGroupKFold-on-family. Per-class F1-optimal thresholds folded into logits (`b_c=-logit(t_c)`) so the grader's fixed 0.5 is optimal. Inference = fold-bag × D4 TTA.

## Environment (H100 research box)
- 1× H100 NVL 96 GB, 40 cores, 313 GB RAM. torch 2.5.1+cu121, transformers 4.46.3, timm 1.0.27.
- Note: transformers ≥5 refuses to load PubMedBERT `.bin` on torch<2.6 → pinned transformers 4.46.3.
- All scratch on `/mnt` (root disk was ~97% full); HF/torch caches redirected to `/mnt`.

## Validation runs (leave-family-out OOF, StratifiedGroupKFold on family_group_id, H100)
| Run | Config | rawF1 @0.5 | rawF1 +bias | LocSkill | Notes |
|---|---|---|---|---|---|
| 1 | nano, 3 folds, 6 ep, BCE, OneCycle | 0.178 | 0.224 | 0.189 | Undertrained, OneCycle annealed LR too fast over few steps; only common classes learned. |
| 2 | nano, 3 folds, 16 ep, BCE, cosine LR | 0.305 | 0.369 | 0.341 | More epochs + cosine LR holding peak; loss 1.43→0.72. Big jump. |
| **3 (final)** | **tiny, 3 folds, 18 ep, BCE, cosine LR** | **0.384** | **0.457** | **0.433** | Capacity + epochs. **rawF1 0.457 is inside the stated 0.45 to 0.58 band.** Shipped. |

**Validated findings:**
1. **Threshold-as-logit-bias lever works and is high-ROI**: lifted OOF rawF1 by +0.045 / +0.064 / +0.073 across runs 1/2/3 for free. The single most-overlooked step (grader thresholds at fixed 0.5).
2. **Epochs + capacity were the dominant levers** here: 0.189 → 0.341 → 0.433 LocSkill from undertrained-nano → trained-nano → trained-tiny.
3. **Fusion pipeline is sound** end-to-end (2-ch ConvNeXt-V2 + PubMedBERT, modality dropout, EMA, per-channel percentile norm, D4 TTA).
4. Per-fold variance is real (fold-2 OOF 0.45 vs fold-3 0.32), expected under leave-family-out; motivates 5-fold + multi-seed bagging next.

**Shipped submission:** `submission.csv` from Run 3 (tiny/3-fold/18-ep, 3-fold prob-bag × 4-view D4 TTA, per-class logit bias folded in). Reproduced by `solution.ipynb`.

## Path to the upper band (now at raw 0.457 / LocSkill 0.433: low end reached)
Remaining levers, highest-ROI first (each ~+0.02 to 0.05, validate on OOF):
1. **5-fold + multi-seed bagging** (currently 3 folds; fold variance is high), better OOF coverage + variance reduction.
2. **Loss A/B/C**: BCE+pos_weight vs ASL(γ_neg=4) vs Distribution-Balanced, decided per-fold on rare-class mean.
3. **ConvNeXt-V2-Small / longer (24 to 30 ep)** capacity for rare classes.
4. **Heavier aug** (multi-label Mixup/CutMix done correctly) + already-on EMA.
5. **Backbone-diversity ensemble** (tiny + EfficientNetV2-S + frozen SubCell/Cell-DINO).
6. Refine the per-class bias (rare-class prevalence-matching is approximate).

## Compliance
Learned deep model only (no classical ML); no hardcoding/metadata/leaderboard probing; reads `./dataset/public/`, writes `./working/submission.csv`; A10G-reproducible (no H100-only artifacts); per-class logit bias is a monotone calibration of the model's own output, baked into solution.ipynb.

## Submissions
- (to fill after final run): public score vs OOF, changes per submission.
