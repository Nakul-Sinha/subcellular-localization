# Approach — Multi-Label Subcellular Localization from Microscopy + Function Text

A detailed, research-backed plan to solve the challenge: predict a protein's set of
subcellular locations (16-class multi-label) by **fusing a 2-channel immunofluorescence
image with a location-stripped UniProt function description**, scored by a
**prior-anchored macro-F1 skill score** under **leave-family-out generalization** to
unseen protein families.

> This document is the engineering map. `requirements.md` lists the concrete tooling,
> models, and compute. Both were produced from a deep multi-source research pass
> (HPA Kaggle 2019/2021 winners, DeepLoc 2.0, multimodal-fusion and long-tailed
> multi-label literature, VLM-QLoRA recipes); citations are inline and collected at the
> end.

---

## 0. TL;DR — the recommended solution in one paragraph

Build **Approach (a): a two-encoder vision + text late-fusion model with a 16-way
sigmoid head, trained end-to-end**. Vision = **ConvNeXt-V2-Tiny/Nano** (ImageNet/FCMAE
pretrained) with its first conv adapted to **2 input channels**, run at native **320×320**.
Text = **PubMedBERT / BiomedBERT-base** (mean-pooled, `max_len=256`), fed as a **weak
functional prior** through **late concatenation + MLP** with **text modality-dropout
(p≈0.3–0.5)** so the model cannot memorize family→location shortcuts (every test family
is unseen). Train on **soft labels** with **BCE + per-class `pos_weight`** as the anchor
loss, A/B-tested against **Asymmetric Loss (ASL)** and **Distribution-Balanced loss**.
Validate with **StratifiedGroupKFold on `family_group_id`** — the *only* honest protocol.
The single highest-leverage, most-overlooked move: **the grader thresholds your sigmoid
output at a fixed 0.5**, so per-class threshold tuning must be **folded back into the
logits as a per-class bias** `b_c = −logit(t_c)` — tuning thresholds without baking them
in does literally nothing. Final inference = **5-fold probability-bagging × 4–8-view D4
TTA → per-class logit bias → 0.5 cut**, comfortably inside an A10G 24 GB / ~30 min budget.
The VLM/QLoRA route (Approach b) is kept as a **secondary, research-only** bet — it carries
documented catastrophic-overfitting risk on ~6 k specialized dark-microscopy images.

---

## 1. Problem facts (verified by EDA on the public data)

| Property | Value | Implication |
|---|---|---|
| Train / test images | **6,304 / 762** | Small dataset → overfitting (not capacity) is the binding constraint. |
| Image | `.npy` uint8 `(2, 320, 320)` — ch0 protein, ch1 nucleus | MT + ER reference channels deliberately absent; nucleus is the only spatial anchor. |
| Image intensity | dark/sparse: protein ch ~68 % nonzero, nucleus ~42 %, per-image max ≈ 120–160 | **Naive ImageNet normalization crushes the signal** → per-channel norm + percentile contrast stretch. |
| Labels | 16 `loc_*` columns, **soft** values in [0,1] (privacy noise) | Train on soft targets directly; do **not** binarize; do **not** add extra label smoothing. |
| Class prevalence | 0.52 (nucleoplasm) → 0.02 (intermediate_filaments) | Severe long-tail; macro-F1 weights all classes equally → **rare classes win the score**. |
| Label cardinality | mean **1.93**/protein; **70 % multi-localize**; 49 % have exactly 2 | True multi-label; per-class sigmoids, never softmax. |
| Family groups | **1,186** families; sizes 2–374 (median 3); **test families all unseen** | Leave-family-out CV is mandatory; in-distribution within-family label agreement ≈ **0.877** → text/family is a powerful *in-distribution* shortcut that must be regularized away. |
| Text length shift | train median 80 words / test median **129** (mean 107 → 166) | Test text is richer (enriched for multi-localizers); choose `max_len` to cover the tail (~256 tokens). |
| Empty-ish text | **231 train rows < 40 chars** | Model must degrade gracefully when text is near-empty → motivates modality dropout. |

**Metric anchor (computed locally):** the most-frequent-set baseline predicts *only*
nucleoplasm (the single class with prevalence ≥ 0.5). Its macro-F1 over present classes is
**`F1_base ≈ 0.043`** (nucleoplasm F1 ≈ 0.69, all other 15 classes F1 = 0). Because
`LocSkill = clip((RawMacroF1 − F1_base)/(1 − F1_base), 0, 1)`, the anchor is tiny — so
**every bit of genuine per-class F1 on rare/secondary classes moves LocSkill a lot**, and a
raw macro-F1 of ~0.45–0.58 maps to LocSkill ≈ **0.42–0.56**.

---

## 2. Why this is hard, and where the score is actually won

Three deliberate difficulties, each with a design response:

1. **Unseen protein families at test.** You cannot memorize "family X → location Y."
   The model must learn a *transferable* mapping from (visual pattern + functional prior)
   to compartments. **Response:** leave-family-out CV for *every* decision; regularize the
   text branch hard (it is the family-memorization trap); prefer CNN inductive bias over
   data-hungry ViT/Swin (which collapse on this scale — see §3.1).

2. **Enriched for hard cases** (multi-localizing + visually confusable). Secondary
   compartments are faint in a 2-channel image. **Response:** the functional text earns
   its keep precisely here; use augmentation/consistency that forces attention beyond the
   single brightest punctum; protect rare-class recall via loss + thresholding.

3. **Intrinsically variable labels** (reliability tiers; soft targets). There is an
   irreducible error floor. **Response:** soft-target training as built-in label smoothing;
   EMA/SWA for flatter, better-calibrated minima.

**Where the score is won:** macro-F1 averages per-class F1 with equal weight, so the
difference between a mediocre and a strong solution is **rare/secondary-class recall** —
the classes with prevalence 0.02–0.10. The whole pipeline is engineered around lifting
those classes above the 0.5 grader cut without flooding them with false positives.

---

## 3. Recommended architecture (Approach a — two-encoder fusion)

```
        ┌───────────────────────────┐
ch0,ch1 │  Vision encoder           │  pooled img emb (e.g. 768-d)
(2×320) │  ConvNeXt-V2-Tiny, 2-ch   ├──────────────┐
        │  stem, 320px              │              │
        └───────────────────────────┘              │   concat
        ┌───────────────────────────┐              ▼
function│  Text encoder             │  mean-pool ┌──────────────┐   16 sigmoids
 text   │  PubMedBERT-base          ├───────────►│ 2-layer MLP  ├──► p(loc_c)
 (≤256) │  (frozen→low-LR)          │  + MODALITY│ dropout 0.3  │
        └───────────────────────────┘   DROPOUT  └──────────────┘
                                              ▲          │
                          aux image-only head │          │ aux text-only head
                          (0.3·BCE)  ──────────┘          └─────► (0.3·BCE)
```

### 3.1 Vision encoder
- **Primary: ConvNeXt-V2-Tiny** (`timm: convnextv2_tiny.fcmae_ft_in22k_in1k`) or
  **-Nano** for speed, at native **320×320**. Large kernels suit punctate organelle
  textures; CNN inductive bias generalizes on small data.
- **Why not ViT/Swin:** a recent leave-family-out HPA study (arXiv 2505.22926) shows
  **Swin-B and ViT-B/16 catastrophically overfit** (val 0.42–0.65 → test 0.018–0.18),
  while CNNs hold val≈test far better. Avoid as the primary on ~6 k images + unseen
  families. Likewise avoid DenseNet-169+ / 1024 px (overfit + blow the time budget).
- **2-channel stem surgery (do NOT random-init conv1):** load with `in_chans=2` —
  `timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, in_chans=2,
  num_classes=16)`. timm repeats the 3 RGB filters, slices to 2, and rescales by `3/2`
  to preserve activation magnitude. Map **ch0 (protein)→R weights, ch1 (nucleus)→G
  weights**. Random init throws away the pretrained edge/texture filters that matter most
  on sparse inputs.
- **Ensemble-diversity member:** EfficientNetV2-S with the same stem surgery.
- **Optional ablation (only keep if it beats ConvNeXt on OOF):** initialize from a
  **public microscopy SSL encoder** — **SubCell** (ViT-B/16, HPA-pretrained,
  `github.com/CellProfiling/subcell-analysis`) or **Cell-DINO** (DINOv2 on HPA), used in
  their documented **dropped-reference-marker** mode since ER+MT channels are absent. Start
  **frozen-backbone + trainable head** (cheap, robust under shift). Caveat: their headline
  numbers assume 4 channels — re-validate on OOF before trusting.

### 3.2 Text encoder
- **Primary: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`** (PubMedBERT,
  ~110 M). From-scratch biomedical vocab tokenizes UniProt jargon ("oxidoreductase",
  "GTPase", "glycosyltransferase") into fewer subwords than general BERT/DeBERTa; tops/near-
  tops BLURB. **Drop-in alternative: BioLinkBERT-base** (best public base-size biomedical
  encoder on BLURB, 83.39) — pick by OOF.
- **Pooling: masked mean** over last hidden states (not CLS — CLS is under-trained for
  classification features; production biomedical embedding models default to mean pooling).
- **`max_len = 256`** (covers the ~166-word test tail at ~1.3–1.5 biomedical tokens/word).
  512 doubles cost for no gain; 128 truncates the family/process tail.
- **Treat text as a weak prior:** freeze for the first 1–2 epochs, then fine-tune at a
  **lower LR (≈1e-5 vs 2e-4 image)**. GO molecular-function→localization signal is real but
  weak (max ~0.047 bits/term; biological-process terms span compartments), and the family
  text is a leave-family-out trap.
- **Reject for the final run:** BioLinkBERT-**large** (340 M strains 24 GB/30 min),
  DeBERTa-v3 (needs biomedical DAPT), generic sentence-transformers (MiniLM/BGE/E5
  underperform in-domain). Reserve large encoders for H100 research only.

### 3.3 Fusion
- **Late fusion**: `concat(img_emb, text_emb) → 2-layer MLP (dropout 0.3–0.5) → 16
  sigmoids`. When one modality is a weak/noisy prior, the fusion survey literature
  (arXiv 2404.18947) finds simple **weighted/gated late fusion ≥ heavy cross-attention**;
  cross-attention earns its cost only on token-grounding tasks and *overfits* family-
  specific correspondences here. **Do not** use bilinear/tensor fusion (parameter blow-up).
- **Text modality-dropout p = 0.3–0.5**: zero the entire text vector for a fraction of
  batches → forces the image branch to carry the load, prevents collapse onto the text
  shortcut, and makes the model robust to the 231 near-empty-text rows.
- **Auxiliary unimodal heads kept throughout:**
  `total = L(fused) + 0.3·L(image_only) + 0.3·L(text_only)`. Keeps both branches supervised
  (anti-collapse), doubles as a collapse detector, and gives two extra calibrated members
  for a late-decision average.
- **Optional ablations only:** FiLM conditioning (text → per-channel γ/β on a mid/late
  vision block); a **low-weight (≤0.2) BiomedCLIP-style InfoNCE** image↔text auxiliary loss;
  **OGM-GE** on-the-fly gradient modulation if the image branch is observed starving text
  on OOF (gave +10 pts over concat on CREMA-D in CVPR'22, drop-in).

### 3.4 Preprocessing (high-leverage — images are dark)
- **Per-channel normalization** using dataset mean/std (compute once on train), **plus
  per-image 1–99.5 percentile contrast stretch** (or `log1p`) before normalizing. This is
  one of the cheapest, highest-ROI steps: naive ImageNet stats map mean-pixel-3–16 images
  to near-black and waste the pretrained stem.

---

## 4. Loss & class-imbalance handling

Resolve the **(real, evidence-level) loss conflict** by OOF delta, not by guessing:

- **Anchor — BCEWithLogitsLoss + per-class `pos_weight`** = `clip(neg/pos, max≈10)`, on
  **soft targets**. This is the one choice with *leave-family-out-specific* evidence:
  arXiv 2505.22926 found **BCE > Focal > ArcFace** on held-out HPA test (Focal/ArcFace
  overfit). Start here.
- **Primary contender — Asymmetric Loss (ASL,** Ben-Baruch/Ridnik ICCV'21**)**:
  `gamma_neg=4, gamma_pos=0–1, clip=0.05`. Targets the ~1.93-positive / ~14-negative
  imbalance per sample; the asymmetric `clip` discards likely-mislabeled easy negatives,
  giving free **soft-label-noise robustness**. Drop-in on the same sigmoid head.
- **Tail booster — Distribution-Balanced (DB) Loss** (Wu ECCV'20): class-balanced
  reweighting + negative-tolerant regularization; beat Focal by ~7.6 macro-F1 on
  long-tailed text benchmarks, with the gain concentrated on **tail classes** (where
  LocSkill is won). Caveat: that evidence is from text, not microscopy — verify on OOF.
- **Fallbacks/ablations:** Focal (γ=2), Class-Balanced (Cui'19). A soft-F1 / Lovász
  surrogate term (used by the HPA 2019 winner) is worth a small-weight ablation.

**Soft-label handling:** feed the `[0,1]` targets straight into BCE/ASL — they are proper
for soft targets and act as built-in label smoothing. **Do not** binarize (discards
calibration) and **do not** stack extra label smoothing (double-softening sinks rare
classes below 0.5).

---

## 5. The threshold / calibration lever (the most-overlooked, highest-ROI step)

The grader applies a **fixed 0.5** threshold to *your* probabilities. F1-optimal-threshold
theory (Lipton/Elkan, arXiv 1402.1892) says the optimal threshold is **below 0.5 for
low-prevalence classes**. Therefore:

1. **Tune per-class thresholds `t_c`** on concatenated **5-fold OOF** probabilities to
   maximize each class's F1.
2. **Fold them back into the logits** as a per-class bias **`b_c = −logit(t_c) = −ln(t_c /
   (1 − t_c))`** added before the sigmoid. This is a monotone reparameterization that moves
   the F1-optimal decision boundary to exactly 0.5, so the grader's fixed cut captures the
   gain. *(Tuning thresholds without baking them in does nothing.)*
3. For the **rarest classes** (intermediate_filaments ≈ 0.02), argmax-F1 is unstable
   (few OOF positives × unseen families) → use **prevalence-matching**: choose `b_c` so the
   positive-prediction rate ≈ train prevalence. **Floor** thresholds to avoid degenerate
   all-positive predictions.
4. Optionally **per-class temperature/Platt calibration** fit on left-out families to rein
   in rare-class overconfidence.

> Compliance note: this is a per-class monotone transform of the trained model's own
> output (a calibration step), not a lookup table or hardcoded prediction — it stays inside
> the learned-model rule. Bake the chosen `b_c` into `solution.py`/`solution.ipynb` so the
> official run reproduces the submitted CSV exactly.

---

## 6. Validation protocol (build this FIRST — every later number depends on it)

- **StratifiedGroupKFold on `family_group_id`** (1,186 families), **5 folds**, with
  multi-label **iterative stratification within groups** so the 0.02-prevalence classes
  survive in every fold. A family must **never** span train/val.
- **Random KFold here over-reports catastrophically** (arXiv 2505.22926: 0.638 val → 0.457
  public is the generalization gap signature). The OOF GroupKFold macro-F1/LocSkill is your
  **single source of truth** — it replicates the unseen-family test protocol exactly.
- **Select everything on OOF macro-F1** (loss choice, thresholds, early-stop, checkpoints) —
  **not** val loss, which diverges from macro-F1 under imbalance.
- Run two mandatory **ablation checks** to prove text is a *prior, not a shortcut*:
  - *Image-ablation:* does removing text drop OOF macro-F1? (text should help.)
  - *Text-ablation:* does a text-only model collapse on unseen families? (it should.)
- **Verified caution (adversarial pass, 3-0):** fusion is **not guaranteed to help
  localization**. In the tpLM benchmark, naive embedding fusion improved 4/6 tasks but caused
  a (non-significant) **decrease on the subcellular-Location benchmark**. So **fusion is a
  hypothesis to test on OOF, not an assumption** — if the fused model does not beat the best
  image-only model on OOF macro-F1, **ship image-only + the strong threshold lever** (§5)
  rather than forcing a text branch that overfits families.

---

## 7. Training recipe

| Component | Setting |
|---|---|
| Optimizer | AdamW; image LR **2e-4**, text LR **1e-5**; weight decay 1e-4–5e-2 |
| Schedule | Cosine decay + 3 % warmup; **8–15 epochs**; early-stop on **OOF macro-F1** |
| Regularization | drop-path 0.1–0.2; MLP/head dropout 0.3–0.5; **EMA (decay 0.999)** → predict with EMA weights; optional SWA over last ~20 % epochs |
| Augmentation | **D4 dihedral** (flips + 90° rotations — microscopy is rotation-invariant); random scale/shift; brightness/contrast jitter sized for the dark mean; per-channel intensity scaling; occasional nucleus-channel dropout |
| Mixup/CutMix (multi-label-correct) | **union-of-labels** for CutMix; **ratio-mixed soft targets** (Mixup-BCE) for Mixup; mild α 0.2–0.4. *Never average one-hot targets.* |
| Batch / resolution | ~48–64 @ 320 px on A10G (fits 24 GB) |
| Seeds / determinism | fixed seeds; `cudnn.deterministic` where practical |

---

## 8. Ensembling & test-time augmentation

- **Several smaller diverse models beat one big model — the single most consistent winning
  finding across HPA 2019 & 2021** (adversarially verified, 3-0; every top team, 4–18 models).
  Diversify by backbone (ConvNeXt-V2 + EfficientNetV2-S), loss (BCE+pos_weight vs ASL), and
  augmentation/seed. This is *also* the best use of a constrained budget (many small > one
  large).
- **5-fold probability-bagging:** train one model per fold (each on a different family
  subset → itself a leave-family-out ensemble), **average sigmoid probabilities** (never
  labels) across folds. Add a 2nd seed/fold (→10 models) if the budget allows.
- **D4 TTA:** average sigmoids over 4–8 flip/rotation views.
- **Order at inference:** `5-fold bag × TTA → per-class logit bias b_c → 0.5 cut`.
- **Backbone diversity** (optional): ConvNeXt-V2-Tiny + EfficientNetV2-S + (frozen
  SubCell/Cell-DINO) averaged.
- **Budget guard:** measure on the actual A10G; keep backbones small (Nano/Tiny @320) and
  **cap TTA at 4** if `762 × ensemble × TTA` approaches 30 min.

---

## 9. Generalization toolkit (closing the train→unseen-family gap)

The binding constraint is overfitting to *training families*, not capacity. Stack these,
each validated by **per-fold OOF delta on the rare-class mean**:

1. **Text modality-dropout + low text LR + delayed unfreeze** — breaks the family shortcut.
2. **Smaller backbones** (Tiny/Nano > Small/Base) + weight decay + drop-path.
3. **Heavy label-preserving augmentation** (D4, scale/shift) — cheapest gap-closer.
4. **EMA / SWA** — flatter minima, better calibration, label-noise robustness (≈ free).
5. **5-fold bagging × TTA** — variance reduction against family-specific overfitting.
6. **Soft-target training** — built-in regularization from the privacy noise.

---

## 10. Approach (b) — VLM + QLoRA (secondary / research-only)

Keep as an **orthogonal ensemble bet**, not the default:

- **Risk:** VLMs fine-tuned on small specialized medical/scientific image sets show
  **catastrophic overfitting vs CNN baselines** (medRxiv 2025 chest-X-ray negative result);
  every HPA Kaggle top team used specialized CNN/ViT backbones, none used generative VLMs.
- **If pursued:** **Qwen2-VL-2B** (not 7B — 7B QLoRA needs ~1.5 h for *1 k* samples on a
  24 GB GPU and blows the 30-min final budget). Render 2-ch → 3-ch RGB
  (R=protein, G=protein-enhanced, B=nucleus) **with percentile contrast stretch**; keep
  `min/max_pixels` modest (resolution barely affects HPA score).
- **Critical:** do **not** rely on generative decoding — attach a **16-way sigmoid head on
  the pooled last-hidden-state** and train BCE/focal with LoRA (calibrated, 0.5-aligned,
  lets you protect rare classes). If staying generative, read per-class probabilities via
  **token-level marginalization** over label tokens, never answer-logprob averaging.
- **Where to run it:** H100 sweeps only; the final A10G submission should be the
  two-encoder model unless the VLM demonstrably wins OOF.

---

## 11. Execution plan / milestone progression

| Milestone | Goal | Exit signal |
|---|---|---|
| **M0 — Harness** | StratifiedGroupKFold-on-family OOF + strict submission validator + LocSkill scorer. Baseline: ConvNeXt-V2-Nano @320, 2-ch stem, BCE+pos_weight, **image-only**. | Honest OOF LocSkill measured; valid `submission.csv`. |
| **M1 — Threshold lever** | Per-class threshold → logit-bias fold-back vs global 0.5. | Largest single OOF jump; nearly free. |
| **M2 — Loss A/B/C** | BCE+pos_weight vs ASL(γ_neg=4) vs DB-Loss, per-fold OOF, rare-class focus. | Pick the OOF winner. |
| **M3 — Add text branch** | PubMedBERT + late fusion + modality-dropout; run image/text ablations. | Keep only if OOF↑ **and** text-only collapses on unseen families. |
| **M4 — Ensemble + TTA** | 5-fold bag × 4-view D4 TTA, probability averaging. | Variance reduction confirmed on OOF. |
| **M5 — Polish** | per-channel norm + percentile stretch; EMA; drop-path; multi-label Mixup/CutMix. | Each kept only on OOF delta. |
| **M6 — Diversity** | EfficientNetV2-S member; SubCell/Cell-DINO frozen init (drop-marker mode). | Add if OOF↑ within budget. |
| **M7 — (opt) VLM** | Qwen2-VL-2B + sigmoid head as an orthogonal member, H100 sweep. | Only if everything above plateaus. |

---

## 12. Ablation priorities (highest expected value first)

1. **OOF harness** (StratifiedGroupKFold-on-family) — nothing is meaningful without it.
2. **Per-class threshold folded as logit bias** vs global 0.5 — biggest, nearly-free jump.
3. **Loss A/B/C** (BCE+pos_weight / ASL / DB) on per-fold OOF.
4. **Text branch + modality dropout** + ablation checks.
5. **5-fold bagging + 4-view D4 TTA** (probability averaging).
6. **Preprocessing** (per-channel norm + percentile stretch / log1p).
7. **EMA + drop-path + multi-label Mixup/CutMix** stack.
8. **Backbone diversity** (EfficientNetV2-S; SubCell/Cell-DINO frozen).
9. Ablate-only: FiLM, low-weight InfoNCE, OGM-GE, resolution 320→384.
10. Only on plateau + spare H100: Qwen2-VL-2B + sigmoid head.

---

## 13. Expected scores & risk register

- **Targets:** raw macro-F1 ≈ **0.45–0.58** → **LocSkill ≈ 0.42–0.56** (description's stated
  band; constant baseline ≈ 0; `F1_base ≈ 0.043`). Rare/secondary classes are the ceiling —
  HPA winners plateaued at AP ≈ 0.31–0.34 on weak classes even with 4 channels + external
  data.
- **Key risks:**
  - *Loss conflict unresolved by external evidence* — BCE-beats-Focal data is HPA-specific;
    ASL/DB data is text-benchmark. **Must A/B on OOF.**
  - *Text branch family-memorization* — mitigated by modality dropout + low LR + ablation
    checks; if it can't be shown to transfer, ship image-only + the strong threshold lever.
  - *Rare-class threshold instability* — prevalence-matching + flooring.
  - *Microscopy-SSL channel mismatch* — SubCell/Cell-DINO assume 4 channels; re-validate
    drop-marker mode.
  - *No external HPAv18 data* (banned/unavailable) — drove the biggest rare-class gain for
    the 2019 winners; here all rare-class signal must come from loss + augmentation +
    thresholds.
  - *Runtime* — measure 5–10-model bag × TTA on real A10G; cap TTA/ensemble to fit 30 min.

---

## 14. Compliance (must-hold constraints)

- **Learned deep model only.** Every `loc_*` value comes from the trained two-encoder
  fusion (or VLM) over the raw image + raw text. **No** GBDT/RF/SVM/kNN/logistic regression
  on extracted features.
- **No hardcoding / no shortcuts.** No lookup tables, regex, `if/elif` chains, or templates
  that set any `loc_*` from `id`, from string matches in `function_text`, or from metadata.
  No dataset fingerprinting, file-order, or hidden-ID leakage. Regex only for non-predictive
  text cleanup if ever needed.
- **Both modalities used by design** — verify fusion helps via image-only/text-only
  ablations.
- **Reproducible in the expected runtime** (A10G-class, 24 GB, ~30 min). H100 is for
  research/sweeps only; the official notebook must not depend on H100-only artifacts.
- **Official script** reads from the public dataset and writes `working/submission.csv`,
  regenerating predictions through the modeling pipeline in isolation (no reading/mirroring
  an uploaded `submission.csv`). The folded-in per-class bias and any TTA/bagging must be
  reproduced in the official code so it matches the submitted CSV.
- **No external/private models or APIs** — only public, reproducible pretrained weights
  (ImageNet ConvNeXt-V2, PubMedBERT, optionally public SubCell/Cell-DINO).

---

## 15. Key references (consulted)

**HPA competitions & analyses**
- Nature Methods, *Analysis of the HPA Image Classification competition (2019)* — PMC6976526
  (DenseNet121 best, BCE/focal/Lovász, AutoAugment 0.477→0.499, TTA/ensemble gains, external
  HPAv18 0.510→0.552).
- Nature Methods, *HPA single-cell weakly-supervised competition (2021)* — PMC9550622
  (focal loss dominance, resolution-has-little-impact, rare-class plateaus, ensembling).
- pudae 3rd-place repo — `github.com/pudae/kaggle-hpa` (focal γ=2, Adam 5e-4, TTA 4/8).
- pfnet-research 7th-place 2021 — `github.com/pfnet-research/kaggle-hpa-2021-7th-place-solution`.
- HPA leave-family-out study — arXiv 2505.22926 (BCE > Focal > ArcFace held-out; ViT/Swin
  collapse; ResNet generalizes).

**Encoders & SSL**
- timm first-conv adaptation — `timm.fast.ai/models` (`in_chans` repeat/slice/rescale).
- SubCell microscopy foundation model — PMC12636579 / `github.com/CellProfiling/subcell-analysis`.
- Cell-DINO (DINOv2 on HPA) — PLOS Comp Bio `pcbi.1013828`.
- PubMedBERT — arXiv 2007.15779; BLURB leaderboard `microsoft.github.io/BLURB`.
- BioLinkBERT — arXiv 2203.15827.
- DeepLoc 2.0 (per-label weighted focal + MCC-optimized thresholds) — PMC9252801.
- GO molecular-function → localization signal (weak, ~0.047 bits) — PMC2652875.

**Fusion, loss, calibration, VLM**
- Multimodal fusion on low-quality data survey — arXiv 2404.18947.
- OGM-GE gradient modulation — CVPR'22, `github.com/GeWu-Lab/OGM-GE_CVPR2022`.
- CMA-CLIP modality-wise attention — arXiv 2112.03562.
- Asymmetric Loss (ASL) — arXiv 2009.14119, `github.com/Alibaba-MIIL/ASL`.
- Distribution-Balanced Loss — arXiv 2007.09654.
- Thresholding classifiers to maximize F1 — arXiv 1402.1892.
- Qwen2-VL QLoRA recipe — philschmid.de / HF TRL cookbook.
- VLMs don't transfer to medical imaging (overfitting) — medRxiv 2025.12.06.25341759.
- Token-marginalization for multi-label LLM classifiers — arXiv 2511.22312.

---

## 16. Appendix — adversarially-verified claim list (deep-research pass)

A second, independent deep-research pass (6 angles → 27 sources → 127 claims → top 25
adversarially verified with a 3-vote refute test; 17 confirmed, 8 killed) was run to
fact-check the plan. It **corroborates** the recommendations above and sharpens two points.
Votes are *confirm-refute*.

**Confirmed (high confidence):**
- **Ensemble several smaller diverse models (4–18), trained with different losses &
  augmentations — beats a single large model** *(3-0)*. The single most consistent finding
  across every top HPA 2019 & 2021 team. → §8.
- **BCE base loss + focal/class-weights + rare-class oversampling + (optionally) a Lovász
  term, with per-class threshold tuning** is the proven imbalance recipe *(3-0)*. → §4–5.
  *(Note: a concrete validated split was focal-α=1 for image-level EfficientNets + BCE for
  cell-level ResNeSt50, pfnet 7th.)*
- **Medium CNN/ViT backbones are competitive; DenseNet-121 > DenseNet-169; DenseNet >
  ResNet** for top 2019 teams *(3-0)*. Supports "smaller backbone, not larger" under
  unseen-family shift. → §3.1.
- **Self-supervised DINOv2-style pretraining (Cell-DINO) is strong in the label-scarce
  regime** *(3-0)* — *caveat:* trained on 4 channels / single-cell; headline gains are
  relative to weak baselines, so re-validate at 2 channels. → §3.1 optional ablation.
- **PubMedBERT is the standard biomedical text encoder for protein-function text** (adopted
  by ProtST); **BiomedCLIP** is a strong CLIP-style biomedical alternative *(3-0)*. → §3.2.
- **ProtST fusion = CLIP-style InfoNCE alignment *plus* a cross-attention fusion module**,
  not contrastive-alone or concat-alone *(3-0)*. Blueprint for the optional InfoNCE/FiLM
  ablations. → §3.3.
- **Text/embedding-fusion choices must be validated per-task; naive "concatenate everything"
  is not universally good** *(3-0)*. → drives the §6 fusion-ablation caution.
- **Learned deep encoders > classical hand-crafted-feature baselines** *(2-1, medium;
  yeast)* — reinforces the learned-model-only constraint. → §14.

**Killed / unresolved (do NOT over-claim these):**
- *"Channel mapping beats channel replication for missing channels"* — **refuted (0-3).**
  Missing-channel handling is **unresolved** → keep both real channels, don't fabricate
  MT/ER, and treat any derived 3rd channel as an ablation (§3.1), not a known win.
- *"A specific DINO ViT-B/8 achieves macro-F1 0.8221 / best backbone is settled"* — **refuted
  (1-2).** Best backbone on 2-channel under leave-family-out is **unresolved** → ablation-
  driven choice with ConvNeXt-V2 as a reasoned default.
- *"ProtST's primary fusion is contrastive-alone"* — **refuted (0-3)** (it adds cross-
  attention; see above).

**Open questions the evidence did not settle (resolve empirically on OOF):**
1. Best vision backbone on **2-channel** HPA under leave-family-out (EfficientNet vs DenseNet
   vs ConvNeXt-V2 vs SSL ViT).
2. Best **missing-channel** strategy (mapping vs replication vs SSL-on-available-channels).
3. Whether **VLM+QLoRA** is competitive within the runtime budget (no surviving evidence in
   the generic pass; the engineering pass judged it a research-only bet — §10).
4. Whether fusing the location-stripped text actually beats the **best image-only** model for
   macro-F1, and which single text encoder anchors the fusion after per-task ablation.

**Dominant caveat:** nearly all primary HPA sources concern **4-channel, single-cell, 19/28-
class** setups — not this **2-channel, 16-class, location-stripped-text, prior-anchored
macro-F1, leave-family-out** task. The image+text fusion evidence is largely transferred from
the protein-**sequence**+text literature (ProtST, tpLM benchmark, BiomedCLIP). Treat every
external recipe as a **blueprint to validate on the OOF harness**, never as a settled result.

*Primary sources added by this pass:* HPA 2019/2021 Nature Methods analyses
(s41592-019-0658-6, s41592-022-01606-z, PMC9550622); Cell-DINO (PLOS pcbi.1013828); ProtST
(arXiv 2301.12040); BiomedCLIP (arXiv 2303.00915); tpLM embedding-fusion benchmark
(bioRxiv 2024.08.24.609531); ASL (arXiv 2009.14119); F1-threshold theory
(csie.ntu.edu.tw/~cjlin/papers/threshold.pdf, PMC4442797); pfnet 7th-place repo.
