# Requirements — Multi-Label Subcellular Localization from Microscopy + Function Text

Everything needed to reproduce the plan in `approach.md`: compute, Python environment,
libraries (with versions), pretrained-model artifacts, data layout, and the research-only
(VLM/H100) extras. Split into **(A) required for the official A10G submission** and
**(B) optional research / sweep stack**.

---

## 1. Compute

### 1.1 Official / final runtime (what the solution must fit)
- **GPU:** A10G-class, **24 GB VRAM** (also fine on RTX 3090/4090, L4, T4-16GB with smaller
  batch). Single GPU.
- **System RAM:** **≥ 32 GB** (64 GB comfortable). The full train image set is ~1.3 GB of
  `.npy` (6,304 × ~205 KB) — it fits in RAM; cache decoded tensors to avoid disk I/O.
- **Disk:** ~5 GB (dataset ~1.45 GB + pretrained weights ~1–2 GB + checkpoints).
- **Wall-clock budget:** **~30 min** end-to-end for the official notebook. Keep backbones
  small (ConvNeXt-V2-Nano/Tiny @320), cap the fold-bag × TTA so inference on 762 test images
  stays well inside budget.
- **No network at inference** assumption — **pre-download all pretrained weights** and ship
  them / cache them locally (see §4). The official run must not hit the internet.

### 1.2 Research / sweep compute (optional, not in the official path)
- **H100 NVL** (or A100 80 GB) for: leave-family-out CV sweeps, loss A/B/C, backbone
  comparison, VLM-QLoRA experiments, augmentation/threshold search.
- Azure CLI is available to provision H100 NVL; run long jobs **detached** (nohup/tmux) with
  a PID file + log, poll in short bounded intervals, and **deallocate when done**.
- **Hard rule:** the official `solution.ipynb` must reproduce **without** any H100-only
  saved weights/embeddings/predictions. H100 is for finding the recipe, not for baking
  artifacts into the submission.

---

## 2. Python environment

- **Python 3.10 or 3.11** (match the expected Kaggle/Docker runtime).
- **CUDA 12.1+** runtime with a matching PyTorch build.
- Pin versions for reproducibility; create a fresh venv/conda env.

```bash
# core (official submission)
python -m pip install \
  "torch==2.3.*" "torchvision==0.18.*" \
  "timm==1.0.*" \
  "transformers==4.44.*" "tokenizers>=0.19" \
  "numpy>=1.26,<2.1" "pandas==2.2.*" \
  "scikit-learn==1.5.*" \
  "iterative-stratification==0.1.7" \
  "albumentations==1.4.*" "opencv-python-headless==4.10.*" \
  "Pillow>=10.3" "tqdm>=4.66" "pyyaml>=6.0"
```

---

## 3. Libraries — what each is for

### 3.1 Required (official two-encoder solution)

| Library | Version (pin) | Role in the pipeline |
|---|---|---|
| **torch / torchvision** | 2.3.x / 0.18.x | Model, training loop, AMP (`bf16`), EMA, DataLoader. |
| **timm** | 1.0.x | Vision backbone (`convnextv2_tiny.fcmae_ft_in22k_in1k`, EfficientNetV2-S) **with `in_chans=2`** first-conv adaptation; drop-path; pretrained weights. |
| **transformers** | 4.44.x | Text encoder (PubMedBERT / BioLinkBERT) `AutoModel` + `AutoTokenizer`. |
| **scikit-learn** | 1.5.x | `StratifiedGroupKFold`, `f1_score`, threshold/metric utilities. |
| **iterative-stratification** | 0.1.7 | Multi-label stratification **within** family groups so rare classes survive each fold. |
| **albumentations** + **opencv-python-headless** | 1.4.x / 4.10.x | D4 dihedral, scale/shift, brightness/contrast, per-channel intensity aug on 2-channel arrays. |
| **numpy / pandas** | as pinned | `np.load` images; read/write the CSVs and `submission.csv`. |
| **Pillow, tqdm, pyyaml** | latest compatible | I/O, progress, config. |

> All of the above are standard, public, and present in the expected Kaggle Docker image —
> no exotic or role-gated dependencies.

### 3.2 Optional quality/utility (nice-to-have, still A10G-safe)
- **`torch-ema`** or a hand-rolled EMA helper (≈30 lines) — exponential moving average of
  weights.
- **`torchmetrics`** — convenient multilabel F1 during validation (or compute with sklearn).
- **`matplotlib` / `seaborn`** — EDA plots, per-class F1 diagnostics (not in official path).
- **Asymmetric Loss** — copy the ~40-line `AsymmetricLoss` from `Alibaba-MIIL/ASL`
  (`losses.py`) into the repo; no pip dependency needed. Same for a Distribution-Balanced
  loss implementation.

### 3.3 Research-only stack (Approach b / VLM-QLoRA — NOT in the official submission)
| Library | Role |
|---|---|
| **bitsandbytes** | 4-bit NF4 quantization for QLoRA. |
| **peft** | LoRA/QLoRA adapters on the VLM. |
| **trl** | SFT trainer recipe for Qwen2-VL. |
| **accelerate** | Multi-GPU / mixed-precision orchestration for sweeps. |
| **flash-attn** (optional) | Faster attention on H100. |
| **qwen-vl-utils** | Image/message formatting for Qwen2-VL. |

```bash
# research/VLM only — run on H100, do NOT ship in the official notebook
python -m pip install "bitsandbytes>=0.43" "peft>=0.12" "trl>=0.9" \
  "accelerate>=0.33" "qwen-vl-utils" "flash-attn>=2.6" --no-build-isolation
```

---

## 4. Pretrained-model artifacts (pre-download; no inference-time network)

| Artifact | Source (HF id) | Approx size | Use |
|---|---|---|---|
| **ConvNeXt-V2-Tiny** | `timm/convnextv2_tiny.fcmae_ft_in22k_in1k` | ~115 MB | Primary vision encoder. |
| **ConvNeXt-V2-Nano** | `timm/convnextv2_nano.fcmae_ft_in22k_in1k` | ~63 MB | Faster baseline / budget member. |
| **EfficientNetV2-S** | `timm/tf_efficientnetv2_s.in21k_ft_in1k` | ~85 MB | Ensemble-diversity member. |
| **PubMedBERT-base** | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | ~440 MB | Primary text encoder. |
| **BioLinkBERT-base** | `michiyasunaga/BioLinkBERT-base` | ~430 MB | Alternative text encoder (pick by OOF). |
| *(optional)* **SubCell** ViT-B/16 | `github.com/CellProfiling/subcell-analysis` | ~330 MB | Microscopy-SSL vision init (drop-marker mode). |
| *(optional)* **Cell-DINO** | PLOS pcbi.1013828 release weights | ~ViT-S/L | Alternative microscopy-SSL init. |
| *(research)* **Qwen2-VL-2B-Instruct** | `Qwen/Qwen2-VL-2B-Instruct` | ~4–5 GB | VLM-QLoRA experiment (H100). |

**Caching:** set `HF_HOME` / `TRANSFORMERS_CACHE` and pre-fetch on a networked machine, then
make the cache available to the offline runtime. timm weights download to the torch hub
cache — pre-fetch likewise. Verify every `from_pretrained` / `create_model(pretrained=True)`
call resolves from local cache before the official run.

**License/compliance:** all required artifacts are **public and reproducible** (ImageNet /
PubMed pretraining; MIT/Apache-style licenses). No private, paid, role-gated, or API-backed
models. SubCell/Cell-DINO are optional and public; confirm their licenses if shipped.

---

## 5. Data requirements

- **Provided dataset** (no external data — HPAv18 and any other external HPA images are
  **not** to be used; treat as banned/unavailable):

  ```
  <dataset_root>/
    train.csv                # id, npy_path, function_text, family_group_id, loc_<16>
    test.csv                 # id, npy_path, function_text
    sample_submission.csv    # id + 16 loc_<class> columns (prefilled with prevalence)
    location_labels.json     # index → class-name map (frozen column order)
    train/<id>.npy           # 6,304 × uint8 (2,320,320)   ~1.3 GB
    test/<id>.npy            #   762 × uint8 (2,320,320)    ~156 MB
  ```
- **Output:** `submission.csv` — `id` + 16 `loc_<class>` probability columns in `[0,1]`, one
  row per `test.csv` protein (match by `id`; no duplicate ids; no NaN/Inf). In a
  Kaggle-style layout, mirror to `./working/submission.csv`.
- **No labels for test** (`loc_*` and `family_group_id` absent in `test.csv`) — that is what
  you predict / why CV must be leave-family-out.

---

## 6. Hardware sizing cheatsheet

| Config | VRAM | Fits A10G 24 GB? | Notes |
|---|---|---|---|
| ConvNeXt-V2-Nano @320, bs 64, +PubMedBERT(256) | ~10–14 GB | ✅ | Comfortable; fast. |
| ConvNeXt-V2-Tiny @320, bs 48, +PubMedBERT(256) | ~16–20 GB | ✅ | Primary config. |
| EfficientNetV2-S @320, bs 48 | ~14–18 GB | ✅ | Diversity member. |
| ViT-B/16 or Swin-B @384 fine-tune | ~20–24 GB | ⚠️ tight | **Not recommended** (overfits unseen families). |
| Qwen2-VL-2B QLoRA, low-res | ~14–22 GB | ✅ (slow) | Research only; **7B blows the 30-min budget**. |

Use **`bf16` mixed precision** and **gradient checkpointing** if a config approaches the VRAM
ceiling. EMA adds one extra weight copy (negligible).

---

## 7. Reproducibility & determinism

- Fix all seeds (`python`, `numpy`, `torch`, CUDA); set
  `torch.backends.cudnn.deterministic=True` / `benchmark=False` for the official run
  (accept a small speed cost), or document the seed and accept minor nondeterminism.
- Persist: fold assignments (the StratifiedGroupKFold split), per-class logit biases `b_c`,
  per-channel normalization stats, and config YAML — so the official notebook reproduces the
  exact submitted CSV.
- The official `solution.ipynb`/`solution.py` must regenerate `working/submission.csv` from
  the public dataset through the modeling pipeline **in isolation** — no reading or
  mirroring of any uploaded/precomputed `submission.csv`.

---

## 8. Time & cost budget (indicative)

| Activity | Where | Rough time |
|---|---|---|
| EDA + OOF harness + strict validator | local / A10G | 1–2 h dev |
| Single fold train (ConvNeXt-V2-Tiny @320, ~12 ep) | A10G | ~10–20 min |
| 5-fold bag train | A10G or H100 | ~1–2 h (A10G) |
| Loss A/B/C + threshold + ablation sweeps | **H100** | a few GPU-hours |
| Official end-to-end run (train-on-public + predict) | A10G | **≤ 30 min** target |
| VLM-QLoRA experiment (Qwen2-VL-2B) | **H100** | several GPU-hours |

> Deallocate H100 VMs when sweeps finish. Prefer a robust CV-backed config over chasing the
> public leaderboard near any deadline.

---

## 9. Quick checklist before the first official submission

- [ ] All pretrained weights resolve from **local cache** (no inference-time network).
- [ ] StratifiedGroupKFold-on-`family_group_id` OOF harness in place; **random KFold never
      used**.
- [ ] Loss chosen by **per-fold OOF macro-F1** (BCE+pos_weight vs ASL vs DB).
- [ ] Per-class threshold **folded back as logit bias** `b_c = −logit(t_c)`; rare classes
      prevalence-matched + floored.
- [ ] Text branch passes **image-ablation** (helps) and **text-ablation** (collapses on
      unseen families) checks.
- [ ] Inference = 5-fold bag × ≤4–8 D4 TTA → logit bias → 0.5 cut, measured **inside 30 min**
      on A10G.
- [ ] `submission.csv` passes the **strict validator** (columns, ids unique & complete,
      `[0,1]` finite, no NaN).
- [ ] Official code path reproduces the submitted CSV (bias + TTA + bagging baked in); **no**
      classical ML, hardcoding, metadata, or external data.
