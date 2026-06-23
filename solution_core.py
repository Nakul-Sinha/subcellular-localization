"""
Multi-Label Subcellular Localization from Microscopy + Function Text
Two-encoder vision+text fusion -> 16-way sigmoid head, leave-family-out CV.

Config via env vars (all optional, sensible defaults):
  DATA_ROOT   path containing train.csv/test.csv/train/*.npy  (default ./dataset/public)
  OUT_DIR     output dir for submission.csv + artifacts        (default ./working)
  BACKBONE    timm vision backbone                             (default convnextv2_nano.fcmae_ft_in22k_in1k)
  TEXT_MODEL  HF text encoder                                  (default microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
  USE_TEXT    1/0 enable text branch                           (default 1)
  LOSS        bce | asl                                        (default bce)
  N_FOLDS     leave-family-out folds                           (default 5)
  EPOCHS      epochs per fold                                  (default 10)
  IMG_SIZE    train/infer image size                          (default 320)
  BATCH       batch size                                       (default 32)
  TTA         1/0 D4 test-time augmentation                   (default 1)
  FAST        1 = smoke test (1 fold, 1 epoch, subset)        (default 0)
  SEED        random seed                                      (default 42)
"""
import os, json, math, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ----------------------------- config -----------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./dataset/public"))
OUT_DIR   = Path(os.environ.get("OUT_DIR", "./working"))
BACKBONE  = os.environ.get("BACKBONE", "convnextv2_nano.fcmae_ft_in22k_in1k")
TEXT_MODEL= os.environ.get("TEXT_MODEL", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
USE_TEXT  = os.environ.get("USE_TEXT", "1") == "1"
LOSS      = os.environ.get("LOSS", "bce")
N_FOLDS   = int(os.environ.get("N_FOLDS", "5"))
EPOCHS    = int(os.environ.get("EPOCHS", "10"))
IMG_SIZE  = int(os.environ.get("IMG_SIZE", "320"))
BATCH     = int(os.environ.get("BATCH", "32"))
TTA       = os.environ.get("TTA", "1") == "1"
FAST      = os.environ.get("FAST", "0") == "1"
SEED      = int(os.environ.get("SEED", "42"))
MAX_LEN   = 256
TEXT_DROP_P = 0.4
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if FAST:
    N_FOLDS, EPOCHS = 1, 1

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["nucleoplasm","nuclear_membrane","nucleoli","nucleoli_fibrillar_center",
    "nuclear_speckles","nuclear_bodies","endoplasmic_reticulum","golgi_apparatus",
    "vesicles","plasma_membrane","cytosol","mitochondria","microtubules","centrosome",
    "actin_filaments","intermediate_filaments"]
LOC_COLS = [f"loc_{c}" for c in CLASSES]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ----------------------------- data -----------------------------
def load_npy_cache(ids, split):
    """Preload all images into RAM as float32 (channel-percentile-normalized)."""
    cache = {}
    for i in ids:
        a = np.load(DATA_ROOT / split / f"{i}.npy").astype(np.float32)  # (2,H,W)
        cache[i] = a
    return cache

def norm_img(a):
    """Per-channel 1-99.5 percentile contrast stretch -> [0,1], then standardize."""
    out = np.empty_like(a)
    for c in range(a.shape[0]):
        ch = a[c]
        lo, hi = np.percentile(ch, 1.0), np.percentile(ch, 99.5)
        if hi <= lo: hi = lo + 1.0
        out[c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out

class ProtDataset(Dataset):
    def __init__(self, df, cache, tok, train=True):
        self.df = df.reset_index(drop=True)
        self.cache = cache; self.tok = tok; self.train = train
        self.ids = self.df["id"].tolist()
        self.has_y = LOC_COLS[0] in self.df.columns
        if self.has_y:
            self.Y = self.df[LOC_COLS].values.astype(np.float32)
        # pre-tokenize
        texts = self.df["function_text"].fillna("").tolist()
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
        self.input_ids = enc["input_ids"].astype(np.int64)
        self.attn = enc["attention_mask"].astype(np.int64)

    def __len__(self): return len(self.df)

    def _aug(self, img):
        # D4 dihedral: random flips + 90-deg rotations (microscopy is rotation-invariant)
        if random.random() < 0.5: img = img[:, ::-1, :]
        if random.random() < 0.5: img = img[:, :, ::-1]
        k = random.randint(0, 3)
        if k: img = np.rot90(img, k, axes=(1, 2))
        # mild brightness/contrast jitter for dark images
        if random.random() < 0.5:
            g = 1.0 + (random.random() - 0.5) * 0.3
            img = np.clip(img * g, 0, 1)
        return np.ascontiguousarray(img)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img = norm_img(self.cache[i])
        if self.train: img = self._aug(img)
        img = torch.from_numpy(img.copy())
        item = {"img": img,
                "input_ids": torch.from_numpy(self.input_ids[idx]),
                "attn": torch.from_numpy(self.attn[idx])}
        if self.has_y: item["y"] = torch.from_numpy(self.Y[idx])
        return item

# ----------------------------- model -----------------------------
import timm
from transformers import AutoModel, AutoTokenizer

class FusionModel(nn.Module):
    def __init__(self, backbone, use_text=True):
        super().__init__()
        self.use_text = use_text
        self.vision = timm.create_model(backbone, pretrained=True, in_chans=2,
                                        num_classes=0, drop_path_rate=0.1)
        vdim = self.vision.num_features
        self.img_proj = nn.Sequential(nn.Linear(vdim, 512), nn.GELU(), nn.Dropout(0.3))
        self.img_head = nn.Linear(512, 16)
        if use_text:
            self.text = AutoModel.from_pretrained(TEXT_MODEL)
            tdim = self.text.config.hidden_size
            self.txt_proj = nn.Sequential(nn.Linear(tdim, 256), nn.GELU(), nn.Dropout(0.3))
            self.txt_head = nn.Linear(256, 16)
            self.head = nn.Sequential(nn.Linear(512 + 256, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, 16))
        else:
            self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, 16))

    def forward(self, img, input_ids=None, attn=None, text_drop_p=0.0):
        iv = self.img_proj(self.vision(img))
        img_logits = self.img_head(iv)
        if self.use_text:
            out = self.text(input_ids=input_ids, attention_mask=attn).last_hidden_state
            mask = attn.unsqueeze(-1).float()
            tv = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)   # masked mean pool
            tv = self.txt_proj(tv)
            txt_logits = self.txt_head(tv)
            if self.training and text_drop_p > 0:
                keep = (torch.rand(tv.size(0), 1, device=tv.device) > text_drop_p).float()
                tv = tv * keep
            fused = self.head(torch.cat([iv, tv], 1))
            return fused, img_logits, txt_logits
        return self.head(iv), img_logits, None

# ----------------------------- loss -----------------------------
class ASL(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__(); self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def forward(self, logits, y):
        xs = torch.sigmoid(logits)
        xs_pos, xs_neg = xs, 1 - xs
        if self.clip > 0: xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt = xs_pos * y + xs_neg * (1 - y)
        g = self.gp * y + self.gn * (1 - y)
        loss *= (1 - pt) ** g
        return -loss.mean()

def make_loss(pos_weight):
    if LOSS == "asl": return ASL()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return lambda logits, y: bce(logits, y)

# ----------------------------- metric -----------------------------
from sklearn.metrics import f1_score
F1_BASE = 0.043  # predict-only-nucleoplasm anchor (computed on train)

def raw_macro_f1(y_true, y_prob, thr=0.5):
    yt = (y_true >= 0.5).astype(int); yp = (y_prob >= thr).astype(int)
    f1s = [f1_score(yt[:, j], yp[:, j], zero_division=0) for j in range(16) if yt[:, j].sum() > 0]
    return float(np.mean(f1s))

def loc_skill(raw): return float(np.clip((raw - F1_BASE) / (1 - F1_BASE), 0, 1))

# ----------------------------- EMA -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

# ----------------------------- train one fold -----------------------------
def train_fold(tr_df, va_df, cache, tok, pos_weight, tag=""):
    model = FusionModel(BACKBONE, USE_TEXT).to(DEVICE)
    crit = make_loss(pos_weight.to(DEVICE))
    no_text_params, text_params = [], []
    for n, p in model.named_parameters():
        (text_params if n.startswith("text.") else no_text_params).append(p)
    opt = torch.optim.AdamW([
        {"params": no_text_params, "lr": 3e-4},
        {"params": text_params, "lr": 2e-5},
    ], weight_decay=1e-2)
    tr_ds = ProtDataset(tr_df, cache, tok, train=True)
    va_ds = ProtDataset(va_df, cache, tok, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    steps = len(tr_dl) * EPOCHS
    warmup = max(1, int(0.05 * steps))
    def lr_lambda(s):
        if s < warmup: return s / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))   # cosine decay, holds peak LR longer than OneCycle
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, 0.999)
    for ep in range(EPOCHS):
        model.train()
        for b in tr_dl:
            img = b["img"].to(DEVICE, non_blocking=True); y = b["y"].to(DEVICE)
            ii = b["input_ids"].to(DEVICE); am = b["attn"].to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                fused, il, tl = model(img, ii, am, text_drop_p=TEXT_DROP_P)
                loss = crit(fused, y) + 0.3 * crit(il, y) + (0.3 * crit(tl, y) if tl is not None else 0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
        log(f"  {tag} epoch {ep+1}/{EPOCHS} loss {loss.item():.4f}")
    # validate with EMA weights
    eval_model = FusionModel(BACKBONE, USE_TEXT).to(DEVICE); ema.copy_to(eval_model); eval_model.eval()
    va_prob = predict(eval_model, va_dl, tta=False)
    return ema.shadow, va_prob

@torch.no_grad()
def predict(model, dl, tta=True):
    model.eval(); probs = []
    flips = [None, "h", "v", "hv"] if tta else [None]
    for b in dl:
        img0 = b["img"].to(DEVICE); ii = b["input_ids"].to(DEVICE); am = b["attn"].to(DEVICE)
        acc = 0
        for f in flips:
            img = img0
            if f == "h": img = torch.flip(img, [2])
            elif f == "v": img = torch.flip(img, [3])
            elif f == "hv": img = torch.flip(img, [2, 3])
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                fused, _, _ = model(img, ii, am)
            acc = acc + torch.sigmoid(fused).float()
        probs.append((acc / len(flips)).cpu().numpy())
    return np.concatenate(probs, 0)

# ----------------------------- threshold lever -----------------------------
def fit_logit_bias(oof_prob, oof_true):
    """Per-class: find F1-optimal threshold t_c on OOF, return logit bias b_c=-logit(t_c)
       so the grader's fixed 0.5 lands at t_c. Rare classes use prevalence matching."""
    yt = (oof_true >= 0.5).astype(int)
    bias = np.zeros(16, dtype=np.float32)
    for j in range(16):
        p = oof_prob[:, j]; t = yt[:, j]
        prev = t.mean()
        if t.sum() < 10:   # rare: prevalence-match the positive-prediction rate
            q = np.quantile(p, 1 - max(prev, 1e-3))
            tc = float(np.clip(q, 0.05, 0.95))
        else:
            best_f1, tc = -1, 0.5
            for cand in np.linspace(0.05, 0.95, 91):
                f1 = f1_score(t, (p >= cand).astype(int), zero_division=0)
                if f1 > best_f1: best_f1, tc = f1, cand
            tc = float(np.clip(tc, 0.05, 0.95))
        bias[j] = -math.log(tc / (1 - tc))
    return bias

def apply_bias(prob, bias):
    logit = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / (1 - np.clip(prob, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(logit + bias[None, :])))

# ----------------------------- main -----------------------------
def main():
    log(f"config: backbone={BACKBONE} use_text={USE_TEXT} loss={LOSS} folds={N_FOLDS} epochs={EPOCHS} img={IMG_SIZE} batch={BATCH} fast={FAST} device={DEVICE}")
    tr = pd.read_csv(DATA_ROOT / "train.csv")
    te = pd.read_csv(DATA_ROOT / "test.csv")
    if FAST:
        tr = tr.groupby("family_group_id").head(2).head(400).reset_index(drop=True)
        te = te.head(50).reset_index(drop=True)
    log(f"train {tr.shape} test {te.shape}")
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)

    # pos_weight from hard prevalence
    Yh = (tr[LOC_COLS].values >= 0.5).astype(np.float32)
    pos = Yh.sum(0); neg = len(Yh) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1, 10), dtype=torch.float32)

    log("loading image cache to RAM...")
    tr_cache = load_npy_cache(tr["id"].tolist(), "train")
    te_cache = load_npy_cache(te["id"].tolist(), "test")

    # leave-family-out folds
    from sklearn.model_selection import StratifiedGroupKFold
    strat = Yh.argmax(1)  # dominant class for stratification proxy
    sgkf = StratifiedGroupKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    folds = list(sgkf.split(tr, strat, groups=tr["family_group_id"].values))
    if FAST: folds = folds[:1]

    oof = np.zeros((len(tr), 16), dtype=np.float32)
    oof_mask = np.zeros(len(tr), dtype=bool)
    test_prob_acc = np.zeros((len(te), 16), dtype=np.float32)
    n_used = 0
    te_ds = ProtDataset(te, te_cache, tok, train=False)
    te_dl = DataLoader(te_ds, batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    def finalize(n_used):
        """Compute OOF metric + threshold lever and write submission (incremental-safe)."""
        ofm = oof_mask
        if ofm.sum() == 0 or n_used == 0: return None
        test_prob = test_prob_acc / n_used
        raw_global = raw_macro_f1(tr[LOC_COLS].values[ofm], oof[ofm])
        bias = fit_logit_bias(oof[ofm], tr[LOC_COLS].values[ofm])
        oof_biased = apply_bias(oof[ofm], bias)
        raw_biased = raw_macro_f1(tr[LOC_COLS].values[ofm], oof_biased)
        test_final = apply_bias(test_prob, bias)
        sub = pd.DataFrame({"id": te["id"].values})
        for j, c in enumerate(LOC_COLS): sub[c] = test_final[:, j]
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        np.save(OUT_DIR / "oof.npy", oof); np.save(OUT_DIR / "bias.npy", bias)
        json.dump({"folds_done": int(n_used), "raw_f1_05": raw_global, "raw_f1_bias": raw_biased,
                   "locskill_05": loc_skill(raw_global), "locskill_bias": loc_skill(raw_biased),
                   "config": {"backbone": BACKBONE, "use_text": USE_TEXT, "loss": LOSS,
                              "folds": N_FOLDS, "epochs": EPOCHS, "img": IMG_SIZE}},
                  open(OUT_DIR / "metrics.json", "w"), indent=2)
        log(f"  [partial after {n_used} fold(s)] OOF rawF1 0.5={raw_global:.4f} +bias={raw_biased:.4f} "
            f"LocSkill={loc_skill(raw_biased):.4f}  -> wrote submission {sub.shape}")
        return raw_biased

    for fi, (tri, vai) in enumerate(folds[:N_FOLDS]):
        t0 = time.time()
        tr_df, va_df = tr.iloc[tri], tr.iloc[vai]
        log(f"=== fold {fi+1}/{N_FOLDS}  train {len(tr_df)} val {len(va_df)} (families disjoint) ===")
        weights, va_prob = train_fold(tr_df, va_df, tr_cache, tok, pos_weight, tag=f"f{fi}")
        oof[vai] = va_prob; oof_mask[vai] = True
        raw = raw_macro_f1(va_df[LOC_COLS].values, va_prob)
        log(f"  fold {fi+1} OOF rawF1 {raw:.4f}  LocSkill {loc_skill(raw):.4f}  ({time.time()-t0:.0f}s)")
        m = FusionModel(BACKBONE, USE_TEXT).to(DEVICE); m.load_state_dict(weights); m.eval()
        test_prob_acc += predict(m, te_dl, tta=TTA); n_used += 1
        del m; torch.cuda.empty_cache()
        finalize(n_used)   # incremental write: valid submission after every fold

    log("DONE")

if __name__ == "__main__":
    main()
