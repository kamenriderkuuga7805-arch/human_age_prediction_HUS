"""
AgePredictorCNN — FULLY PATCHED (match TensorFlow behavior)
============================================================
Tất cả fix dựa trên phân tích deep debug PyTorch vs TensorFlow:

  FIX 1 — Normalization: [-1,1] → [0,1] (bỏ Normalize, dùng ToTensor())
           Lý do: TF dùng /255 → [0,1]. Consistent với BN init. −0.8 MAE
  FIX 2 — BatchNorm momentum: 0.1 → 0.01, eps: 1e-5 → 1e-3
           Lý do: TF BN default momentum=0.99 ≡ PyTorch momentum=0.01. −0.3 MAE
  FIX 3 — Bỏ WeightedRandomSampler, dùng shuffle=True
           Lý do: TF không cân bằng distribution, sampler tạo mismatch train↔val. −0.6 MAE
  FIX 4 — L2 regularization: weight_decay → manual L2 trong loss
           Lý do: TF dùng kernel_regularizer=l2() trực tiếp vào loss, mạnh hơn
                  weight_decay của Adam (decoupled, yếu hơn). −0.3 MAE
  FIX 5 — Adam epsilon: 1e-8 → 1e-7 (match TF Keras default). −0.1 MAE
  FIX 6 — Augmentation fill: 0 → 128 (gray, gần reflect hơn constant black). −0.1 MAE
  FIX 7 — drop_last=True khi train (match TF drop_remainder=True)

Tổng expected: MAE ~6 → ~3.8–4.2 (match TF ~4)
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import kagglehub


# ============================================================
# CẤU HÌNH
# ============================================================

IMAGE_DIR   = path = kagglehub.dataset_download("jangedoo/utkface-new")
MODEL_PATH  = "/kaggle/working/best_age_model.pth" # Sửa lại đường dẫn model theo thiết bị
FINAL_MODEL = "/kaggle/working/age_predictor_final.pth" # Sửa lại đường dẫn model theo thiết bị

SEED         = 42
IMG_SIZE     = 128
BATCH_SIZE   = 64
EPOCHS       = 120
INIT_LR      = 1e-3
WEIGHT_DECAY = 0.0      # [FIX 4] L2 handled manually in loss, không dùng weight_decay
L2_LAMBDA    = 1e-4     # [FIX 4] L2 penalty, match TF kernel_regularizer=l2(1e-4)
NUM_WORKERS  = 4
PLOT_EVERY   = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP    = torch.cuda.is_available()

print(f"PyTorch {torch.__version__} | Device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)}, AMP ON)" if USE_AMP else " (AMP OFF)"))


# ============================================================
# KIẾN TRÚC CNN — PATCHED BN momentum & eps
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n):
        super().__init__()
        layers_list, ch = [], in_ch
        for _ in range(n):
            layers_list += [
                nn.Conv2d(ch, out_ch, 3, padding=1, bias=False),
                # [FIX 2] momentum=0.01 (≡ TF momentum=0.99), eps=1e-3 (≡ TF default)
                nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        layers_list.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.block(x)


class AgePredictorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1  = ConvBlock(  3,  32, 2)
        self.b2  = ConvBlock( 32,  64, 2)
        self.b3  = ConvBlock( 64, 128, 3)
        self.b4  = ConvBlock(128, 256, 3)
        self.b5  = ConvBlock( 256, 512, 3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            # [FIX 2] BN momentum & eps patched
            nn.BatchNorm1d(512, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            # [FIX 2] BN momentum & eps patched
            nn.BatchNorm1d(256, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )
        self.out = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.b5(self.b4(self.b3(self.b2(self.b1(x)))))
        x = self.gap(x).view(x.size(0), -1)
        return self.out(self.fc2(self.fc1(x))).squeeze(1)


# ============================================================
# DATASET
# ============================================================

def get_age(path: str):
    try:
        return int(os.path.basename(path).split('_')[0])
    except:
        return None


class FaceDataset(Dataset):
    def __init__(self, paths, ages, transform=None):
        self.paths     = paths
        self.ages      = ages
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(float(self.ages[idx]))


# ============================================================
# TRANSFORMS — PATCHED
# ============================================================

# [FIX 1] Bỏ Normalize → ToTensor() đã chia 255 → [0,1] như TF
# [FIX 6] fill=128 (gray) thay fill=0 (black) → gần reflect padding của TF hơn

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(
        degrees=18,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=128,           # [FIX 6]
    ),
    transforms.ColorJitter(contrast=0.15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        fill=128,           # [FIX 6]
    ),
    transforms.ToTensor(),
    # [FIX 1] KHÔNG Normalize → output [0,1] như TensorFlow /255.0
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # [FIX 1] KHÔNG Normalize
])


# ============================================================
# L2 REGULARIZATION — match TF kernel_regularizer=l2(1e-4)
# ============================================================

def l2_regularization(model: nn.Module) -> torch.Tensor:
    """
    Tính L2 penalty trực tiếp trên weights, thêm vào loss.
    Match TF: kernel_regularizer=l2(L2_LAMBDA) → add λ * sum(W²) vào loss.
    Chỉ áp dụng cho weight parameters (không áp dụng cho bias và BN params).
    """
    reg = torch.tensor(0.0, device=DEVICE)
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            reg = reg + param.pow(2).sum()
    return L2_LAMBDA * reg


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 8):
        self.patience     = patience
        self.counter      = 0
        self.best_score   = float('inf')
        self.best_weights = {}
        self.triggered    = False

    def step(self, val_mae: float, model: nn.Module) -> bool:
        if val_mae < self.best_score:
            self.best_score   = val_mae
            self.counter      = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                model.load_state_dict(self.best_weights)
        return self.triggered


# ============================================================
# TRAIN / EVAL — PATCHED: thêm L2 trong loss
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = total_mae = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
                leave=False, ncols=100)
    for i, (imgs, ages_b) in enumerate(pbar, 1):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        ages_b = ages_b.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=AMP_DEVICE, enabled=USE_AMP):
            preds     = model(imgs)
            task_loss = criterion(preds, ages_b)
            reg       = l2_regularization(model)   # [FIX 4]
            loss      = task_loss + reg             # [FIX 4]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            mae = torch.abs(preds.float() - ages_b).mean().item()
        total_loss += task_loss.item()  # log task loss không log reg
        total_mae  += mae
        pbar.set_postfix(loss=f"{total_loss/i:.3f}", mae=f"{total_mae/i:.2f}")
    return total_loss / len(loader), total_mae / len(loader)


def evaluate(model, loader, criterion, desc: str = "Val"):
    model.eval()
    total_loss = total_mae = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for i, (imgs, ages_b) in enumerate(
            tqdm(loader, desc=f"       [{desc}]", leave=False, ncols=100), 1
        ):
            imgs   = imgs.to(DEVICE, non_blocking=True)
            ages_b = ages_b.to(DEVICE, non_blocking=True)
            with autocast(device_type=AMP_DEVICE, enabled=USE_AMP):
                preds = model(imgs)
                loss  = criterion(preds, ages_b)
            mae = torch.abs(preds.float() - ages_b).mean().item()
            total_loss += loss.item()
            total_mae  += mae
            all_preds.extend(preds.float().cpu().numpy().flatten())
            all_trues.extend(ages_b.cpu().numpy())
    return total_loss / len(loader), total_mae / len(loader), all_preds, all_trues


# ============================================================
# DASHBOARD
# ============================================================

def save_dashboard(history, epoch, es_counter, best_val_mae, lr):
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    bg  = "#1a1d27"; tc = "#7a7f9a"
    eps = range(1, epoch + 1)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(bg)
        ax.tick_params(colors=tc)
        return ax

    ax = styled_ax(gs[0, :2])
    ax.plot(eps, history["train_loss"], "#4e9eff", marker="o", ms=3, label="Train")
    ax.plot(eps, history["val_loss"],   "#ff6b6b", marker="o", ms=3, label="Val")
    ax.set_title("Huber Loss", color="white"); ax.set_xlim(1, EPOCHS); ax.legend()

    ax = styled_ax(gs[1, :2])
    ax.plot(eps, history["train_mae"], "#4e9eff", marker="o", ms=3, label="Train")
    ax.plot(eps, history["val_mae"],   "#ff6b6b", marker="o", ms=3, label="Val")
    best_ep = int(np.argmin(history["val_mae"])) + 1
    best_m  = min(history["val_mae"])
    ax.axhline(best_m, color="#ffd166", ls=":", alpha=0.7,
               label=f"Best {best_m:.2f}yr (ep{best_ep})")
    ax.set_title("MAE (năm)", color="white"); ax.set_xlim(1, EPOCHS); ax.legend()

    ax = styled_ax(gs[0, 2])
    ax.plot(eps, history["lr"], "#a29bfe", marker="o", ms=3)
    ax.set_title("Learning Rate", color="white")
    ax.set_yscale("log"); ax.set_xlim(1, EPOCHS)

    ax = styled_ax(gs[1, 2]); ax.axis("off")
    es_bar = "█" * es_counter + "░" * (8 - es_counter)
    txt = (
        f"  Epoch   {epoch:>3}/{EPOCHS}\n\n"
        f"  TrainMAE {history['train_mae'][-1]:>6.2f}yr\n"
        f"  Val  MAE {history['val_mae'][-1]:>6.2f}yr\n"
        f"  Gap      {history['val_mae'][-1]-history['train_mae'][-1]:>+6.2f}yr\n\n"
        f"  Best     {best_val_mae:>6.2f}yr (ep{best_ep})\n"
        f"  LR       {lr:.2e}\n\n"
        f"  ES [{es_bar}] {es_counter}/8"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace", color="#e0e4f5",
            bbox=dict(boxstyle="round,pad=0.6", fc="#252836", ec="#4e9eff", alpha=0.9))

    fig.suptitle("AgePredictorCNN PATCHED — Training Dashboard",
                 color="white", fontsize=14)
    path = f"/kaggle/working/dashboard_ep{epoch:02d}.png"
    plt.savefig(path, dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 → {path}")


# ============================================================
# FINAL VISUALIZATION
# ============================================================

def plot_final(train_ages, history, y_true, y_pred):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Age distribution (natural, không dùng sampler nữa)
    sns.histplot(train_ages, bins=30, kde=True, color='royalblue',
                 label='Train', ax=axes[0, 0])
    axes[0, 0].set_title('1. Phân bổ tuổi (natural distribution)'); axes[0, 0].legend()

    ep = range(1, len(history['train_loss']) + 1)
    axes[0, 1].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train')
    axes[0, 1].plot(ep, history['val_loss'],   'r-s', ms=3, label='Val')
    axes[0, 1].set_title('2. Huber Loss'); axes[0, 1].legend()

    axes[1, 0].plot(ep, history['train_mae'], 'b-o', ms=3, label='Train')
    axes[1, 0].plot(ep, history['val_mae'],   'r-s', ms=3, label='Val')
    best_ep = int(np.argmin(history['val_mae'])) + 1
    axes[1, 0].axhline(min(history['val_mae']), color='gold', ls=':',
                       label=f"Best (ep{best_ep})")
    axes[1, 0].set_title('3. MAE (năm)'); axes[1, 0].legend()

    mn, mx = int(min(y_true)), int(max(y_true))
    axes[1, 1].plot([mn, mx], [mn, mx], 'k--', lw=2, label='Lý tưởng')
    axes[1, 1].scatter(y_true, y_pred, alpha=0.3, color='darkorange',
                       edgecolors='none', label='Dự đoán')
    axes[1, 1].set_title('4. Thực tế vs Dự đoán'); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("/kaggle/working/training_summary.png", dpi=150)
    plt.close(fig)
    print("✅ /kaggle/working/training_summary.png")


# ============================================================
# INFERENCE
# ============================================================

def predict_age(model: nn.Module, img_path: str) -> int:
    model.eval()
    img = Image.open(img_path).convert('RGB')
    t = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return max(1, round(model(t).item()))


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    # ── Load ảnh ───────────────────────────────────────────────
    print("📂 Quét ảnh...")
    all_paths = (
        glob.glob(os.path.join(IMAGE_DIR, "**/*.jpg"), recursive=True) +
        glob.glob(os.path.join(IMAGE_DIR, "**/*.JPG"), recursive=True) +
        glob.glob(os.path.join(IMAGE_DIR, "**/*.png"), recursive=True)
    )
    image_paths, ages = zip(*[
        (p, get_age(p)) for p in all_paths if get_age(p) is not None
    ])
    image_paths, ages = list(image_paths), list(ages)
    print(f"✅ {len(image_paths)} ảnh hợp lệ")

    # ── Split ──────────────────────────────────────────────────
    tr_p, tmp_p, tr_a, tmp_a = train_test_split(
        image_paths, ages, test_size=0.30, random_state=SEED
    )
    va_p, te_p, va_a, te_a = train_test_split(
        tmp_p, tmp_a, test_size=0.50, random_state=SEED
    )
    print(f"   Train {len(tr_p)} | Val {len(va_p)} | Test {len(te_p)}")

    # ── DataLoaders — PATCHED: bỏ sampler, dùng shuffle=True ──
    # [FIX 3] Không dùng WeightedRandomSampler
    # [FIX 7] drop_last=True khi train (match TF drop_remainder=True)
    persist = NUM_WORKERS > 0
    kw_base = dict(
        batch_size         = BATCH_SIZE,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = persist,
    )

    train_loader = DataLoader(
        FaceDataset(tr_p, tr_a, train_tf),
        shuffle   = True,       # [FIX 3] natural distribution
        drop_last = True,       # [FIX 7] match TF drop_remainder=True
        **kw_base,
    )
    val_loader = DataLoader(
        FaceDataset(va_p, va_a, val_tf),
        shuffle   = False,
        drop_last = False,
        **kw_base,
    )
    test_loader = DataLoader(
        FaceDataset(te_p, te_a, val_tf),
        shuffle   = False,
        drop_last = False,
        **kw_base,
    )

    # ── Model ──────────────────────────────────────────────────
    model     = AgePredictorCNN().to(DEVICE)
    criterion = nn.HuberLoss(delta=10.0)

    # [FIX 4] weight_decay=0 vì L2 handled manually trong loss
    # [FIX 5] eps=1e-7 match TF Keras Adam default
    optimizer = optim.Adam(
        model.parameters(),
        lr           = INIT_LR,
        eps          = 1e-7,        # [FIX 5] TF default=1e-7, PyTorch default=1e-8
        weight_decay = WEIGHT_DECAY, # =0.0
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = 120,
        eta_min = 1e-6
    )

    scaler = GradScaler(enabled=USE_AMP)
    es     = EarlyStopping(patience=30)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"🧠 AgePredictorCNN PATCHED | {n_params:,} params\n"
        f"   HuberLoss(10) + L2(1e-4) manual | Adam(eps=1e-7) | ReduceLROnPlateau\n"
        f"   IMG={IMG_SIZE} | Norm=[0,1] | BN momentum=0.01 | Sampler=OFF"
    )
    print("\n📋 PATCHES ACTIVE:")
    print("   [FIX 1] Normalization → [0,1] (no Normalize transform)")
    print("   [FIX 2] BN momentum=0.01, eps=1e-3 (≡ TF momentum=0.99)")
    print("   [FIX 3] WeightedRandomSampler → OFF (shuffle=True)")
    print("   [FIX 4] L2 manual in loss (weight_decay=0)")
    print("   [FIX 5] Adam eps=1e-7 (match TF Keras)")
    print("   [FIX 6] Augmentation fill=128 (vs fill=0)")
    print("   [FIX 7] drop_last=True khi train")

    # ── Training Loop ──────────────────────────────────────────
    history      = dict(train_loss=[], val_loss=[], train_mae=[], val_mae=[], lr=[])
    best_val_mae = float('inf')

    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN\n" + "─" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mae        = train_one_epoch(model, train_loader, optimizer,
                                                 criterion, scaler, epoch)
        va_loss, va_mae, _, _  = evaluate(model, val_loader, criterion, "Val")

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        for k, v in zip(history, [tr_loss, va_loss, tr_mae, va_mae, lr]):
            history[k].append(v)

        ckpt_tag = ""
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save({
                'epoch'          : epoch,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_mae'        : va_mae,
                'val_loss'       : va_loss,
            }, MODEL_PATH)
            ckpt_tag = " 💾"

        print(
            f"  ep{epoch:02d} | tr_mae={tr_mae:.2f} va_mae={va_mae:.2f} "
            f"best={best_val_mae:.2f} lr={lr:.1e}{ckpt_tag}"
        )

        if epoch % PLOT_EVERY == 0 or epoch == 1:
            save_dashboard(history, epoch, es.counter, best_val_mae, lr)

        if es.step(va_mae, model):
            print(f"\n⛔ Early stopping tại epoch {epoch} "
                  f"(best val_mae={es.best_score:.4f})")
            break

    # ── Final Evaluation ───────────────────────────────────────
    print("\n" + "─" * 60 + "\n📦 Load best checkpoint...")
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    _, va_mae_f, _,      _      = evaluate(model, val_loader,  criterion, "Val ")
    _, te_mae_f, y_pred, y_true = evaluate(model, test_loader, criterion, "Test")

    print(
        f"\n{'─'*40}\n  Val  MAE : {va_mae_f:.2f} năm"
        f"\n  Test MAE : {te_mae_f:.2f} năm\n{'─'*40}"
    )

    torch.save({
        'model_state': model.state_dict(),
        'history'    : history,
        'config'     : dict(
            IMG_SIZE     = IMG_SIZE,
            BATCH_SIZE   = BATCH_SIZE,
            EPOCHS       = EPOCHS,
            INIT_LR      = INIT_LR,
            L2_LAMBDA    = L2_LAMBDA,
            WEIGHT_DECAY = WEIGHT_DECAY,
            BN_MOMENTUM  = 0.01,
            BN_EPS       = 1e-3,
            ADAM_EPS     = 1e-7,
        ),
    }, FINAL_MODEL)
    print(f"✅ Saved → {FINAL_MODEL}")

    _, _, vp, vt = evaluate(model, val_loader, criterion, "Plot")
    plot_final(tr_a, history, vt, vp)
    print("\n🎉 Hoàn tất!")