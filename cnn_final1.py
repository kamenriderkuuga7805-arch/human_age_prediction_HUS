import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ & HỆ THỐNG
# ==========================================
IMAGE_DIR = "C:/Users/Thang/PycharmProjects/pythonProject/age_predictions/UTKFace"  # <-- Thay đường dẫn thư mục ảnh của bạn vào đây

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
RIDGE_LAMBDA = 0.01  # Hệ số phạt L2 tương đương alpha = 0.01 trong hồi quy Ridge
NUM_WORKERS = 8


# ==========================================
# 2. ĐỊNH NGHĨA MẠNG CNN TỰ BUILD (CUSTOM CNN)
# ==========================================
class CustomAgeCNN(nn.Module):
    def __init__(self):
        super(CustomAgeCNN, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)  # Đầu ra hồi quy (1 node)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        return self.classifier(x)


# ==========================================
# 3. DATASET VÀ CÁC HÀM TRỢ GIÚP
# ==========================================
def extract_age_from_filename(filename):
    try:
        return int(os.path.basename(filename).split('_')[0])
    except:
        return None


class FaceDataset(Dataset):
    def __init__(self, paths, ages, transform=None):
        self.paths = paths
        self.ages = ages
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        age = float(self.ages[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)


# ==========================================
# 4. HÀM VẼ TOÀN BỘ 3 BIỂU ĐỒ TRỰC QUAN HÓA
# ==========================================
def plot_all_results(train_ages, sample_weights, history, y_true, y_pred):
    print("📊 Đang khởi tạo và dựng bộ 3 biểu đồ trực quan...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Đồ thị 1: So sánh phân bổ dữ liệu trước/sau khi cân bằng bằng Sampler
    indices = np.arange(len(train_ages))
    prob = np.array(sample_weights) / sum(sample_weights)
    resampled_indices = np.random.choice(indices, size=len(train_ages), replace=True, p=prob)
    resampled_ages = [train_ages[i] for i in resampled_indices]

    sns.histplot(train_ages, bins=30, kde=True, color='royalblue', label='Ban đầu', ax=axes[0])
    sns.histplot(resampled_ages, bins=30, kde=True, color='seagreen', label='Sau cân bằng', ax=axes[0], alpha=0.6)
    axes[0].set_title('1. Phân bổ tuổi dữ liệu', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tuổi')
    axes[0].set_ylabel('Số lượng mẫu')
    axes[0].legend()

    # Đồ thị 2: Loss Curve qua các Epoch
    epochs_range = range(1, len(history['train_loss']) + 1)
    axes[1].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    axes[1].plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss')
    axes[1].set_title('2. Loss Curve (MSE) qua từng Epoch', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Đồ thị 3: Biểu đồ dự đoán và giá trị thực tế (Actual vs Predicted)
    min_age, max_age = int(min(y_true)), int(max(y_true))
    axes[2].plot([min_age, max_age], [min_age, max_age], color='black', linestyle='--', linewidth=2,
                 label='Lý tưởng (Đúng 100%)')
    axes[2].scatter(y_true, y_pred, alpha=0.3, color='darkorange', edgecolors='none', label='Mẫu dự đoán')
    axes[2].set_title('3. Tương quan: Thực tế vs Dự đoán', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Tuổi thật (Ground Truth)')
    axes[2].set_ylabel('Tuổi dự đoán ($\hat{y}$)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. KHỐI THỰC THI CHÍNH (MAIN)
# ==========================================
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"🔥 Thiết bị đang chạy: {device}")

    print("📂 Đang quét thư mục ảnh...")
    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

    image_paths, ages = [], []
    for path in all_image_paths:
        age = extract_age_from_filename(path)
        if age is not None:
            image_paths.append(path)
            ages.append(age)

    print(f"✅ Tổng số ảnh hợp lệ: {len(image_paths)}")

    # Chia tập Train / Validation
    train_paths, val_paths, train_ages, val_ages = train_test_split(
        image_paths, ages, test_size=0.2, random_state=42
    )

    # Tính toán trọng số xử lý mất cân bằng
    train_age_counts = Counter(train_ages)
    sample_weights = [1.0 / train_age_counts[age] for age in train_ages]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Khởi tạo Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FaceDataset(train_paths, train_ages, transform=train_transform)
    val_dataset = FaceDataset(val_paths, val_ages, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Khởi tạo Model, Loss, Optimizer
    model = CustomAgeCNN().to(device)
    criterion = nn.MSELoss()

    # Kích hoạt Ridge bằng cách truyền RIDGE_LAMBDA (0.01) vào weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=RIDGE_LAMBDA)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_mae = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    print("\n🚀 Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        start_time = time.time()  # Bắt đầu đo thời gian epoch

        # --- PHASE: TRAIN ---
        model.train()
        running_loss, running_mae = 0.0, 0.0
        for images, ages_batch in train_loader:
            images = images.to(device, non_blocking=True)
            ages_batch = ages_batch.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, ages_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_mae += torch.sum(torch.abs(outputs - ages_batch)).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_mae = running_mae / len(train_dataset)

        # --- PHASE: VALIDATION ---
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for images, ages_batch in val_loader:
                images = images.to(device, non_blocking=True)
                ages_batch = ages_batch.to(device, non_blocking=True).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, ages_batch)

                val_loss += loss.item() * images.size(0)
                val_mae += torch.sum(torch.abs(outputs - ages_batch)).item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_mae = val_mae / len(val_dataset)

        scheduler.step(epoch_val_loss)

        # Ghi nhận lịch sử
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_mae'].append(epoch_mae)
        history['val_mae'].append(epoch_val_mae)

        end_time = time.time()
        epoch_duration = end_time - start_time  # Tính tổng thời gian chạy 1 epoch

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | T.gian: {epoch_duration:.1f}s | "
              f"Train Loss: {epoch_loss:.2f} - Val Loss: {epoch_val_loss:.2f} | "
              f"Train MAE: {epoch_mae:.2f} - Val MAE: {epoch_val_mae:.2f}")

        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            torch.save(model.state_dict(), 'best_custom_ridge_model.pth')

    print(f"\n🎉 Huấn luyện hoàn tất! Tải mô hình tốt nhất để chạy dự đoán vẽ đồ thị cuối...")

    # --- PHASE: LẤY KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG ĐỂ VẼ BIỂU ĐỒ 3 ---
    model.load_state_dict(torch.load('best_custom_ridge_model.pth'))
    model.eval()

    final_preds, final_trues = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            final_preds.extend(outputs.cpu().numpy().flatten())
            final_trues.extend(labels.numpy())

    # Gọi hàm xuất bộ 3 biểu đồ ra màn hình
    plot_all_results(train_ages, sample_weights, history, final_trues, final_preds)