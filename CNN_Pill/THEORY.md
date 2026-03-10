# Lý Thuyết Phân Loại Viên Thuốc Bằng CNN

## Mục lục

1. [Tổng quan về Computer Vision](#1-tổng-quan-về-computer-vision)
2. [Convolutional Neural Network (CNN)](#2-convolutional-neural-network-cnn)
3. [EfficientNetV2 Architecture](#3-efficientnetv2-architecture)
4. [Transfer Learning](#4-transfer-learning)
5. [NDC Code System](#5-ndc-code-system)
6. [Quy trình Training](#6-quy-trình-training)
7. [Data Augmentation](#7-data-augmentation)
8. [Loss Function & Optimization](#8-loss-function--optimization)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Resume Training & Checkpoint Management](#10-resume-training--checkpoint-management)
11. [Early Stopping](#11-early-stopping)

---

## 1. Tổng quan về Computer Vision

### 1.1 Computer Vision là gì?

Computer Vision (CV) là lĩnh vực trí tuệ nhân tạo tập trung vào việc giúp máy tính "nhìn" và hiểu được hình ảnh. Các bài toán CV bao gồm:
- **Image Classification:** Phân loại hình ảnh vào các danh mục
- **Object Detection:** Phát hiện và định vị vật thể
- **Segmentation:** Phân chia hình ảnh thành các vùng
- **Image Generation:** Tạo mới hình ảnh

### 1.2 Vấn đề phân loại viên thuốc

Bài toán phân loại viên thuốc là một bài toán **Fine-grained Image Classification** - phân loại chi tiết, nơi:
- Sự khác biệt giữa các lớp rất nhỏ
- Cùng một loại thuốc có thể có màu sắc, kích thước khác nhau
- Nhiều loại thuốc trông rất giống nhau
- Cần quan sát chi tiết nhỏ như chữ, số, logo trên bề mặt viên thuốc

**Thách thức:**
- 960 lớp thuốc khác nhau
- Chỉ có ~3-4 mẫu cho mỗi lớp
- Sự biến đổi cao trong光照 (lighting), góc chụp, nền

---

## 2. Convolutional Neural Network (CNN)

### 2.1 Tại sao dùng CNN?

CNN được thiết kế đặc biệt cho dữ liệu hình ảnh nhờ các đặc điểm:
- **Local Connectivity:** Mỗi neuron chỉ kết nối với một vùng nhỏ của input
- **Parameter Sharing:** Cùng một bộ filter dùng cho toàn bộ ảnh
- **Translation Invariance:** Nhận ra đối tượng dù nó ở vị trí nào

### 2.2 Cấu trúc CNN cơ bản

```
Input Image (H x W x 3)
    ↓
Convolutional Layer (Feature Extraction)
    ↓
Activation Function (ReLU)
    ↓
Pooling Layer (Downsampling)
    ↓
Fully Connected Layer (Classification)
    ↓
Output (Class Probabilities)
```

### 2.3 Convolutional Layer

**Hoạt động:**
- Một filter (kernel) trượt qua ảnh
- Tính tích chập (convolution) giữa filter và vùng ảnh
- Tạo ra feature map

**Công thức:**
```
Output(i,j) = Σ Σ Input(i+m, j+n) × Filter(m,n)
              m   n
```

**Ví dụ:** Filter 3x3 trượt qua ảnh 224x224 → Feature map 222x222

### 2.4 Pooling Layer

Giảm kích thước feature map, giữ thông tin quan trọng:

**Max Pooling:**
```
| 1 | 3 | 2 | 4 |         | 3 | 4 |
| 8 | 6 | 7 | 5 |   →     | 8 | 7 |
| 9 | 2 | 1 | 3 |         | 9 | 3 |
| 4 | 5 | 6 | 7 |         | 5 | 7 |
   (2x2 pool, stride=2)
```

### 2.5 Activation Function

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```

- Nếu x < 0: output = 0
- Nếu x ≥ 0: output = x

**Tại sao ReLU?**
- Tính toán đơn giản, nhanh
- Giảm vanish gradient
- Sparse activation (chỉ ~50% neuron active)

---

## 3. EfficientNetV2 Architecture

### 3.1 Ý tưởng chính của EfficientNet

EfficientNet tối ưu hóa trade-off giữa:
- **Accuracy** (Độ chính xác)
- **Model Size** (Kích thước mô hình)
- **Speed** (Tốc độ inference)

### 3.2 Compound Scaling

EfficientNet sử dụng **Compound Scaling** - scaling đồng thời:
- **Depth (Sâu):** Thêm layers
- **Width (Rộng):** Tăng filters
- **Resolution (Cao):** Tăng input size

**Công thức scaling:**
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

with α × β² × γ² ≈ 2 (constraint)
```

### 3.3 EfficientNetV2 vs V1

| Đặc điểm | EfficientNetV1 | EfficientNetV2 |
|----------|---------------|---------------|
| Training | Slow | Fast (FLOPs-based) |
| Inference | Good | Better |
| Augmentation | Heavy | Lighter |
| Architecture | MBConv | MBConv + Fused-MBConv |

### 3.4 MBConv Block

```
Input (H×W×C)
    ↓
Expansion (1×1 Conv) → Expand channels
    ↓
Depthwise Conv (3×3) → Spatial features
    ↓
Squeeze & Excitation → Channel attention
    ↓
Projection (1×1 Conv) → Reduce channels
    ↓
Skip Connection (if input==output)
    ↓
Output
```

### 3.5 Fused-MBConv (V2 mới)

Thay thế Depthwise + Projection bằng một Conv lớn:
- Tăng tốc độ training
- Giảm memory access

### 3.6 EfficientNetV2-S Architecture (sử dụng trong dự án)

```
Input: 224 × 224 × 3
    ↓
Stem: Conv3×3, 32 channels
    ↓
Stage 1: F-MBConv1, k3×3, 64 channels, ×2
    ↓
Stage 2: F-MBConv2, k3×3, 64 channels, ×4
    ↓
Stage 3: F-MBConv2, k3×3, 128 channels, ×4
    ↓
Stage 4: MBConv4, k5×5, 256 channels, ×6
    ↓
Stage 5: MBConv6, k5×5, 512 channels, ×6
    ↓
Stage 6: MBConv6, k5×5, 1024 channels, ×6
    ↓
Stage 7: MBConv6, k3×3, 1024 channels, ×6
    ↓
Head: Conv1×1, 1280 channels
    ↓
Pooling + Classifier
    ↓
Output: 960 classes
```

### 3.7 Implementation trong dự án

Mô hình sử dụng thư viện **timm** (PyTorch Image Models) để load EfficientNetV2-S:

```python
# Từ efficientnet_pill.py
self.backbone = timm.create_model(
    model_name='tf_efficientnetv2_s',
    pretrained=True,
    num_classes=0,  # Remove classification head
    global_pool='avg'  # Global average pooling
)
```

**Classifier Head tùy chỉnh:**

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=dropout_rate),      # 0.5
    nn.Linear(self.feature_dim, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout_head_extra), # 0.25
    nn.Linear(512, num_classes)      # num_classes = 960
)
```

---

## 4. Transfer Learning

### 4.1 Khái niệm Transfer Learning

**Transfer Learning** là kỹ thuật sử dụng kiến thức từ bài toán này để giải bài toán khác.

```
Source Task: ImageNet (1000 classes, 1.2M images)
                ↓
            Pre-trained Model
                ↓
Target Task: Pill Classification (960 classes, 3.7K images)
```

### 4.2 Tại sao Transfer Learning hiệu quả?

**Giả thiết:** Các features học được từ ImageNet có thể tái sử dụng:
- Edge, texture detectors (low-level)
- Shape, pattern detectors (mid-level)
- Object part detectors (high-level)

**Lợi ích:**
- Giảm thời gian training
- Cải thiện accuracy với ít dữ liệu
- Avoid overfitting

### 4.3 Fine-tuning Strategies

**Strategy 1: Freeze Backbone**
```
Backbone (frozen): Không update weights
Head (trainable): Update weights
```

**Strategy 2: Unfreeze + Lower LR**
```
Backbone: LR = 0.002 × 0.001 = 0.000002 (slow update)
Head: LR = 0.002 (fast update)
```

### 4.4 Training Strategy trong dự án

**Config mặc định:**
```yaml
training:
  unfreeze_epoch: 999  # Không unfreeze backbone
```

**Phase 1 (Epoch 0 - unfreeze_epoch-1):**
- Freeze EfficientNetV2 backbone
- Chỉ train classifier head
- Learning rate: 0.002
- Mục đích: Head học cách phân loại với features có sẵn

**Phase 2 (Epoch unfreeze_epoch+):**
- Nếu `unfreeze_epoch < num_epochs`, unfreeze toàn bộ
- Backbone LR: 0.002 × backbone_lr_multiplier (mặc định 0.001)
- Fine-tune toàn bộ mạng

---

## 5. NDC Code System

### 5.1 NDC là gì?

**NDC (National Drug Code)** là mã định danh thuốc duy nhất tại Mỹ, được quản lý bởi FDA.

### 5.2 Cấu trúc NDC

```
Format: XXXXX-XXXX-XX_HASH

Labeler  | Product | Package
(5 digits)| (4 digits)| (2 digits)
  00591  -  5307  -   01    _   BE305F72
```

**Các thành phần:**

1. **Labeler Code (5 digits):**
   - Mã nhà sản xuất/marketing
   - Do FDA cấp
   - Ví dụ: 00591 = Actavis Pharma

2. **Product Code (4 digits):**
   - Mã sản phẩm
   - Do hãng tự định nghĩa
   - Mỗi strength, dosage form có code khác

3. **Package Code (2 digits):**
   - Mã loại đóng gói
   - Ví dụ: 01 = bottle 100 tablets

4. **Hash (8 chars):**
   - Identifier duy nhất trong dataset
   - Để phân biệt các phiên bản

### 5.3 Ví dụ minh họa

```
NDC: 00591-5307-01_BE305F72
    ↓
Labeler: 00591 → Actavis Pharma, Inc.
Product: 5307 → Promethazine HCl 25mg
Package: 01 → Bottle 100 tablets
    ↓
Generic Name: Promethazine Hydrochloride
Strength: 25 mg
Dosage Form: Tablet
Route: Oral
```

### 5.4 Tra cứu NDC

**Các nguồn:**
- NIH Drug Portal: https://druginfo.nlm.nih.gov/drugportal/
- FDA DailyMed: https://dailymed.nlm.nih.gov/
- FDA NDC Directory: https://fdasis.nlm.nih.gov/

---

## 6. Quy trình Training

### 6.1 Data Flow

```
Raw Image (various sizes)
    ↓
Resize to 224×224
    ↓
Data Augmentation (train only)
    ↓
Normalize (ImageNet stats)
    ↓
Convert to Tensor (3, 224, 224)
    ↓
Batch (Batch Size, 3, 224, 224)
    ↓
Forward pass through model
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Gradient Clipping (max_grad_norm = 0.5)
    ↓
Update weights (Optimizer)
```

### 6.2 Forward Pass

```
Input: x (Batch, 3, 224, 224)
    ↓
EfficientNetV2 Backbone (from timm)
    Output: features (Batch, 1280)
    ↓
Classifier Head
    - Dropout(0.5)
    - Linear(1280 → 512)
    - BatchNorm1d + ReLU
    - Dropout(0.25)
    - Linear(512 → 960)
    ↓
Output: logits (Batch, 960)
    ↓
Softmax
    Output: probabilities (Batch, 960)
```

### 6.3 Backpropagation

```
Loss = CrossEntropy(y_pred, y_true)
    ↓
∂Loss/∂Weights (Gradient)
    ↓
Gradient Clipping (max_norm = 0.5)
    ↓
Optimizer Update (AdamW):
    weights = weights - lr × gradient
    ↓
New Weights
```

### 6.4 Đường dẫn dữ liệu

**Đường dẫn ảnh trong dataset:**
```
data/classification_data/fcn_mix_weight/
├── dc_224/          # Cropped images (224x224)
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── dr_224/          # Reflected images (224x224)
    ├── 100.jpg
    ├── 101.jpg
    └── ...
```

**Đường dẫn CSV labels:**
```
data/
└── all_labels.csv        # Mapping filename -> NDC label
```

**Đường dẫn label encoder:**
```
data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/
└── label_encoder.pickle
```

### 6.5 Data Split Strategy

Dự án sử dụng **Random Split** thay vì K-Fold Cross-Validation:

```
Total Dataset (3,728 images)
    ↓
Random Split (seed=42):
    ↓
┌─────────────────────────────────┐
│ Train: 70% (2,610 images)    │
│ Validation: 15% (559 images)  │
│ Test: 15% (559 images)        │
└─────────────────────────────────┘
```

**Lợi ích của Random Split:**
- Sử dụng toàn bộ dữ liệu hiệu quả hơn
- Đơn giản hơn K-Fold
- Khuyến nghị cho dataset ePillID

---

## 7. Data Augmentation

### 7.1 Mục đích

Tăng diversity của training data để:
- Giảm overfitting
- Mô hình học invariant features
- Mô hình generalize tốt hơn

### 7.2 Các kỹ thuật sử dụng (Dataset Module)

**1. Random Horizontal Flip (p=0.5)**
```
Original: [ ][ ][ ]
          [P][I][L]
          [ ][ ][ ]

Flipped:  [ ][ ][ ]
         [L][I][P]
         [ ][ ][ ]
```

**2. Random Vertical Flip (p=0.3)**
```
Lật ngược ảnh theo chiều dọc
```

**3. Random Rotation (±30°)**
```
Xoay ảnh ngẫu nhiên trong khoảng ±30 độ
```

**4. Random Affine**
```
- Translate: ±15% (dịch chuyển)
- Scale: 0.85-1.15 (thu phóng)
- Shear: 5° (lệch)
```

**5. Random Resized Crop**
```
- Scale: 0.7-1.0 (kích thước random)
- Resize về 224x224
```

**6. Color Jitter**
```
brightness:  ±0.3
contrast:    ±0.3
saturation:  ±0.3
hue:         ±0.1
```

**7. Random Grayscale (p=0.1)**
```
Chuyển sang grayscale với xác suất 10%
```

**8. Gaussian Blur**
```
kernel_size: 3
sigma: 0.1-2.0
```

**9. Normalization**
```
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std  = [0.229, 0.224, 0.225]  # ImageNet std

normalized = (image - mean) / std
```

### 7.3 Training vs Validation

| Transform | Train | Val/Test |
|-----------|-------|----------|
| Resize (110%) | ✓ | ✗ |
| Horizontal Flip | ✓ | ✗ |
| Vertical Flip | ✓ | ✗ |
| Rotation (±30°) | ✓ | ✗ |
| Random Affine | ✓ | ✗ |
| Random Resized Crop | ✓ | ✗ |
| Color Jitter | ✓ | ✗ |
| Random Grayscale | ✓ | ✗ |
| Gaussian Blur | ✓ | ✗ |
| Resize (224x224) | ✗ | ✓ |
| Normalize | ✓ | ✓ |

---

## 8. Loss Function & Optimization

### 8.1 Cross-Entropy Loss

**Công thức:**
```
L = -Σ y_i × log(ŷ_i)
    i
```

Trong đó:
- `y_i`: Ground truth (0 hoặc 1)
- `ŷ_i`: Predicted probability

**Với one-hot encoding:**
```
L = -log(ŷ_true_class)
```

### 8.2 Label Smoothing

**Vấn đề:** Overconfidence - mô hình quá tự tin (probability → 100%)

**Giải pháp Label Smoothing:**
```
y_smooth = (1 - α) × y_one_hot + α × K

Where:
- α = 0.1 (smoothing factor trong config)
- K = số classes
```

**Ví dụ:**
```
Without smoothing:
y = [0, 0, 1, 0, 0, ..., 0]

With smoothing (α=0.1, K=960):
y = [0.0001, 0.0001, 0.9, 0.0001, 0.0001, ..., 0.0001]
```

### 8.3 Class Weights

**Vấn đề:** Imbalanced dataset - một số lớp có ít mẫu

**Giải pháp:** Weighted loss
```
weight_i = Total_samples / (Num_classes × Count_class_i)
```

Lớp có ít mẫu → weight cao hơn → loss contribution lớn hơn

### 8.4 Optimizer: AdamW

**AdamW = Adam + Weight Decay**

**Adam:**
```
m_t = β₁×m_{t-1} + (1-β₁)×g_t
v_t = β₂×v_{t-1} + (1-β₂)×g_t²

θ_t = θ_{t-1} - α × m_t / (√v_t + ε)
```

**AdamW tách weight decay:**
```
θ_t = θ_{t-1} - η×λ×θ_{t-1} - α × m_t / (√v_t + ε)
```

**Hyperparameters:**
- Learning rate (α): 0.002
- β₁: 0.9
- β₂: 0.999
- Weight decay (λ): 0.01

### 8.5 Learning Rate Scheduler: Cosine Annealing

```
LR_t = LR_min + 0.5 × (LR_max - LR_min) × (1 + cos(π×t/T))
```

```
LR
│
│     ___LR_max___
│    /             \
│   /               \___LR_min
│  /
│_/
 └────────────────────→ Epoch
```

**Lợi ích:**
- LR giảm dần → fine-tuning ở cuối
- Cosine decay mượt mà hơn so với step decay

### 8.6 Gradient Clipping

```
if ||gradient|| > max_norm:
    gradient = gradient × max_norm / ||gradient||
```

**Mục đích:** Tránh exploding gradient
**Config:** `max_grad_norm = 0.5`

---

## 9. Evaluation Metrics

### 9.1 Accuracy

**Top-1 Accuracy:**
```
% correct predictions where true class is #1 prediction
```

**Top-5 Accuracy:**
```
% correct predictions where true class is in top 5 predictions
```

### 9.2 Kết quả dự án

| Metric | Giá trị |
|--------|---------|
| Top-1 Accuracy | ~23.46% |
| Top-5 Accuracy | ~43% |
| Num Classes | 960 |
| Samples/Class | ~3.9 |

**Lưu ý:** Accuracy thấp là bình thường vì:
- Số lớp rất lớn (960)
- Số mẫu ít (3-4/class)
- Fine-grained classification (các lớp rất giống nhau)

---

## 10. Resume Training & Checkpoint Management

### 10.1 Checkpoint là gì?

**Checkpoint** là snapshot trạng thái của mô hình tại một thời điểm training, bao gồm:
- **Model state dict:** Weights của tất cả layers
- **Optimizer state dict:** Trạng thái optimizer (momentum, etc.)
- **Epoch:** Epoch hiện tại
- **Best metrics:** accuracy tốt nhất đạt được
- **Config:** Cấu hình training đã dùng

### 10.2 Cấu trúc checkpoint file

```python
checkpoint = {
    'epoch': epoch,                              # Epoch hiện tại
    'model_state_dict': model.state_dict(),          # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'best_acc1': best_acc1,                      # Best Top-1 accuracy
    'best_acc5': best_acc5,                      # Best Top-5 accuracy
    'fold_idx': fold_idx,                        # Fold index
    'num_classes': num_classes                   # Số classes
}
```

### 10.3 Cách Resume Training hoạt động

```
Training Process:
┌─────────────────────────────────────────────┐
│ Epoch 0-10: Training                        │
│     ↓                                        │
│  Crash / Stop                                │
│     ↓                                        │
│  Save checkpoint at epoch 10                 │
│     ↓                                        │
│  Resume from checkpoint                      │
│     ↓                                        │
│  Load: model weights, optimizer state        │
│     ↓                                        │
│  Continue from epoch 11                      │
│     ↓                                        │
│  Train to completion                         │
└─────────────────────────────────────────────┘
```

### 10.4 Cách sử dụng

```bash
# Resume từ best model
python3 train.py --resume checkpoints/run_XXX/best_fold0.pth

# Resume từ checkpoint epoch cụ thể
python3 train.py --resume checkpoints/run_XXX/checkpoint_fold0_epoch15.pth
```

**Chức năng resume sẽ:**
1. Load model state dict
2. Load optimizer state dict
3. Khôi phục epoch, best_acc1, best_acc5
4. Tiếp tục training từ epoch + 1
5. Giữ nguyên best accuracy đã đạt được

### 10.5 TensorBoard Logging

**TensorBoard** là công cụ visualize training progress:

**Metrics được log:**
- **Loss:** Train và validation loss mỗi epoch
- **Accuracy:** Top-1 và Top-5 accuracy
- **Learning Rate:** LR hiện tại

**Cách sử dụng:**
```bash
# Start TensorBoard
tensorboard --logdir checkpoints/run_XXX/logs

# Mở browser
http://localhost:6006
```

### 10.6 Plot Training Metrics

**plot_metrics.py** vẽ 2 biểu đồ:

1. **Train Loss over Epochs**
   - Trục X: Epoch
   - Trục Y: Loss
   - Annotation: Final loss và Best loss

2. **Accuracy Comparison**
   - Trục X: Epoch
   - Trục Y: Accuracy (%)
   - 3 đường: Train, Validation, Test (nếu có)
   - Annotation: Best validation accuracy

**Cách sử dụng:**
```bash
python3 plot_metrics.py --log_dir checkpoints/run_XXX/logs/train --save curves
```

---

## 11. Early Stopping

### 11.1 Early Stopping là gì?

**Early Stopping** là kỹ thuật tự động dừng training khi model không còn cải thiện, giúp:
- Tránh overfitting
- Tiết kiệm thời gian training
- Giảm sử dụng tài nguyên

### 11.2 Cách hoạt động

```
┌─────────────────────────────────────────────────┐
│ Validation Accuracy                           │
│                                          ╱│
│                                       ╱    │
│                                    ╱       │  ↑
│                                ╱          │  │ Best
│                             ╱             │  │
│                          ╱                │  │
│  ___________╱_______________________       │  ↓
│                                        └───┤ Epoch
│                                           │
│                                           │
│  Early Stop patience=5                       │
│  (dừng sau 5 epochs không cải thiện)        │
└─────────────────────────────────────────────────┘
```

### 11.3 Config trong dự án

```yaml
training:
  early_stopping_patience: 5      # Số epoch chờ
  early_stopping_delta: 0.005     # Cải thiện tối thiểu
  mode: 'max'                    # Tối đa hóa accuracy
```

### 11.4 Logic

```python
# Từ train.py - EarlyStopping class
if score > best_score + min_delta:
    best_score = score
    counter = 0
else:
    counter += 1

if counter >= patience:
    early_stop = True
    break
```

**Quy tắc:**
- Nếu val_acc > best_acc + 0.005: Reset counter
- Ngược lại: Tăng counter
- Nếu counter >= 5: Dừng training

---

## Tài liệu tham khảo

1. **EfficientNetV2:** Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
2. **CNN Basics:** LeCun et al., "Gradient-based learning applied to document recognition", 1998
3. **Transfer Learning:** Yosinski et al., "How transferable are features in deep neural networks?", 2014
4. **ImageNet:** Deng et al., "ImageNet: A large-scale hierarchical image database", 2009
5. **ePillID:** Uddin et al., "ePillID: A Benchmark for Pill Identification using Deep Learning", 2021
6. **PyTorch Checkpointing:** PyTorch Documentation, "Saving and Loading Models"
7. **timm:** Ross Wightman, "PyTorch Image Models (timm)", GitHub

---

## Tóm tắt

Dự án sử dụng **EfficientNetV2-S** (từ thư viện **timm**) với **Transfer Learning** từ ImageNet để phân loại **960 loại thuốc** dựa trên hình ảnh. Mô hình được train với **Two-Phase strategy**, **Aggressive Data Augmentation**, **Label Smoothing**, và **Class Weights** để đạt được **Top-1 Accuracy ~23.46%** - một kết quả tốt cho bài toán khó với ít dữ liệu.

**Các tính năng kỹ thuật:**
- **Resume Training:** Tiếp tục training từ checkpoint khi bị gián đoạn
- **Early Stopping:** Tự động dừng khi không cải thiện
- **Gradient Clipping:** Tránh exploding gradient
- **Checkpoint Management:** Lưu best model và periodic checkpoints
- **TensorBoard Integration:** Visualize training progress
- **Plot Metrics:** Vẽ biểu đồ training curves

**Key takeaways:**
- CNN học features hierarchically (low → high level)
- Transfer learning cho phép train tốt với ít dữ liệu
- EfficientNetV2 balance giữa accuracy, speed, size
- NDC code là standard cho identification thuốc
- Data augmentation và regularization quan trọng cho small dataset
- Early stopping giúp tránh overfitting và tiết kiệm thời gian
