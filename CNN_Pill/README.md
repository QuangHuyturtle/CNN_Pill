# Pill Image Classification with EfficientNetV2

Phân loại viên thuốc từ hình ảnh sử dụng EfficientNetV2-S với Transfer Learning, kèm giao diện Web Demo.

## Tổng quan dự án

Đây là dự án phân loại viên thuốc sử dụng mạng CNN (EfficientNetV2-S) được huấn luyện trên dataset ePillID. Mô hình có khả năng phân loại **960 loại thuốc** khác nhau với độ chính xác Top-1 khoảng **23.46%**.

### Tính năng chính

- Phân loại 960 loại thuốc dựa trên hình ảnh
- Giao diện Web Demo trực quan với drag-and-drop upload
- Tra cứu thông tin thuốc qua NDC Code trên NIH Drug Database
- Hỗ trợ batch inference cho nhiều hình ảnh
- Visualization kết quả dự đoán
- **Resume training từ checkpoint** - Tiếp tục huấn luyện khi bị gián đoạn
- **TensorFlow warning suppression** - Tắt thông báo không cần thiết

### Cấu trúc dự án

```
CNN_Pill/
├── models/                              # Code mô hình
│   ├── __init__.py
│   └── efficientnet_pill.py             # Định nghĩa mô hình EfficientNetV2
├── utils/                               # Code xử lý dữ liệu
│   ├── __init__.py
│   └── dataset.py                       # Dataset class và DataLoader
├── data/                                # Dữ liệu dataset
│   ├── all_labels.csv                   # Metadata toàn bộ dataset
│   └── ePillID_data/                    # Đường dẫn dữ liệu đúng
│       └── classification_data/
│           └── fcn_mix_weight/dc_224/  # Hình ảnh (224x224)
├── data/folds/                          # Fold cross-validation
│   └── pilltypeid_nih_sidelbls0.01_metric_5folds/
│       ├── label_encoder.pickle         # Label encoder
│       └── base/                        # 5-fold cross-validation files
│           ├── pilltypeid_..._0.csv
│           ├── pilltypeid_..._1.csv
│           └── ...
├── checkpoints/                         # Checkpoint saved models
│   └── run_YYYYMMDD_HHMMSS/            # Run directory với timestamp
│       ├── best_fold0.pth               # Best model checkpoint
│       ├── checkpoint_fold0_epochX.pth  # Periodic checkpoints
│       ├── config.yaml                 # Saved config
│       └── logs/                       # TensorBoard logs
├── templates/                           # HTML templates cho web app
│   └── index.html                       # Web UI
├── uploads/                             # Thư mục upload ảnh (tự tạo)
├── .gitignore                           # Git ignore file
├── app.py                               # Flask Web Application
├── train.py                             # Script huấn luyện (với resume)
├── inference.py                         # Script inference
├── config.yaml                          # File cấu hình
├── requirements.txt                     # Dependencies
├── README.md                            # File hướng dẫn này
└── THEORY.md                            # Tài liệu lý thuyết chi tiết
```

---

## Thông tin Dataset

- **Tên Dataset:** ePillID
- **Số lớp:** 960 loại thuốc (NDC codes)
- **Số lượng ảnh:** ~3,728 hình ảnh
- **Kích thước ảnh:** 224 x 224 pixels
- **Chia dữ liệu:** 80% train / 20% validation (random split)
- **Đường dẫn ảnh:** `ePillID_data/classification_data/fcn_mix_weight/dc_224/`

### NDC Code Format

Mỗi loại thuốc được đại diện bởi một **NDC Code** (National Drug Code) với format:

```
Labeler (5 digits) - Product (4 digits) - Package (2 digits) _ Hash
Ví dụ: 00591-5307-01_BE305F72
```

- **Labeler:** Mã nhà sản xuất
- **Product:** Mã sản phẩm
- **Package:** Mã đóng gói
- **Hash:** Identifier duy nhất

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.0+ (khuyến nghị nếu có GPU)
- 8GB+ RAM
- 4GB+ VRAM (nếu dùng GPU)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**
- PyTorch 2.0+ & torchvision
- timm (PyTorch Image Models)
- Flask (Web Framework)
- OpenCV, Pillow (Xử lý ảnh)
- pandas, numpy, scikit-learn

---

## Sử dụng

### 1. Web Application (Được khuyến nghị cho Demo)

#### Khởi động web app

```bash
cd CNN_Pill
python app.py
```

Web app sẽ chạy tại: **http://localhost:5000**

#### Tính năng Web App

1. **Upload ảnh:** Kéo thả hoặc click để chọn ảnh thuốc
2. **Phân loại:** Click nút "Phân Loại" để dự đoán
3. **Xem kết quả:**
   - Top 5 dự đoán với độ confidence
   - NDC Code được phân tích thành Labeler/Product/Package
   - Nút tra cứu trên NIH Drug Database
4. **Tra cứu thuốc:** Click vào nút "🔍 Tra cứu trên NIH Drug Database" để xem thông tin chi tiết:
   - Tên generic thuốc
   - Liều lượng
   - Dạng bào chế
   - Nhà sản xuất
   - Thành phần hoạt tính

#### Hạn chế

- Chỉ chạy local (localhost), không deploy online
- File upload tối đa 16MB
- Hỗ trợ định dạng: PNG, JPG, JPEG, BMP, GIF, TIF, WEBP

---

### 2. Huấn luyện mô hình

#### Chạy với config mặc định

```bash
cd CNN_Pill
python train.py
```

#### Huấn luyện với config tùy chỉnh

```bash
python train.py --config my_config.yaml
```

#### Resume training từ checkpoint (TÍNH NĂNG MỚI)

Khi training bị gián đoạn (mất điện, crash, etc.), bạn có thể tiếp tục từ checkpoint đã lưu:

```bash
# Resume từ best model
python train.py --resume checkpoints/run_20260125_065908/best_fold0.pth

# Resume từ checkpoint epoch cụ thể
python train.py --resume checkpoints/run_20260125_065908/checkpoint_fold0_epoch15.pth
```

**Chức năng resume sẽ:**
- Khôi phục model weights
- Khôi phục optimizer state
- Tiếp tục training từ epoch tiếp theo
- Giữ nguyên best accuracy đã đạt được

#### Các tham số command line

```bash
python train.py [OPTIONS]

Options:
  --config PATH    Path to config file (default: config.yaml)
  --data_dir PATH  Path to dataset directory (default: data)
  --save_dir PATH  Path to save checkpoints (default: checkpoints)
  --resume PATH    Path to checkpoint to resume from (optional)
```

---

### 3. File cấu hình (config.yaml)

```yaml
model:
  # Model size: 's' (small), 'm' (medium), 'l' (large)
  size: 's'
  pretrained: true
  dropout: 0.3
  freeze_backbone: true

data:
  data_dir: "data"
  batch_size: 32
  num_workers: 0           # Set = 0 cho Windows
  augmentation: true

training:
  num_epochs: 50
  learning_rate: 0.001
  optimizer: 'adamw'
  scheduler: 'cosine'
  backbone_lr_multiplier: 0.1
  warmup_epochs: 5
  unfreeze_epoch: 10
  use_class_weights: true
  label_smoothing: 0.1
  gradient_clip: 1.0
```

---

### 4. Dự đoán (Inference)

#### Dự đoán một hình ảnh

```bash
python inference.py \
    --checkpoint checkpoints/best_fold0.pth \
    --encoder data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle \
    --image path/to/pill.jpg
```

#### Dự đoán hàng loạt (batch)

```bash
python inference.py \
    --checkpoint checkpoints/best_fold0.pth \
    --directory path/to/images/ \
    --output results.csv \
    --top_k 5
```

#### Tạo visualization

```bash
python inference.py \
    --checkpoint checkpoints/best_fold0.pth \
    --image pill.jpg \
    --visualize \
    --output pill_pred.jpg
```

---

## Kiến trúc mô hình

```
Input Image (3, 224, 224)
    ↓
EfficientNetV2-S Backbone (pretrained on ImageNet)
    ├── Convolutional Blocks
    ├── MBConv Blocks
    └── Feature Extraction
    ↓
Global Average Pooling
    ↓
Classifier Head:
    ├── Dropout (0.3)
    ├── Linear (features → 512)
    ├── BatchNorm1d + ReLU
    ├── Dropout (0.15)
    └── Linear (512 → 960)
    ↓
Output (960 classes - NDC Codes)
```

### Chiến lược huấn luyện (Two-Phase Training)

**Phase 1 (Epoch 0-9):**
- Freeze backbone
- Chỉ train classifier head
- Learning rate: 0.001

**Phase 2 (Epoch 10+):**
- Unfreeze toàn bộ mô hình
- Backbone LR: 0.0001 (1/10 của head LR)
- Fine-tune toàn bộ mạng

---

## Data Augmentation

Các kỹ thuật augmentation được áp dụng cho training set:

- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Random Translation (±10%)
- Color Jitter (brightness ±0.2, contrast ±0.2, saturation ±0.2)
- Normalize (ImageNet stats)

---

## Theo dõi huấn luyện

### TensorBoard

```bash
tensorboard --logdir checkpoints/run_XXX/logs
```

Mở trình duyệt tại: http://localhost:6006

### Metrics được theo dõi

- **Top-1 Accuracy:** % dự đoán đúng lớp cao nhất
- **Top-5 Accuracy:** % lớp đúng nằm trong top 5
- **Loss:** Cross-Entropy loss với label smoothing
- **Learning Rate:** Theo dõi thay đổi LR qua các epoch

---

## Quản lý checkpoints

### Cấu trúc checkpoint directory

Mỗi lần chạy training sẽ tạo một thư mục mới với timestamp:

```
checkpoints/
└── run_20260125_065908/
    ├── best_fold0.pth              # Best model (highest val acc)
    ├── checkpoint_fold0_epoch10.pth # Checkpoint mỗi 10 epoch
    ├── checkpoint_fold0_epoch20.pth
    ├── config.yaml                 # Config được sử dụng
    └── logs/
        └── train/                  # TensorBoard logs
```

### Xóa checkpoints cũ

```bash
# Xóa tất cả checkpoints
rm -rf checkpoints/*

# Xóa run cụ thể
rm -rf checkpoints/run_20260124_163841
```

**Lưu ý:** File `.gitignore` đã được cấu hình để không commit:
- `checkpoints/` - Model files (rất nặng)
- `logs/` - TensorBoard logs
- `data/` - Dataset

---

## Kết quả huấn luyện

### Model: EfficientNetV2-S

| Metric        | Giá trị  |
|---------------|----------|
| Top-1 Acc     | ~23.46%  |
| Top-5 Acc     | ~43%     |
| Num Classes   | 960      |
| Train Samples | ~2,982   |
| Val Samples   | ~746     |

**Lưu ý:** Đây là bài toán phân loại 960 lớp với mỗi lớp chỉ ~3-4 mẫu, nên accuracy thấp là bình thường.

---

## Troubleshooting

### Lỗi thường gặp

**1. RuntimeError: DataLoader worker died**
```yaml
# Giải pháp: Set num_workers = 0 trong config.yaml
data:
  num_workers: 0
```

**2. CUDA out of memory**
```yaml
# Giải pháp: Giảm batch_size
data:
  batch_size: 16  # hoặc 8
```

**3. FileNotFoundError: Image not found**
- Kiểm tra đường dẫn dữ liệu: `ePillID_data/classification_data/fcn_mix_weight/dc_224/`
- Đảm bảo dataset đã được giải nén đúng vị trí

**4. TensorFlow warnings khi training**
- Đã tự động tắt trong train.py
- Nếu vẫn thấy, set environment variable:
  ```bash
  export TF_ENABLE_ONEDNN_OPTS=0
  export TF_CPP_MIN_LOG_LEVEL=3
  ```

**5. Web app không mở được NIH Drug Portal**
- Kiểm tra kết nối internet
- Tắt popup blocker trong trình duyệt

### Mẹo tối ưu hóa

- Dùng GPU nếu có (huấn luyện nhanh hơn 10-20x)
- Batch size lớn hơn → training ổn định hơn nhưng cần nhiều VRAM
- Label smoothing giúp tránh overconfidence
- Class weights giúp cân bằng dataset imbalance
- Sử dụng `--resume` để tiếp tục training khi bị gián đoạn

---

## Các tính năng kỹ thuật

### Resume Training Implementation

```python
# Sử dụng trong train.py
python train.py --resume checkpoints/run_xxx/best_fold0.pth

# Code tự động:
# 1. Load model state dict
# 2. Load optimizer state dict
# 3. Khôi phục epoch, best_acc1, best_acc5
# 4. Tiếp tục training từ epoch + 1
```

### TensorFlow Warning Suppression

```python
# Đã thêm vào train.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### Git Ignore Setup

File `.gitignore` đã được cấu hình để bỏ qua:
- Checkpoints (`.pth` files)
- TensorBoard logs
- Dữ liệu (`data/`)
- Python cache (`__pycache__/`)

---

## Tài liệu tham khảo

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [ePillID Dataset](https://github.com/MedICL-Lab/ePillID-pytorch)
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models)
- [NIH Drug Portal](https://druginfo.nlm.nih.gov/drugportal/)

---

## License

Dự án này được tạo ra cho mục đích học tập và nghiên cứu. Dataset ePillID thuộc về các tác giả tương ứng.

---

## Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub repository.
