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
│   ├── classification_data/
│   │   └── fcn_mix_weight/dc_224/      # Hình ảnh (224x224)
│   └── folds/
│       └── pilltypeid_nih_sidelbls0.01_metric_5folds/
│           ├── label_encoder.pickle     # Label encoder
│           └── base/                    # 5-fold cross-validation files
│               ├── pilltypeid_..._0.csv
│               ├── pilltypeid_..._1.csv
│               └── ...
├── checkpoints/                         # Checkpoint saved models
│   └── best_fold0.pth                   # Best model checkpoint
├── templates/                           # HTML templates cho web app
│   └── index.html                       # Web UI
├── uploads/                             # Thư mục upload ảnh (tự tạo)
├── app.py                               # Flask Web Application
├── train.py                             # Script huấn luyện
├── inference.py                         # Script inference
├── config.yaml                          # File cấu hình
├── requirements.txt                     # Dependencies
└── README.md                            # File hướng dẫn này
```

---

## Thông tin Dataset

- **Tên Dataset:** ePillID
- **Số lớp:** 960 loại thuốc (NDC codes)
- **Số lượng ảnh:** ~3,728 hình ảnh
- **Kích thước ảnh:** 224 x 224 pixels
- **Chia dữ liệu:** 80% train / 20% validation (random split)

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

#### Các tham số command line

```bash
python train.py --config config.yaml --fold 0 --resume checkpoints/best_fold0.pth
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
  data_dir: "../ePillID_data"
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

## Kết quả huấn luyện

### Model: EfficientNetV2-S

| Metric        | Giá trị  |
|---------------|----------|
| Top-1 Acc     | ~23.46%  |
| Top-5 Acc     | ~45%     |
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
- Kiểm tra đường dẫn `data_dir` trong config.yaml
- Đảm bảo dataset đã được giải nén đúng vị trí

**4. Web app không mở được NIH Drug Portal**
- Kiểm tra kết nối internet
- Tắt popup blocker trong trình duyệt

### Mẹo tối ưu hóa

- Dùng GPU nếu có (huấn luyện nhanh hơn 10-20x)
- Batch size lớn hơn → training ổn định hơn nhưng cần nhiều VRAM
- Label smoothing giúp tránh overconfidence
- Class weights giúp cân bằng dataset imbalance

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
