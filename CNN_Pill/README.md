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
- **Early Stopping** - Tự động dừng training khi không cải thiện
- **TensorBoard Integration** - Visualize training progress
- **Plot Training Metrics** - Vẽ biểu đồ training curves

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
│   └── classification_data/              # Đường dẫn dữ liệu đúng
│       └── fcn_mix_weight/
│           ├── dc_224/                 # Hình ảnh cropped (224x224)
│           └── dr_224/                 # Hình ảnh reflected (224x224)
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
├── plot_metrics.py                      # Script vẽ biểu đồ training metrics
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
- **Chia dữ liệu:** 70% train / 15% validation / 15% test (random split)
- **Đường dẫn ảnh:** `classification_data/fcn_mix_weight/dc_224/` và `dr_224/`

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
- **GPU (một trong các loại sau):**
  - CUDA 11.0+ (cho NVIDIA GPU, khuyến nghị)
  - MPS (cho Apple Silicon M1/M2/M3)
  - Hoặc CPU (sẽ chậm hơn)
- 8GB+ RAM
- 4GB+ VRAM (nếu dùng GPU NVIDIA) / Unified Memory (Apple Silicon)

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
python3 train.py
```

#### Huấn luyện với config tùy chỉnh

```bash
python3 train.py --config my_config.yaml
```

#### Resume training từ checkpoint

Khi training bị gián đoạn (mất điện, crash, etc.), bạn có thể tiếp tục từ checkpoint đã lưu:

```bash
# Resume từ best model
python3 train.py --resume checkpoints/run_20260308_083313/best_fold0.pth

# Resume từ checkpoint epoch cụ thể
python3 train.py --resume checkpoints/run_20260308_083313/checkpoint_fold0_epoch15.pth
```

**Chức năng resume sẽ:**
- Khôi phục model weights
- Khôi phục optimizer state
- Tiếp tục training từ epoch tiếp theo
- Giữ nguyên best accuracy đã đạt được

#### Các tham số command line

```bash
python3 train.py [OPTIONS]

Options:
  --config PATH    Path to config file (default: config.yaml)
  --data_dir PATH  Path to dataset directory (default: data)
  --save_dir PATH  Path to save checkpoints (default: checkpoints)
  --resume PATH    Path to checkpoint to resume from (optional)
  --seed INT       Random seed for training (default: 42)
```

---

### 3. File cấu hình (config.yaml)

```yaml
model:
  # Model size: 's' (small), 'm' (medium), 'l' (large)
  size: 's'
  pretrained: true
  dropout: 0.5                # Dropout probability cho lớp đầu tiên
  dropout_head_extra: 0.25     # Dropout probability cho lớp thứ hai
  freeze_backbone: true         # Freeze backbone trong giai đoạn đầu
  weight_decay: 0.01

data:
  data_dir: "data"
  batch_size: 8                # Batch size (giảm nếu OOM)
  num_workers: 1               # Number of DataLoader workers (0 cho Windows)
  augmentation: true

training:
  num_epochs: 25               # Số epoch tối đa
  learning_rate: 0.002         # Learning rate ban đầu
  optimizer: 'adamw'          # Optimizer type
  scheduler: 'cosine'         # Learning rate scheduler
  backbone_lr_multiplier: 0.001 # LR multiplier cho backbone
  warmup_epochs: 3            # Warmup epochs
  unfreeze_epoch: 999          # Epoch để unfreeze backbone (999 = không unfreeze)
  use_class_weights: true       # Sử dụng class weights cho imbalanced dataset
  label_smoothing: 0.1         # Label smoothing factor
  max_grad_norm: 0.5           # Gradient clipping max norm
  early_stopping_patience: 5    # Số epoch chờ trước khi dừng
  early_stopping_delta: 0.005   # Cải thiện tối thiểu để reset counter
```

---

### 4. Dự đoán (Inference)

#### Dự đoán một hình ảnh

```bash
python3 inference.py \
    --checkpoint checkpoints/run_XXX/best_fold0.pth \
    --encoder data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle \
    --image path/to/pill.jpg
```

#### Dự đoán hàng loạt (batch)

```bash
python3 inference.py \
    --checkpoint checkpoints/run_XXX/best_fold0.pth \
    --directory path/to/images/ \
    --output results.csv \
    --top_k 5
```

#### Tạo visualization

```bash
python3 inference.py \
    --checkpoint checkpoints/run_XXX/best_fold0.pth \
    --image pill.jpg \
    --visualize \
    --output pill_pred.jpg
```

---

## Kiến trúc mô hình

```
Input Image (3, 224, 224)
    ↓
EfficientNetV2-S Backbone (pretrained on ImageNet, từ timm)
    ├── Convolutional Blocks
    ├── MBConv Blocks
    └── Feature Extraction
    Output: features (1280)
    ↓
Global Average Pooling
    ↓
Classifier Head:
    ├── Dropout (0.5)
    ├── Linear (1280 → 512)
    ├── BatchNorm1d + ReLU
    ├── Dropout (0.25)
    └── Linear (512 → num_classes)
    ↓
Output (960 classes - NDC Codes)
```

### Chiến lược huấn luyện

**Phase 1 (Epoch 0 - unfreeze_epoch-1):**
- Freeze backbone
- Chỉ train classifier head
- Learning rate: 0.002
- Mục đích: Head học cách phân loại với features có sẵn

**Phase 2 (Epoch unfreeze_epoch+):**
- Unfreeze toàn bộ mô hình (nếu unfreeze_epoch < num_epochs)
- Backbone LR: 0.002 × 0.001 = 0.000002
- Fine-tune toàn bộ mạng

**Early Stopping:**
- Theo dõi validation accuracy
- Nếu không cải thiện sau `early_stopping_patience` epochs
- → Tự động dừng training

---

## Data Augmentation

Các kỹ thuật augmentation được áp dụng cho training set:

- **Random Horizontal Flip** (p=0.5)
- **Random Vertical Flip** (p=0.3)
- **Random Rotation** (±30°)
- **Random Affine**:
  - Translate: ±15%
  - Scale: 0.85-1.15
  - Shear: 5°
- **Random Resized Crop** (scale 0.7-1.0)
- **Color Jitter** (brightness ±0.3, contrast ±0.3, saturation ±0.3, hue ±0.1)
- **Random Grayscale** (p=0.1)
- **Gaussian Blur** (kernel_size=3, sigma 0.1-2.0)
- **Normalize** (ImageNet stats)

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

### Vẽ biểu đồ Training Metrics

Sau khi training xong, bạn có thể vẽ biểu đồ để xem quá trình training:

```bash
# Vẽ 2 biểu đồ và lưu vào file
python3 plot_metrics.py --log_dir checkpoints/run_XXX/logs/train --save training_curves

# Chỉ lưu file, không hiển thị trên màn hình
python3 plot_metrics.py --log_dir checkpoints/run_XXX/logs/train --save curves --no_show

# Chỉnh sửa fold index
python3 plot_metrics.py --log_dir checkpoints/run_XXX/logs/train --fold 0 --save curves
```

**Các biểu đồ được tạo:**
- **Biểu đồ Loss:** Train Loss over Epochs
- **Biểu đồ Accuracy:** Train vs Validation Accuracy comparison

---

## Quản lý checkpoints

### Cấu trúc checkpoint directory

```
checkpoints/
└── run_20260308_083313/
    ├── best_fold0.pth              # Best model (highest val acc)
    ├── checkpoint_fold0_epoch10.pth # Checkpoint mỗi 10 epoch
    ├── checkpoint_fold0_epoch20.pth
    ├── config.yaml                 # Config được sử dụng
    └── logs/
        └── train/                  # TensorBoard logs
```

---

## Kết quả huấn luyện

### Model: EfficientNetV2-S

| Metric        | Giá trị  |
|---------------|----------|
| Top-1 Acc     | ~23.46%  |
| Top-5 Acc     | ~43%     |
| Num Classes   | 960      |
| Train Samples | ~2,610   |
| Val Samples   | ~559     |
| Test Samples  | ~559     |

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

**2. CUDA/MPS out of memory**
```yaml
# Giải pháp: Giảm batch_size
data:
  batch_size: 4  # hoặc 2
```

**3. FileNotFoundError: Image not found**
- Kiểm tra đường dẫn dữ liệu: `classification_data/fcn_mix_weight/dc_224/`
- Đảm bảo dataset đã được giải nén đúng vị trí
- Kiểm tra file `all_labels.csv` có tồn tại trong `data/`

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
- Early stopping giúp tránh overfitting và tiết kiệm thời gian
- Gradient clipping giúp training ổn định hơn
- Sử dụng `--resume` để tiếp tục training khi bị gián đoạn

---

## Tài liệu tham khảo

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [ePillID Dataset](https://github.com/MedICL-Lab/ePillID-pytorch)
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models)
- [NIH Drug Portal](https://druginfo.nlm.nih.gov/drugportal/)

---

## License

Dự án này được tạo ra cho mục đích học tập và nghiên cứu. Dataset ePillID thuộc về các tác giả tương ứng.
