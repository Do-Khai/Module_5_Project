# Dự án MLOps: Dự đoán giá nhà

Dự án này là một ví dụ hoàn chỉnh về quy trình MLOps, bao gồm quản lý phiên bản dữ liệu (DVC), xây dựng feature store (Feast), và huấn luyện mô hình dự đoán giá nhà.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu cài đặt](#yêu-cầu-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Bước 1: Thiết lập môi trường](#bước-1-thiết-lập-môi-trường)
  - [Bước 2: Tái tạo pipeline dữ liệu với DVC](#bước-2-tái-tạo-pipeline-dữ-liệu-với-dvc)
  - [Bước 3: Thiết lập Feature Store với Feast](#bước-3-thiết-lập-feature-store-với-feast)
  - [Bước 4: Huấn luyện mô hình](#bước-4-huấn-luyện-mô-hình)
- [Giải thích các thành phần](#giải-thích-các-thành-phần)
  - [DVC Pipeline](#dvc-pipeline)
  - [Feast Feature Store](#feast-feature-store)
  - [Training Scripts](#training-scripts)

## Tổng quan

Mục tiêu của dự án là dự đoán giá bán nhà (`SalePrice`) dựa trên các đặc điểm khác nhau của ngôi nhà. Dự án sử dụng bộ dữ liệu "House Prices - Advanced Regression Techniques" từ Kaggle.

Quy trình MLOps được áp dụng bao gồm:
1.  **Quản lý phiên bản dữ liệu thô** bằng DVC.
2.  **Tiền xử lý dữ liệu** để làm sạch và xử lý các giá trị thiếu, được quản lý như một stage trong DVC pipeline.
3.  **Xây dựng Feature Store** bằng Feast để quản lý và cung cấp features cho việc huấn luyện và dự đoán.
4.  **Huấn luyện và đánh giá** các mô hình Machine Learning, với các tham số được quản lý qua file `params.yaml`.

## Cấu trúc thư mục

```
house-prices-project/
├── data/
│   ├── raw/
│   │   └── train.csv      # Dữ liệu thô
│   └── processed/
│       └── processed_train.csv # Dữ liệu đã qua xử lý bởi DVC
├── feature_repo/          # Thư mục cho Feast Feature Store
│   ├── data/
│   │   └── house_features.parquet # Dữ liệu cho feature store
│   ├── __init__.py
│   ├── house_features.py    # Định nghĩa feature view
│   └── feature_store.yaml   # Cấu hình feature store
├── models/                  # Thư mục chứa model và metrics đã huấn luyện
│   ├── model_poly.joblib
│   └── metrics_poly.json
├── notebooks/
│   └── eda.ipynb            # Notebook khám phá dữ liệu ban đầu
├── src/
│   ├── __init__.py
│   ├── process_data.py      # Script tiền xử lý dữ liệu
│   ├── prepare_feast_data.py # Script chuẩn bị dữ liệu cho Feast
│   └── train_poly.py        # Script huấn luyện mô hình Polynomial Regression
├── dvc.yaml                 # Định nghĩa các stage của DVC pipeline
├── params.yaml              # Quản lý các tham số cho mô hình
├── README.md                # File hướng dẫn này
└── requirements.txt         # Các thư viện Python cần thiết
```

## Yêu cầu cài đặt

- Python 3.8+
- DVC
- Feast
- Scikit-learn, Pandas, Numpy, PyYAML

## Hướng dẫn sử dụng

### Bước 1: Thiết lập môi trường

1.  Clone repository về máy của bạn.
2.  Tạo và kích hoạt một môi trường ảo (virtual environment):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên macOS/Linux
    # venv\\Scripts\\activate    # Trên Windows
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

### Bước 2: Tái tạo pipeline dữ liệu với DVC

DVC được sử dụng để quản lý phiên bản dữ liệu và pipeline tiền xử lý.

1.  **Lấy dữ liệu thô**: Nếu bạn clone repo mà chưa có dữ liệu, hãy tải file `train.csv` từ cuộc thi Kaggle https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques và đặt vào thư mục `data/raw/`.

2.  **Chạy pipeline**: Lệnh này sẽ tự động chạy stage `process_data` được định nghĩa trong `dvc.yaml`. Nó sẽ thực thi `src/process_data.py` để tạo ra file `data/processed/processed_train.csv`.
    ```bash
    dvc repro
    ```

### Bước 3: Thiết lập Feature Store với Feast

1.  **Chuẩn bị dữ liệu cho Feast**: Chạy script để chuyển đổi dữ liệu đã xử lý sang định dạng Parquet mà Feast yêu cầu.
    ```bash
    python src/prepare_feast_data.py
    ```
2.  **Khởi tạo Feature Store**: Di chuyển vào thư mục `feature_repo` và chạy lệnh `apply` để đăng ký các feature view.
    ```bash
    cd feature_repo
    feast apply
    cd ..
    ```

### Bước 4: Huấn luyện mô hình

Sau khi pipeline dữ liệu và feature store đã sẵn sàng, bạn có thể huấn luyện mô hình.

1.  **Tùy chỉnh tham số (Tùy chọn)**: Mở file `params.yaml` và chỉnh sửa các tham số như `test_size` hoặc `degree` của `PolynomialFeatures` nếu muốn.

2.  **Chạy script training**:
    ```bash
    python src/train_poly.py
    ```
    Script này sẽ:
    - Đọc dữ liệu đã xử lý từ `data/processed/processed_train.csv`.
    - Đọc các tham số từ `params.yaml`.
    - Chia dữ liệu, tiền xử lý và huấn luyện mô hình Polynomial Regression.
    - Lưu model đã huấn luyện vào `models/model_poly.joblib`.
    - Lưu các chỉ số đánh giá (RMSE, R2 score) vào `models/metrics_poly.json`.

3.  **Tái tạo pipeline training với DVC (Nâng cao)**: Nếu bạn đã định nghĩa stage `train_poly` trong `dvc.yaml`, bạn có thể chạy:
    ```bash
    dvc repro train_poly
    ```
    DVC sẽ tự động chạy lại quá trình training nếu có bất kỳ thay đổi nào trong code (`src/train_poly.py`), dữ liệu (`data/processed/processed_train.csv`), hoặc tham số (`params.yaml`).

## Giải thích các thành phần

### DVC Pipeline (`dvc.yaml`)
- **`process_data`**: Stage này nhận file `data/raw/train.csv` làm đầu vào, chạy `src/process_data.py` để xử lý và cho ra `data/processed/processed_train.csv`.
- **`train_poly`**: Stage này nhận dữ liệu đã xử lý và file `params.yaml` làm đầu vào, chạy `src/train_poly.py` để huấn luyện mô hình, và cho ra model artifact (`models/model_poly.joblib`) cùng file metrics (`models/metrics_poly.json`).

### Feast Feature Store (`feature_repo/`)
- **`features.py`**: Định nghĩa `feature_view` cho các đặc trưng của ngôi nhà, với `house_id` là entity.
- **`feature_store.yaml`**: Cấu hình provider là `local` để lưu trữ dữ liệu feature store trên máy.

### Training Scripts (`src/`)
- **`process_data.py`**: Script thực hiện các bước làm sạch dữ liệu cơ bản.
- **`prepare_feast_data.py`**: Chuyển đổi file CSV thành Parquet và thêm cột timestamp để tương thích với Feast.
- **`train_poly.py`**: Script chính cho việc huấn luyện, sử dụng `scikit-learn Pipeline` để đóng gói các bước tiền xử lý và mô hình, đảm bảo tính nhất quán và tránh rò rỉ dữ liệu.
