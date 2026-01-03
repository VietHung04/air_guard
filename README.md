# Air Quality Timeseries — PM2.5 Forecasting & AQI Alerts (Supervised + Semi‑Supervised)

Mini-project “end‑to‑end pipeline” trên bộ **Beijing Multi‑Site Air Quality (12 stations)** nhằm xây dựng:
1) **Dự báo PM2.5** (regression + ARIMA)  
2) **Phân lớp AQI (AQI level/class)** để **cảnh báo theo trạm**  
3) **Bán giám sát (Semi‑Supervised Learning)** để cải thiện khi **thiếu nhãn AQI / nhãn không chuẩn** (Self‑Training → Co‑Training)

Thiết kế theo triết lý:
- **OOP**: thư viện trong `src/` (train/eval/feature engineering).
- **Notebook‑per‑task**: mỗi notebook làm 1 nhiệm vụ rõ ràng.
- **Papermill**: chạy pipeline tự động bằng `run_papermill.py`.

---

## 1) Dataset

- Nguồn: **Beijing Multi‑Site Air Quality** (12 stations, dữ liệu theo giờ).
- Repo hỗ trợ 2 cách nạp dữ liệu trong notebook `preprocessing_and_eda.ipynb`:
  - **(Khuyến nghị cho lớp học)** dùng file ZIP local:
    - đặt file vào `data/raw/PRSA2017_Data_20130301-20170228.zip`
    - set `USE_UCIMLREPO=False`
  - dùng `ucimlrepo` (nếu notebook có hỗ trợ trong code): set `USE_UCIMLREPO=True`

> Lưu ý “leakage”: **không dùng trực tiếp `PM2.5` / `pm25_24h` trong feature đầu vào cho mô hình phân lớp AQI**.

---

## 2) Cấu trúc thư mục

```
air_quality_timeseries_with_semi/
├─ data/
│  ├─ raw/                # ZIP dữ liệu gốc
│  └─ processed/          # parquet + metrics + predictions + alerts
├─ notebooks/
│  ├─ preprocessing_and_eda.ipynb
│  ├─ feature_preparation.ipynb
│  ├─ classification_modelling.ipynb
│  ├─ regression_modelling.ipynb
│  ├─ arima_forecasting.ipynb
│  ├─ semi_dataset_preparation.ipynb          
│  ├─ semi_self_training.ipynb                
│  ├─ semi_co_training.ipynb                  
│  ├─ semi_supervised_report.ipynb            
│  └─ runs/                                   # output notebooks khi chạy papermill
├─ src/
│  ├─ classification_library.py
│  ├─ regression_library.py
│  ├─ timeseries_library.py
│  └─ semi_supervised_library.py              
├─ run_papermill.py
├─ requirements.txt
└─ README.md
```

---

## 3) Cài đặt môi trường

### 3.1 Tạo môi trường (Conda) và kernel cho Papermill
Repo mặc định chạy papermill với kernel tên **`beijing_env`** (xem `run_papermill.py`).

```bash
conda create -n beijing_env python=3.11 -y
conda activate beijing_env
pip install -r requirements.txt

# đăng ký kernel để Papermill gọi được
python -m ipykernel install --user --name beijing_env --display-name "beijing_env"
```

### 3.2 Kiểm tra nhanh
```bash
python -c "import pandas, sklearn, papermill; print('OK')"
```

---

## 4) Chạy pipeline (Papermill)

Chạy toàn bộ pipeline:

```bash
python run_papermill.py
```

Kết quả:
- Notebook chạy xong sẽ nằm ở `notebooks/runs/*_run.ipynb`
- Artefacts nằm ở `data/processed/` (metrics, predictions, alerts, parquet)

---

## 5) Mô tả pipeline notebooks (Notebook‑per‑task)

| Thứ tự | Notebook | Mục tiêu | Output chính |
|---:|---|---|---|
| 01 | `preprocessing_and_eda.ipynb` | đọc dữ liệu, làm sạch, tạo time features cơ bản | `data/processed/cleaned.parquet` |
| 02 | `semi_dataset_preparation.ipynb` | **giữ dữ liệu chưa nhãn + giả lập thiếu nhãn (train‑only)** | `data/processed/dataset_for_semi.parquet` |
| 03 | `feature_preparation.ipynb` | tạo dataset supervised cho phân lớp | `data/processed/dataset_for_clf.parquet` |
| 04 | `semi_self_training.ipynb` | **Self‑Training** cho AQI classification | `metrics_self_training.json`, `alerts_self_training_sample.csv` |
| 05 | `semi_co_training.ipynb` | **Co‑Training (2 views)** cho AQI classification | `metrics_co_training.json`, `alerts_co_training_sample.csv` |
| 06 | `classification_modelling.ipynb` | baseline supervised classification | `metrics.json`, `predictions_sample.csv` |
| 07 | `regression_modelling.ipynb` | dự báo PM2.5 (regression) | `regression_metrics.json`, `regressor.joblib` |
| 08 | `arima_forecasting.ipynb` | ARIMA forecasting cho 1 trạm | `arima_pm25_*` |
| 09 | `semi_supervised_report.ipynb` | **Storytelling report**: so sánh baseline vs semi + alert theo trạm | notebook report chạy trong `notebooks/runs/` |

---

## 6) Thư viện OOP (src/)

### 6.1 `src/classification_library.py`
- `time_split(df, cutoff)`: chia train/test theo thời gian
- `train_classifier(train_df, test_df, target_col='aqi_class')` → trả về `{model, metrics, pred_df}`
- Guard leakage: loại cột như `PM2.5`, `pm25_24h`, `datetime` khỏi features.

### 6.2 `src/semi_supervised_library.py` 
- `mask_labels_time_aware(...)`: giả lập thiếu nhãn **chỉ trong TRAIN**
- `SelfTrainingAQIClassifier`: vòng lặp pseudo‑label theo ngưỡng `tau`
- `CoTrainingAQIClassifier`: co‑training 2 views + late‑fusion
- `add_alert_columns(...)`: tạo `is_alert` theo ngưỡng mức AQI (vd từ `"Unhealthy"`)

---

## 7) MINI PROJECT: Semi‑Supervised AQI + Alerts theo trạm

### 7.1 Mục tiêu
Xây dựng hệ thống:
- dự đoán `aqi_class` cho từng timestamp/trạm
- sinh **cảnh báo** theo trạm (`is_alert`)
- khi **thiếu nhãn AQI** (hoặc nhãn không chuẩn), dùng **Self‑Training** và **Co‑Training** để cải thiện chất lượng.

### 7.2 Thiết kế thí nghiệm (bắt buộc)
1) **Baseline supervised**  
   - Chạy `classification_modelling.ipynb`  
   - Lấy `accuracy`, `f1_macro` từ `data/processed/metrics.json`

2) **Giả lập thiếu nhãn (train‑only)**  
   - Chạy `semi_dataset_preparation.ipynb` với:
     - `LABEL_MISSING_FRACTION ∈ {0.7, 0.9, 0.95, 0.98}`

3) **Self‑Training**  
   - Chạy `semi_self_training.ipynb` với:
     - `TAU ∈ {0.8, 0.9, 0.95}`
   - Phân tích: vòng lặp nào bắt đầu “bão hoà”, số pseudo‑labels tăng/giảm ra sao.

4) **Co‑Training**  
   - Chạy `semi_co_training.ipynb` với `TAU` giống Self‑Training
   - Bắt buộc thử 2 chế độ:
     - **Auto split views** (để `VIEW1_COLS=None`, `VIEW2_COLS=None`)
     - **Manual views**: tự thiết kế 2 views và giải thích vì sao hợp lý.


## 8) Chạy nhanh từng notebook (không dùng Papermill)
Bạn có thể mở Jupyter và chạy tuần tự từng notebook theo thứ tự ở mục (5).

---

## 9) Author
Project được thực hiện bởi:
Trang Le

## 10) License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.
