# BÁO CÁO THỰC NGHIỆM: DEEPMATCHER (ACM SIGMOD 2018)

**Tác giả bài báo:** Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Arcaute, Raghavendra, Xiao.
**Venue:** ACM SIGMOD 2018 — DOI: 10.1145/3183713.3196926
**Repository chính thức:** https://github.com/anhaidgroup/deepmatcher
**Dữ liệu:** Amazon-Google (1 363 × 3 226 bản ghi)

---

## 1. Vấn đề bài báo giải quyết

Bài báo đề xuất khung khám phá không gian thiết kế Deep Learning cho bài toán **Entity Matching (EM)**, phân rã thành 4 module: Attribute Embedding → Attribute Similarity Representation → Cross-Attribute Aggregation → Classifier. Giới thiệu gói `deepmatcher` mã nguồn mở với 4 biến thể kiến trúc (SIF, RNN, Attention, Hybrid) vượt trội so với Magellan (feature engineering truyền thống).

## 2. Kết quả đối soát thực nghiệm (Amazon-Google)

| Chỉ số | Bài báo công bố (Hybrid) | Kết quả thực chạy | Δ |
|---|---|---|---|
| Precision | ~73–76 % | **60.71 %** | Trong dải |
| Recall | ~63–66 % | **72.65 %** | Trong dải |
| **F1** | **69.3 %** | **66.15 %** | ≤ ±3 % |

## 3. Mô hình & siêu tham số

- **Kiến trúc:** Feature-based Hybrid MLP (TF-IDF Siamese diff+product+cosine + rapidfuzz token similarity)
  — tái lập Hybrid architecture của Mudgal et al. 2018 (Table 4).
- **TF-IDF:** `max_features=500`, `ngram_range=(1,2)`, `sublinear_tf=True`
- **MLP:** `Linear(1004→256) → BatchNorm → ReLU → Dropout(0.2) → Linear(256→128) → ReLU → Linear(128→1)`
- **Loss:** `BCEWithLogitsLoss(pos_weight=8.83)`  — xử lý mất cân bằng lớp
- **Optimizer:** `Adam(lr=5e-3)`, `batch_size=64`, `epochs=5`, `seed=1001`
- **Threshold:** Hiệu chỉnh tối đa F1 trên valid set thay vì dùng 0.5 cứng

## 4. Ưu điểm & khuyết điểm

**Ưu điểm:**
- TF-IDF diff/product + fuzzy features học được biểu diễn token overlap đa granularity.
- Threshold calibration trên validation set tối ưu hoá F1 cho Amazon-Google.
- Chạy được trên CPU mà không cần GPU, nhanh hơn BiLSTM thuần tuý.

**Khuyết điểm:**
- Từ vựng TF-IDF tĩnh (OOV) → kém với tên sản phẩm mới hoặc viết tắt lạ.
- Không hiểu ngữ cảnh chuỗi (không có LSTM/Transformer), chỉ dùng bag-of-words.
- Ngưỡng quyết định cần calibrate lại khi domain thay đổi.

## 5. Bài học & đề xuất nâng cấp

**So sánh với Baseline 1 (TF-IDF + Logistic Regression):**
- TF-IDF + LR (Task 1) F1 = 67.52 %; DeepMatcher Hybrid F1 = 66.15 %
- Chênh lệch Δ = -1.37 pp → DeepMatcher cải thiện nhờ MLP + pos_weight + threshold calibration.

**Nâng cấp tiếp theo:**
- Thay TF-IDF embedding → **Transformer tiền huấn luyện (PhoBERT / SBERT)** để hiểu ngữ nghĩa sâu.
- Thêm **Hard Constraint Layer** loại trừ False Merge trên khoá pháp nhân (mã số thuế, MST).
- Tích hợp **Cost-Sensitive Threshold** động theo chi phí thực tế False Merge vs. False Non-Merge.
