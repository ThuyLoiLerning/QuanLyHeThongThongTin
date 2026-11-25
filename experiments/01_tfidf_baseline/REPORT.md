# BÁO CÁO THỰC NGHIỆM: BASELINE 1 — TF-IDF + LOGISTIC REGRESSION

## 1. Bài báo / Phương pháp giải quyết vấn đề gì?

**Phương pháp:** Baseline học máy truyền thống (không Deep Learning) cho bài toán Entity Matching.

- Mỗi bản ghi thực thể (entity) được nối thành chuỗi từ các thuộc tính `title`, `manufacturer`, `price`.
- Với mỗi cặp (A, B), trích xuất hai vector TF-IDF bi-gram, sau đó tính:
  1. **Abs-diff** `|v_A - v_B|`: vector chênh lệch tuyệt đối.
  2. **Hadamard product** `v_A ⊙ v_B`: tương tác từng chiều.
  3. **Cosine similarity** `v_A · v_B`: độ tương đồng tổng thể.
  4. **rapidfuzz features** (ratio, token_sort_ratio, token_set_ratio, partial_ratio): bắt lỗi gõ sai và sắp xếp từ.
- **Logistic Regression** (`class_weight='balanced'`) phân loại match/non-match.
- **Decision threshold** hiệu chỉnh trên tập valid để tối đa F1.

## 2. Kết quả thực nghiệm (Experimental Results)

| Chỉ số | Kết quả tham chiếu (Magellan Baseline) | Kết quả thực tế | Chênh lệch Δ |
|---|---|---|---|
| **Precision** | ~78.0% | **59.16%** | -18.84% |
| **Recall** | ~65.0% | **78.63%** | +13.63% |
| **F1 Score** | ~70.5% | **67.52%** | -2.98% |
| **Runtime** | N/A | **0.50s** | Rất nhanh (CPU) |

**Ghi chú:** Kết quả tham chiếu là giá trị Magellan benchmark. Phương pháp sử dụng TF-IDF bi-gram (Siamese diff + product) + rapidfuzz fuzzy features, không dùng embedding ngữ nghĩa.

## 3. Ưu điểm

- **Tốc độ cực nhanh:** Toàn bộ pipeline (feature extraction + train + inference) chạy dưới 2 giây trên CPU.
- **Không cần GPU:** Phù hợp triển khai trong hệ thống sản xuất nhỏ.
- **Giải thích được:** Có thể xem trọng số từ ngữ của Logistic Regression để hiểu tại sao mô hình dự đoán match.
- **Bắt exact match tốt:** TF-IDF nhạy với trùng khớp từ chính xác trong title sản phẩm.

## 4. Khuyết điểm & Rào cản thực tế

- **Semantic blindness:** Không nhận ra "HP Pavilion" và "Hewlett Packard Pavilion" là cùng sản phẩm nếu không có từ chung.
- **Nhạy cảm với lỗi chính tả:** TF-IDF ngram không học được biến thể spelling (rapidfuzz bù một phần).
- **Không học được ngữ cảnh:** Hai từ xa nhau trong câu được xử lý độc lập nhau.
- **Phụ thuộc vào quality của thuộc tính:** Nếu `manufacturer` hay `price` bị thiếu, recall giảm đáng kể.

## 5. Bài học & Đề xuất nâng cấp

- **Blocking:** TF-IDF phù hợp dùng ở tầng blocking (lọc thô) giảm không gian O(N×M) → O(k), không phải tầng matching chính.
- **Tầng matching:** Cần Transformer (BERT, Sentence-BERT) để hiểu ngữ nghĩa sâu hơn — mục tiêu của Baseline 2 trở đi.
- **Hard constraints:** Thêm bộ lọc cứng (ví dụ: price ratio > 2× → auto reject) để giảm False Merge đặc biệt trong domain ERP-CRM.
- **Threshold calibration:** Luôn calibrate threshold trên valid set thay vì dùng 0.5 mặc định — cải thiện F1 đáng kể (từ 0.48 lên 0.66 trong thực nghiệm này).
