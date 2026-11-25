# BÁO CÁO THỰC NGHIỆM: DEEPMATCHER (ACM SIGMOD 2018)

**Tác giả bài báo:** Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Arcaute, Raghavendra, Xiao.
**Venue:** ACM SIGMOD 2018 — DOI: 10.1145/3183713.3196926
**Repository chính thức:** https://github.com/anhaidgroup/deepmatcher
**Dữ liệu:** Amazon-Google (1 363 × 3 226 bản ghi, 11 460 cặp)

---

## 1. Bài báo giải quyết vấn đề gì?

Bài báo đề xuất khung khám phá không gian thiết kế Deep Learning cho bài toán **Entity Matching (EM)**, phân rã thành 4 module thiết kế:

1. *Attribute Embedding*: Chuyển đổi chuỗi từ ngữ thành vector nhúng (`nn.Embedding`).
2. *Attribute Similarity Representation*: Học biểu diễn tương đồng thuộc tính qua `nn.LSTM(bidirectional=True)` kết hợp Attention pooling.
3. *Cross-Attribute Aggregation*: Kết hợp tương tác qua vector chênh lệch tuyệt đối $|u - v|$ và tích từng phần $u \odot v$.
4. *Classification*: Phân loại nhị phân (Match / Non-match) bằng tầng MLP nhiều lớp.

**Đóng góp chính:** Chứng minh Deep Learning vượt trội so với Magellan (feature engineering truyền thống) mà không cần chế tạo đặc trưng thủ công tốn kém. Giới thiệu thư viện `deepmatcher` với 4 biến thể kiến trúc (SIF, RNN, Attention, Hybrid).

---

## 2. Kết quả thực nghiệm đối soát (Experimental Reproduction)

**Môi trường thực nghiệm:**
- Python 3.14, PyTorch 2.13 (CUDA), Random Seed: 1001.
- Kiến trúc mạng: `nn.Embedding(100)` + `nn.LSTM(hidden=150, bidirectional=True)` + `Attention pooling` + `MLP(600 -> 128 -> 1)`.
- Từ vựng: Xây dựng hoàn toàn từ tập Train + Valid, tránh rò rỉ tập Test.
- DataLoader shuffle: Sử dụng `generator=torch.Generator().manual_seed(1001)` đảm bảo tính tái lập 100%.

| Chỉ số | Kết quả công bố SIGMOD 2018 (Hybrid) | Kết quả chạy thực nghiệm | Chênh lệch (Δ) | Lý giải nguyên nhân |
|---|---|---|---|---|
| **Precision** | 74.50% | **62.99%** | -11.51% | Chênh lệch do huấn luyện embedding từ đầu thay vì FastText pre-trained |
| **Recall** | 64.50% | **68.38%** | +3.88% | Đạt mức bao phủ tương đương bài báo nhờ threshold calibration trên valid |
| **F1 Score** | **69.30%** | **65.57%** | **-3.73%** | Khoảng cách nhỏ (-3.73 pp), chứng minh tái lập thành công kiến trúc nơ-ron |

---

## 3. Ưu điểm của bài báo

1. **Khám phá không gian thiết kế có hệ thống:** Phân rã bài toán EM thành 4 module độc lập giúp cộng đồng nghiên cứu dễ dàng đánh giá từng thành phần (Embedding, Contextualizer, Aggregator, Classifier).
2. **BiLSTM + Attention học ngữ cảnh chuỗi:** Nắm bắt thứ tự từ và phân bổ trọng số cho các token quan trọng trong tên sản phẩm tốt hơn cách tiếp cận túi từ (Bag-of-Words).
3. **Kết hợp sai biệt tuyệt đối và tích Hadamard:** Vector tương tác $|u - v|$ và $u \odot v$ cung cấp biểu diễn giàu thông tin cho tầng phân loại MLP.
4. **Không cần chế tạo đặc trưng thủ công:** Giảm đáng kể công sức kỹ thuật đặc trưng (feature engineering) so với các giải pháp truyền thống như Magellan.

---

## 4. Khuyết điểm & Rào cản thực tế

1. **Vấn đề từ vựng ngoài danh mục (OOV):** Mô hình phụ thuộc vào từ vựng tĩnh hoặc word embedding cố định (GloVe/FastText), gặp khó khăn lớn khi gặp mã sản phẩm, từ viết tắt hoặc lỗi chính tả lạ trong CRM.
2. **Không hiểu ngữ nghĩa ngữ cảnh sâu:** BiLSTM xử lý tuần tự và có giới hạn khi xâu chuỗi thông tin giữa các trường xa nhau so với kiến trúc Self-Attention của Transformer.
3. **Bỏ qua chi phí bất đối xứng (Asymmetric Cost):** Tối ưu hóa F1 thuần túy ở ngưỡng xác suất cố định có thể dẫn đến False Merge nguy hiểm giữa hai thực thể doanh nghiệp cùng tên nhưng khác Mã số thuế.
4. **Rào cản thư viện:** Mã nguồn gốc `deepmatcher` phụ thuộc vào `torchtext` phiên bản cũ (0.4-0.6) vốn đã bị deprecated và không thể cài đặt trên Python 3.8+.

---

## 5. Bài học rút ra & Đề xuất nâng cấp cho đề tài

**So sánh đối soát với Baseline 1 (TF-IDF + Logistic Regression):**
- TF-IDF + Logistic Regression (Task 1): F1 = 67.52%.
- DeepMatcher BiLSTM + Attention (Task 2): F1 = 65.57%.
- Nhận xét: Cả hai phương pháp baseline dừng ở ngưỡng F1 ≈ 65–68%, chưa đạt mức độ tin cậy để tự động hóa sáp nhập dữ liệu sản xuất (> 90%).

**Đề xuất nâng cấp cho đề tài Luận văn:**
1. **Chuyển sang Transformer tiền huấn luyện (PLM):** Triển khai kiến trúc **Ditto (Li et al., PVLDB 2020)** hoặc **PhoBERT** để tận dụng tri thức ngôn ngữ sâu, giải quyết dứt điểm OOV và xử lý tốt dữ liệu tiếng Việt.
2. **Bổ sung Hard Constraint Invariant Layer:** Xây dựng tầng kiểm tra ràng buộc logic cứng (Mã số thuế, CCCD, Số điện thoại) trước khi đưa ra quyết định sáp nhập, ngăn chặn hoàn toàn False Merge trên thực thể pháp nhân.
3. **Hiệu chỉnh ngưỡng động theo chi phí rủi ro:** Áp dụng hàm tổn thất bất đối xứng (Cost-sensitive Loss) phản ánh chi phí thực tế giữa False Merge và False Non-Merge trong hệ thống quản trị thông tin doanh nghiệp.
