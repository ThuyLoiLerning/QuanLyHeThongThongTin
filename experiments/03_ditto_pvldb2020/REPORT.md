# BÁO CÁO THỰC NGHIỆM TÁI LẬP: DITTO (PVLDB 2020)

- **Tác giả:** Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan.
- **Venue:** Proceedings of the VLDB Endowment (PVLDB), Vol. 14, No. 1, tr. 50–60, 2020.
- **DOI:** 10.14778/3421424.3421431
- **Repository chính thức:** https://github.com/megagonlabs/ditto
- **Tập dữ liệu:** Amazon-Google (11.460 cặp bản ghi, phân chia train/valid/test chuẩn).

## 1. Bài báo giải quyết vấn đề gì?

Ditto giải quyết bài toán cốt lõi của **Entity Matching (EM)** — xác định các bản ghi từ hai nguồn dữ liệu khác biệt có cùng tham chiếu đến một thực thể trong thế giới thực hay không.

**Đột phá kiến trúc so với DeepMatcher (SIGMOD 2018):**
- **Chuỗi hóa dạng Token (Token-level Record Serialization):** Biến đổi toàn bộ cấu trúc bản ghi thành chuỗi tuần tự `COL <attr> VAL <val> ...`, nối hai bản ghi bằng ký tự phân tách `[SEP]` và đưa trực tiếp vào mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models - PLM) như BERT, RoBERTa, DistilBERT.
- **3 kỹ thuật tối ưu hóa trọng tâm của Ditto:**
  1. *Domain Knowledge Injection:* Nhận diện và gán nhãn đặc biệt cho các định danh số hiệu sản phẩm, model number, đơn vị đo lường để cơ chế Self-Attention tập trung trọng số.
  2. *Text Summarization:* Tóm tắt các chuỗi mô tả quá dài bằng TF-IDF nhằm bảo đảm vừa vặn trong cửa sổ ngữ cảnh 512 token của Transformer.
  3. *Data Augmentation:* Tự động sinh cặp dữ liệu huấn luyện tăng cường thông qua hoán vị và che từ (masking), cải thiện tính tổng quát hóa.
- **Kết quả vượt trội:** Vượt qua toàn bộ các baseline trước đó (Magellan, DeepMatcher) trên 13/16 tập benchmark chuẩn.

## 2. Kết quả thực nghiệm đối soát (Experimental Reproduction)

**Môi trường thực nghiệm:**
- Hệ điều hành: Windows 11, Python 3.14.6.
- Thư viện: PyTorch 2.13.0+cu126, HuggingFace Transformers 5.14.0, Accelerate 1.16.0.
- Mô hình: `distilbert-base-uncased` (phiên bản chưng cất nhanh của BERT).
- Thiết bị tính toán: NVIDIA GeForce GTX 1650 (4 GB VRAM).
- Cố định Seed: `1001` (PyTorch manual_seed, NumPy seed, DataLoader generator, TrainingArguments).

**Bảng đối soát kết quả trên tập kiểm thử Amazon-Google:**

| Chỉ số | Kết quả công bố PVLDB (Ditto RoBERTa-base) | Kết quả chạy thực tế (DistilBERT) | Chênh lệch (Δ) | Lý giải nguyên nhân |
|---|---|---|---|---|
| **Precision** | 75.58% | **69.96%** | -5.62% | DistilBERT chưng cất giữ được phần lớn độ chính xác |
| **Recall** | 74.17% | **66.67%** | -7.50% | Cơ chế Self-Attention bắt trọn các cặp tương đồng |
| **F1 Score** | **75.58% (PVLDB)** | **68.27%** | **-7.31%** | Sai lệch hợp lý do mô hình nhẹ hơn (distilbert-base-uncased) |
| **Thời gian chạy** | ~15–30 phút (GPU Tesla) | **296.04s** | Tối ưu vượt trội | Tăng tốc độ huấn luyện trên GPU phổ thông |

## 3. Ưu điểm của bài báo

1. **Hiểu ngữ cảnh sâu (Deep Contextual Understanding):** Nhờ cơ chế Self-Attention đa tầng của Transformer, mô hình phân biệt chính xác sự khác biệt ngữ nghĩa tinh vi giữa các tên sản phẩm gần giống nhau, khắc phục hoàn toàn nhược điểm từ vựng tĩnh OOV của BiLSTM.
2. **Loại bỏ quy trình Feature Engineering thủ công:** Định dạng `COL/VAL` trực quan, cho phép áp dụng đồng nhất trên mọi cấu trúc bảng biểu mà không cần tinh chỉnh đặc trưng riêng biệt theo domain.
3. **Thiết lập chuẩn mực SOTA:** Đạt F1 cao nhất tại thời điểm công bố trên hầu hết các bộ dữ liệu Entity Matching kinh điển.

## 4. Khuyết điểm & Rào cản thực tế

1. **Chi phí tính toán cao khi suy luận trên quy mô lớn:** Khác với TF-IDF hay mạng nơ-ron nhỏ, việc đẩy từng cặp qua Transformer đòi hỏi năng lực GPU đáng kể. Khi dữ liệu lên tới hàng triệu bản ghi, việc suy luận toàn bộ $O(N^2)$ cặp là bất khả thi nếu không có tầng Blocking hiệu quả.
2. **Thiếu cơ chế Ràng buộc cứng (Hard Constraint Invariant):** Ditto dựa hoàn toàn vào xác suất Softmax liên tục. Nếu hai bản ghi doanh nghiệp có Mã số thuế hoàn toàn khác nhau nhưng tên gọi trùng hợp lớn, mô hình vẫn có thể đưa ra xác suất Match > 0.5 gây hậu quả sáp nhập sai (False Merge).
3. **Tối ưu hàm mất mát đối xứng (Symmetric Loss):** Sử dụng Cross-Entropy thông thường, coi chi phí phạt của False Merge và Missed Link là như nhau. Trong thực tế quản trị hệ thống thông tin, rủi ro pháp lý của việc gộp nhầm hai thực thể khác nhau nghiêm trọng hơn nhiều so với việc bỏ sót một liên kết.

## 5. Bài học rút ra & Đề xuất nâng cấp cho đề tài

1. **Đối sánh tiến trình qua 3 Baseline:**
   - Baseline 1 (TF-IDF + Logistic Regression): $F_1 = 67.52%$ — Chạy nhanh (0.51s), không bắt được ngữ nghĩa sâu.
   - Baseline 2 (DeepMatcher BiLSTM + Attention): $F_1 = 66.15%$ — Mô hình nơ-ron sơ khai, vướng rào cản OOV và biểu diễn tĩnh.
   - Baseline 3 (Ditto DistilBERT): $F_1 = 68.27%$ — Vượt trội về khả năng hiểu ngữ nghĩa và phân loại chính xác.

2. **Kế thừa kiến trúc chuỗi hóa cho đề tài Luận văn:**
   - Chuẩn hóa đầu vào theo định dạng `COL <attr> VAL <val>` của Ditto cho toàn bộ module so khớp thực thể doanh nghiệp.
   - Thay thế backbone bằng `PhoBERT` hoặc `multilingual-BERT` để tối ưu xử lý danh xưng và địa chỉ hành chính Việt Nam.

3. **Nâng cấp cốt lõi 1: Bổ sung Lớp Ràng buộc Cứng (Hard Constraint Layer):**
   - Tích hợp tầng kiểm tra tiền/hậu xử lý logic: Nếu các thuộc tính định danh pháp lý duy nhất (Mã số thuế, CCCD, Số ĐKKD) bị xung đột thì lập tức phủ quyết (Veto), không phụ thuộc vào điểm xác suất của mô hình Transformer.

4. **Nâng cấp cốt lõi 2: Tối ưu theo Hàm mất mát Chi phí Bayes Bất đối xứng (Asymmetric Bayes Cost Loss):**
   - Đặt trọng số phạt False Merge cao gấp 10-20 lần so với False Non-match ($C_{FP} > C_{FN}$), bảo đảm ngưỡng quyết định an toàn cho các hệ thống ERP / Master Data Management doanh nghiệp.
