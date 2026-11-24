# Lộ Trình Nghiên Cứu Đề Tài: Đối Sánh Và Hợp Nhất Dữ Liệu Khách Hàng Giữa ERP Và CRM Với Dữ Liệu Không Đồng Nhất

- **Học phần:** Quản trị Hệ thống Thông tin Thông minh (CSE631) - ĐH Thủy Lợi
- **GVHD:** TS. Nguyễn Khắc Chiến
- **Mã đề tài:** ĐT02
- **Trọng tâm:** ERP – CRM | Entity Resolution / Record Linkage | Data Fusion

---

## 1. Bản Chất Bài Toán & Mục Tiêu

### 1.1. Bản chất kỹ thuật
- **Vấn đề:** Hệ thống ERP và CRM trong doanh nghiệp thường được triển khai độc lập, phục vụ mục đích khác nhau. Dữ liệu khách hàng bị phân mảnh, dị thể (heterogeneous), không có khóa chính chung (Foreign Key / Global ID).
- **Thách thức:**
  - Khác biệt lược đồ (Schema Heterogeneity): ERP lưu theo pháp nhân, tài chính; CRM lưu theo cơ hội bán hàng, liên hệ.
  - Khác biệt định dạng dữ liệu (Data Value Heterogeneity): Tên sai lệch, viết tắt ("Cty TNHH" vs "Công ty TNHH"), SĐT khác format (`+84` vs `09...`), địa chỉ phi chuẩn hóa.
  - Bùng nổ tổ hợp so sánh: $N$ bản ghi ERP $\times$ $M$ bản ghi CRM $\rightarrow O(N \times M)$ phép so sánh nếu không dùng kỹ thuật lọc (Blocking).
- **Đầu ra mong muốn:** Pipeline tự động đối sánh (Entity Matching) và hợp nhất thành một bản ghi duy nhất đáng tin cậy (Golden Record / Master Data Record).

### 1.2. Mục tiêu sản phẩm nộp môn học
1. **Source Code:** Pipeline hoàn chỉnh (Python) + Giao diện Demo (Streamlit).
2. **Dataset:** Dữ liệu ERP & CRM thử nghiệm (có nhãn kiểm chứng - Ground Truth).
3. **Báo cáo học phần:** Bản Word chuẩn mẫu ĐH Thủy Lợi (15 - 25 trang).
4. **Slide thuyết trình & Video demo:** Chuẩn bị bảo vệ vấn đáp cuối kỳ (chiếm 50% tổng điểm).

---

## 2. Kế Hoạch Lộ Trình Chi Tiết (6 Giai Đoạn)

### Giai đoạn 1: Cơ sở lý thuyết & Khảo sát công nghệ (Tuần 1 - Tuần 2)
**Mục tiêu:** Nắm vững lý thuyết để viết Chương 1 & Chương 2 của báo cáo.
- [ ] **Nghiên cứu kiến trúc ERP & CRM:**
  - Lược đồ dữ liệu khách hàng trong ERP (kế toán, hóa đơn, công nợ, mã số thuế).
  - Lược đồ dữ liệu khách hàng trong CRM (lead, deal, contact info, hành vi tương tác).
  - Phân tích nguyên nhân gây dị thể dữ liệu (Data Heterogeneity).
- [ ] **Khảo sát quy trình Entity Resolution (ER):**
  - **Tiền xử lý (Preprocessing):** Chuẩn hóa chuỗi tiếng Việt, regex, unidecode.
  - **Chặn phân cụm (Blocking / Indexing):** Standard Blocking, Sorted Neighborhood, Q-Gram Indexing nhằm giảm không gian tìm kiếm từ $O(N \times M)$ xuống $O(k)$.
  - **Hàm đo khoảng cách chuỗi (String Similarity Metrics):**
    - Edit-based: Levenshtein, Jaro-Winkler (tốt cho tên người, tên công ty).
    - Token-based: Jaccard, Cosine Similarity kết hợp TF-IDF (tốt cho địa chỉ).
    - Phonetic: Double Metaphone / Soundex (nếu áp dụng tên tiếng Anh).
  - **Mô hình phân loại đối sánh:**
    - Heuristic / Rule-based (kết hợp ngưỡng điểm trọng số).
    - Machine Learning Classification (Logistic Regression, Random Forest, XGBoost) dựa trên vector đặc trưng tương đồng.
- [ ] **Khảo sát kỹ thuật hợp nhất dữ liệu (Data Fusion / Golden Record):**
  - Chiến lược giải quyết xung đột (Conflict Resolution Strategies): Recency (ưu tiên bản ghi mới nhất), Trustworthiness (ưu tiên hệ thống ERP cho thông tin tài chính/MST, CRM cho thông tin liên hệ), Completeness (chọn trường đầy đủ nhất).
- **Tài liệu/Thư viện cần đọc:** Python `recordlinkage`, `dedupe`, `thefuzz`, `scikit-learn`.

---

### Giai đoạn 2: Xây dựng Dữ liệu Thực nghiệm (Tuần 3)
**Mục tiêu:** Tạo bộ dữ liệu đại diện cho tình huống thực tế của doanh nghiệp.
- [ ] **Thiết kế Lược đồ (Schema Definition):**
  - Schema ERP: `[erp_id, tax_code, company_name, representative_name, billing_address, phone, created_date]`
  - Schema CRM: `[crm_id, lead_name, contact_person, email, mobile_phone, shipping_address, source, last_interaction]`
- [ ] **Sinh dữ liệu mẫu (Synthetic Data Generation):**
  - Viết script Python (dùng thư viện `Faker` phiên bản tiếng Việt `vi_VN`).
  - Sinh tập cơ sở gồm $K$ khách hàng thật.
  - Tạo 2 view: ERP (bổ sung noise kiểu hành chính, thuế) và CRM (bổ sung noise kiểu gõ sai, viết tắt, thiếu trường).
  - Gieo nhiễu có chủ đích: Lỗi gõ telex/vni, đảo từ, bỏ dấu tiếng Việt, viết tắt ("Cty", "TNHH", "P.", "Q.", "TP."), đổi đầu số điện thoại.
- [ ] **Tạo Ground Truth (Nhãn chân thực):**
  - Lưu bảng ánh xạ cặp ID chuẩn `(erp_id, crm_id, is_match=1/0)` để phục vụ huấn luyện ML và đánh giá kết quả.

---

### Giai đoạn 3: Thiết kế & Xây dựng Pipeline Kỹ thuật (Tuần 4 - Tuần 5)
**Mục tiêu:** Code hoàn chỉnh luồng xử lý từ input thô đến golden record.

```
Dữ liệu thô (ERP + CRM)
       │
       ▼
[Module 1: Preprocessing & Standardization]  ──> Làm sạch chuỗi, chuẩn hóa tiếng Việt, format số
       │
       ▼
[Module 2: Blocking / Indexing]              ──> Lọc bớt cặp không tiềm năng
       │
       ▼
[Module 3: Feature Extraction & Matching]   ──> Tính điểm tương đồng từng trường & phân loại Match/Non-match
       │
       ▼
[Module 4: Data Fusion / Merging]           ──> Giải quyết xung đột thuộc tính, xuất Golden Record
```

- [ ] **Module 1: Chuẩn hóa dữ liệu (Preprocessing):**
  - Chuyển lowercase, strip whitespace thừa.
  - Chuẩn hóa viết tắt doanh nghiệp ("công ty trách nhiệm hữu hạn" $\leftrightarrow$ "cty tnhh").
  - Chuẩn hóa số điện thoại về định dạng E.164 hoặc `09xxxxxxxx`.
  - Tách/chuẩn hóa cấu trúc địa chỉ (Số nhà, Phường/Xã, Quận/Huyện, Tỉnh/TP).
- [ ] **Module 2: Bộ lọc cặp tiềm năng (Blocking):**
  - Triển khai Blocking theo: Mã số thuế (Exact match nếu có), hoặc Block theo Tỉnh/Thành phố kết hợp First-Letter của tên.
  - Đánh giá tỷ lệ giảm tải (Reduction Ratio) và Pair Completeness.
- [ ] **Module 3: Tính toán độ tương đồng & Phân lớp (Matching Engine):**
  - Xây dựng Feature Vector cho từng cặp ứng viên:
    - $Sim_{name} = \text{Jaro-Winkler}(\text{Name}_{erp}, \text{Name}_{crm})$
    - $Sim_{address} = \text{TF-IDF Cosine}(\text{Addr}_{erp}, \text{Addr}_{crm})$
    - $Sim_{phone} = \text{Exact / Levenshtein}(\text{Phone}_{erp}, \text{Phone}_{crm})$
    - $Sim_{tax} = \text{Exact}(\text{Tax}_{erp}, \text{Tax}_{crm})$
  - Xây dựng 2 phương pháp để so sánh:
    1. *Phương pháp 1 (Baseline):* Weighted Average Threshold (Rule-based).
    2. *Phương pháp 2 (Nâng cao):* Machine Learning Classifier (Logistic Regression / Random Forest).
- [ ] **Module 4: Hợp nhất dữ liệu (Entity Merging / Golden Record):**
  - Xây dựng quy tắc Resolve Conflict:
    - Mã số thuế / Tên pháp nhân $\rightarrow$ Lấy từ ERP.
    - Email / SĐT di động cá nhân $\rightarrow$ Lấy bản ghi có tương tác gần nhất hoặc từ CRM.
    - Địa chỉ $\rightarrow$ Lấy bản ghi có độ chi tiết cao hơn.
  - Xuất bảng dữ liệu hợp nhất `Master_Customer_Record`.

---

### Giai đoạn 4: Đánh giá Thực nghiệm & Phân tích Kết quả (Tuần 6)
**Mục tiêu:** Đo lường định lượng cho báo cáo học thuật.
- [ ] **Tính toán các chỉ số đánh giá (Evaluation Metrics):**
  - **Precision:** $\frac{TP}{TP + FP}$ (Độ chính xác: số cặp dự đoán trùng có thực sự trùng?).
  - **Recall:** $\frac{TP}{TP + FN}$ (Độ bao phủ: phát hiện được bao nhiêu % cặp trùng thực tế?).
  - **F1-Score:** $2 \times \frac{Precision \times Recall}{Precision + Recall}$.
  - **Execution Time:** Thời gian chạy toàn bộ pipeline khi scale số lượng bản ghi (1k, 5k, 10k bản ghi).
- [ ] **Phân tích lỗi (Error Analysis):**
  - Vì sao bị False Positive (nhận nhầm hai khách hàng khác nhau thành một)?
  - Vì sao bị False Negative (cùng một khách hàng nhưng thuật toán bỏ sót)?
- [ ] **Bảng so sánh:**
  - So sánh hiệu năng giữa: Có Blocking vs Không Blocking.
  - So sánh: Rule-based Threshold vs ML-based Classification.

---

### Giai đoạn 5: Phát triển Giao diện Ứng dụng Demo (Tuần 6 - Tuần 7)
**Mục tiêu:** Có ứng dụng trực quan để quay video và trình diễn lúc vấn đáp.
- [ ] **Xây dựng UI bằng Streamlit (`app.py`):**
  - **Tab 1: Quản lý Dữ liệu:** Cho phép upload file ERP (CSV/Excel) và CRM (CSV/Excel), xem preview dữ liệu thô.
  - **Tab 2: Cấu hình Matching:** Cho phép chọn thuật toán (Rule-based / ML), điều chỉnh ngưỡng similarity (slider threshold).
  - **Tab 3: Kết quả Đối sánh:** Hiển thị danh sách các cặp matched với độ tin cậy (Confidence Score), highlight trường khớp/lệch.
  - **Tab 4: Dữ liệu Hợp nhất (Golden Records):** Xem và tải về file CSV/Excel khách hàng đã được hợp nhất.
  - **Tab 5: Dashboard Đánh giá:** Biểu đồ Confusion Matrix, Precision, Recall, F1.

---

### Giai đoạn 6: Hoàn thiện Báo cáo, Slide & Video (Tuần 7 - Kết thúc)
**Mục tiêu:** Tối đa điểm quá trình (BTL 20%) và điểm thi cuối kỳ (Vấn đáp 50%).
- [ ] **Soạn thảo báo cáo Word (Theo format luận văn / báo cáo học cao học):**
  - *Mở đầu:* Lý do chọn đề tài, mục tiêu, đối tượng phạm vi nghiên cứu.
  - *Chương 1:* Tổng quan về ERP, CRM và bài toán tích hợp dữ liệu dị thể.
  - *Chương 2:* Cơ sở lý thuyết về Entity Resolution, Blocking, String Distance và Data Fusion.
  - *Chương 3:* Thiết kế kiến trúc giải pháp và quy trình xử lý.
  - *Chương 4:* Thực nghiệm, kết quả đánh giá định lượng và phân tích.
  - *Kết luận & Hướng phát triển:* Đóng góp của đề tài, hạn chế và tiềm năng ứng dụng thực tế.
- [ ] **Thiết kế Slide thuyết trình (15 - 20 slides):**
  - Tập trung vào: Vấn đề $\rightarrow$ Kiến trúc giải pháp $\rightarrow$ Kết quả thực nghiệm $\rightarrow$ Demo UI.
- [ ] **Quay video demo ngắn (5 - 7 phút):**
  - Thuyết minh luồng chạy từ nạp dữ liệu đến xuất Golden Record.
- [ ] **Chuẩn bị các câu hỏi vấn đáp trọng tâm:**
  - *"Tại sao phải dùng Blocking? Nếu không có Blocking điều gì xảy ra?"*
  - *"Tại sao chọn khoảng cách Jaro-Winkler cho trường Tên mà không dùng Levenshtein?"*
  - *"Khi hai trường dữ liệu xung đột nhau thì hệ thống quyết định lấy giá trị nào?"*

---

## 3. Cấu Trúc Thư Mục Dự Án Đề Xuất

```
QuanTriHeThongThongTin/
├── CLAUDE.md                   # Chỉ dẫn ngữ cảnh dự án
├── LO_TRINH_NGHIEN_CUU.md      # File tài liệu này
├── data/
│   ├── raw/                    # Dữ liệu mẫu ERP & CRM thô
│   ├── processed/              # Dữ liệu sau tiền xử lý
│   └── ground_truth/           # File nhãn đối sánh chuẩn
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Script sinh dữ liệu ERP/CRM có nhiễu
│   ├── preprocessor.py         # Module làm sạch & chuẩn hóa tiếng Việt
│   ├── blocking.py             # Module kỹ thuật chặn phân cụm
│   ├── matcher.py              # Module tính similarity & phân loại
│   └── fusion.py               # Module giải quyết xung đột & xuất Golden Record
├── app.py                      # Ứng dụng Web Demo (Streamlit)
├── notebooks/
│   └── experiments.ipynb       # Jupyter Notebook thử nghiệm & vẽ biểu đồ
├── reports/
│   ├── Báo_Cáo_BTL_CSE631_DT02.docx
│   └── Slide_Thuyet_Trinh.pptx
└── requirements.txt            # Danh sách thư viện Python
```

---

## 4. Các Mốc Thời Gian Trọng Yếu (Milestones)

| Mốc thời gian | Nhiệm vụ chính | Sản phẩm đầu ra |
|---|---|---|
| **Tuần 1 - 2** | Nghiên cứu lý thuyết ER, ERP-CRM, Blocking | Đề cương chi tiết báo cáo, tài liệu tổng quan |
| **Tuần 3** | Viết script sinh dữ liệu có nhiễu, tạo Ground Truth | Bộ file CSV ERP, CRM và nhãn chuẩn |
| **Tuần 4 - 5** | Hiện thực hóa Pipeline Python (4 module) | Mã nguồn core matching trong thư mục `src/` |
| **Tuần 6** | Đánh giá chỉ số (Precision, Recall, F1), làm UI Streamlit | `app.py` chạy hoàn chỉnh, số liệu thực nghiệm |
| **Tuần 7 (Nộp BTL)** | Viết xong báo cáo Word, hoàn thiện Slide | File Word báo cáo hoàn chỉnh, nộp lấy 20% điểm |
| **Sau tuần 7** | Luyện vấn đáp, quay video demo | Chuẩn bị thi vấn đáp kết thúc môn (50% điểm) |
