---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    background-color: #F8F9FA;
    color: #0F2537;
  }
  h1 {
    color: #0F2537;
  }
  h2 {
    color: #00A896;
  }
  .highlight {
    color: #E76F51;
  }
  .card {
    background: #FFFFFF;
    border: 1px solid #00A896;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
  }
---

<!-- _backgroundColor: #0F2537 -->
<!-- _color: #FFFFFF -->

# BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU ĐỀ TÀI (ĐỢT 1)
## Đối Sánh Và Hợp Nhất Dữ Liệu Khách Hàng Giữa ERP Và CRM Với Dữ Liệu Không Đồng Nhất

**Mã đề tài:** ĐT02 | **Học phần:** Quản trị Hệ thống Thông tin Thông minh (CSE631)
**Giảng viên hướng dẫn:** TS. Nguyễn Khắc Chiến
**Đơn vị:** Khoa Công nghệ Thông tin — Trường Đại học Thủy Lợi

---

# 1. Bối Cảnh & Vấn Đề Dị Thể Dữ Liệu ERP – CRM
### Dữ liệu phân mảnh, không đồng nhất và thiếu khóa định danh chung

* **Hệ Thống ERP (Back-office):**
  * Quản lý tài chính, hóa đơn, công nợ, xuất kho.
  * Tên pháp nhân chuẩn: *CÔNG TY TNHH CÔNG NGHỆ Á CHÂU*
  * Có Mã số thuế (`0102345678`), địa chỉ trụ sở đầy đủ, SĐT bàn kế toán.

* **Hệ Thống CRM (Front-office):**
  * Quản lý cơ hội bán hàng, marketing, chăm sóc khách hàng.
  * Tên ghi nhận tự do/viết tắt: *Cty A Chau Tech (Anh Minh)*
  * Thường khuyết MST, dùng SĐT di động cá nhân, địa chỉ giao hàng rút gọn.

* **Thách thức cốt lõi:**
  * **Không có Foreign Key chung** $\rightarrow$ Không thể JOIN bảng theo cách thông thường.
  * **Bùng nổ không gian so sánh:** $N \times M = O(N \times M)$ (10.000 $\times$ 10.000 = 100 triệu cặp).

---

# 2. Mục Tiêu Nghiên Cứu & Phạm Vi Đề Tài

### Mục tiêu kỹ thuật:
1. Làm chủ quy trình **Entity Resolution (ER)** cho dữ liệu tiếng Việt có cấu trúc doanh nghiệp & hành chính.
2. Triển khai kỹ thuật **Blocking (Chặn phân cụm)** để giảm tối đa không gian so sánh $O(N \times M)$.
3. Thực nghiệm so sánh 2 hướng tiếp cận:
   * **Phương pháp Rule-based / Heuristic:** Kết hợp trọng số các hàm khoảng cách chuỗi.
   * **Phương pháp Machine Learning:** Trích xuất vector tương đồng $\rightarrow$ Bộ phân lớp (Classification).
4. Xây dựng module **Data Fusion** giải quyết xung đột thuộc tính, xuất bản ghi chuẩn (**Golden Record**).

### Phạm vi & Sản phẩm đầu ra:
* Bộ dữ liệu mô phỏng khách hàng doanh nghiệp & cá nhân có **Ground Truth** (nhãn chuẩn kiểm chứng).
* Mã nguồn module hóa Python + Ứng dụng Web Demo trực quan bằng **Streamlit**.
* Báo cáo chuyên đề và slide bảo vệ vấn đáp cuối kỳ.

---

# 3. Kiến Trúc Pipeline Giải Pháp Đề Xuất

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     Module 1: TIỀN XỬ LÝ       │ ───> │       Module 2: BLOCKING        │
│ Làm sạch text, unidecode, regex │      │ Chặn phân cụm theo MST / Tỉnh   │
│ chuẩn hóa Cty/TNHH, format SĐT  │      │ Giảm 95%+ tổ hợp so sánh O(NxM) │
└─────────────────────────────────┘      └─────────────────────────────────┘
                 │                                        │
                 ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│      Module 4: DATA FUSION      │ <─── │   Module 3: MATCHING ENGINE     │
│ Hợp nhất bản ghi, xử lý xung    │      │ Tính similarity (Jaro-Winkler,  │
│ đột thuộc tính -> Golden Record │      │ Levenshtein, TF-IDF) & phân lớp │
└─────────────────────────────────┘      └─────────────────────────────────┘
```

---

# 4. Thiết Kế Thử Nghiệm & Lược Đồ Dữ Liệu

### Lược đồ thuộc tính:
* **ERP Schema:** `[erp_id, tax_code, company_name, representative, billing_address, phone, created_date]`
* **CRM Schema:** `[crm_id, lead_name, contact_person, email, mobile_phone, shipping_address, last_contact]`
* **Ground Truth:** `[erp_id, crm_id, is_match]` (Tỷ lệ match thực tế: 40% - 60%).

### Cơ chế gieo nhiễu thực tế (Noise Injection):
* Sinh dữ liệu gốc bằng `Faker` (locale `vi_VN`).
* **Lỗi Telex/VNI:** Sai lệch ký tự gõ phím (*Á Châu* $\leftrightarrow$ *As Châu*).
* **Viết tắt & Từ đồng nghĩa:** *Cty* $\leftrightarrow$ *Công ty*, *TNHH* $\leftrightarrow$ *Trách nhiệm hữu hạn*.
* **Nhiễu địa chỉ:** Lược bỏ tiền tố Phường/Quận, thay đổi trật tự số nhà.
* **Nhiễu số điện thoại:** Biến thiên định dạng `+84` / `09xx` / dấu chấm phân tách.

---

# 5. Kết Quả Tiến Độ Hiện Tại

### Đã hoàn thành (Giai đoạn chuẩn bị):

* **Nghiên cứu lý thuyết nền tảng:**
  * Hệ thống hóa bài toán Entity Resolution, Record Linkage và Data Fusion.
  * Khảo sát các hàm đo khoảng cách chuỗi tối ưu cho tiếng Việt (*Jaro-Winkler*, *Levenshtein*, *Cosine TF-IDF*).

* **Xây dựng lộ trình khoa học:**
  * Hoàn thiện tài liệu `LO_TRINH_NGHIEN_CUU.md` gồm 6 giai đoạn rõ ràng.
  * Phân bổ timeline bám sát lịch học và mốc đánh giá của học phần CSE631.

* **Đặc tả kiến trúc & kịch bản thực nghiệm:**
  * Thiết kế chi tiết Schema hai hệ thống ERP và CRM.
  * Lập cấu trúc thư mục module hóa `src/` và kịch bản ứng dụng Demo Streamlit.

---

# 6. Kế Hoạch Thực Hiện Giai Đoạn Tiếp Theo

* **Tuần 3:**
  * Hiện thực hóa script Python sinh dữ liệu mẫu (`data_generator.py`) kèm Ground Truth.
* **Tuần 4:**
  * Lập trình Core Pipeline: `preprocessor.py`, `blocking.py`, `matcher.py` (Rule-based & ML).
* **Tuần 5 - 6:**
  * Lập trình module `fusion.py` tạo Golden Record.
  * Đánh giá định lượng các chỉ số: **Precision, Recall, F1-Score, Reduction Ratio**.
  * Phát triển giao diện Web Demo bằng **Streamlit** (`app.py`).
* **Tuần 7:**
  * Hoàn thiện báo cáo chuyên đề Word theo quy chuẩn ĐHTL.
  * Hoàn thiện slide và kịch bản vấn đáp bảo vệ kết thúc môn.

---

<!-- _backgroundColor: #0F2537 -->
<!-- _color: #FFFFFF -->

# KẾT LUẬN & XIN Ý KIẾN HƯỚNG DẪN

> **Đề tài đã hoàn tất giai đoạn nghiên cứu cơ sở lý thuyết, định hình kiến trúc kỹ thuật và kế hoạch thực nghiệm.**

### Các nội dung kính xin ý kiến chỉ đạo từ Thầy TS. Nguyễn Khắc Chiến:

1. **Quy mô tập dữ liệu thực nghiệm:** Đề xuất 1.000 vs 10.000 bản ghi để vừa đảm bảo tính tin cậy học thuật vừa tối ưu thời gian thực thi trên máy cá nhân.
2. **Hướng phân lớp đối sánh:** Đánh giá việc kết hợp Machine Learning Classifier (Random Forest / Logistic Regression) so với Rule-based truyền thống.
3. **Quy tắc giải quyết xung đột (Data Fusion):** Các tiêu chí ưu tiên thuộc tính phù hợp với thực tiễn doanh nghiệp Việt Nam.

## EM XIN TRÂN TRỌNG CẢM ƠN THẦY!
