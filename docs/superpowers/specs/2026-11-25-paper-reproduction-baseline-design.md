# Thiết Kế Chi Tiết: Tái Lập Thực Nghiệm Các Bài Báo Nền Tảng Cho Đề Tài ERP–CRM Entity Resolution

- **Ngày lập:** 2026-11-25
- **Học phần:** CSE631 - Quản Trị Hệ Thống Thông Tin Thông Minh (ĐH Thủy Lợi)
- **GVHD:** TS. Nguyễn Khắc Chiến
- **Mã đề tài:** ĐT02 (Đối sánh và hợp nhất dữ liệu khách hàng ERP–CRM)

---

## 1. Mục Tiêu & Căn Cứ Yêu Cầu

### 1.1. Mục tiêu
Thiết lập thư mục thực nghiệm `experiments/` độc lập, tải toàn văn bài báo gốc (PDF), mã nguồn chính thức (từ đúng các liên kết GitHub thầy Chiến chỉ định trong đề cương) và dữ liệu benchmark chuẩn để:
1. Tái lập thực nghiệm từng bài báo trên máy cá nhân (chạy mã nguồn gốc của tác giả).
2. Đối soát số liệu thực chạy (Precision, Recall, F1, Runtime) với số liệu tác giả công bố trong bài báo.
3. Phân tích ưu/nhược điểm kỹ thuật và chỉ ra khoảng trống cần nâng cấp cho đề tài nghiên cứu.

### 1.2. Căn cứ từ Đề cương thầy Chiến (`CSE631_DeTai_02_ERP_CRM_EntityMatching.pdf`)
- **Mục 2.1 (Tiêu chí 1):** Tái lập pipeline blocking → matching → resolution với ít nhất hai baseline hiện hành và ít nhất hai tập benchmark có nhãn.
- **Mục 5 & 14:** Nêu đích danh:
  - Mudgal et al. (SIGMOD 2018): *Deep Learning for Entity Matching: A Design Space Exploration*. Repo: `https://github.com/anhaidgroup/deepmatcher`.
  - Li et al. (PVLDB 2020/2021): *Deep Entity Matching with Pre-Trained Language Models* (Ditto). Repo: `https://github.com/megagonlabs/ditto`.
- **Mục 9.3 & 10.1:** Baseline rule và học máy truyền thống: `scikit-learn` (TF-IDF + Logistic Regression).
- **Mục 8:** Dữ liệu benchmark chuẩn DeepMatcher/Ditto từ ĐH Wisconsin: `https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/`.

---

## 2. Danh Sách Các Bài Báo & Nguồn Tài Nguyên Chỉ Định

| STT | Bài báo / Công cụ | Venue / Năm | Liên kết Mã nguồn (GitHub) | Dữ liệu Benchmark | Tài liệu gốc |
|---|---|---|---|---|---|
| **01** | **TF-IDF + Logistic Regression** | Baseline chuẩn (Mục 9.3) | `scikit-learn`, `rapidfuzz` | Amazon-Google, DBLP-ACM | Mục 9.3 & 10.1 đề cương |
| **02** | **DeepMatcher** (Mudgal et al.) | ACM SIGMOD 2018 | [anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher) | Amazon-Google (exp_data/raw) | DOI: 10.1145/3183713.3196926 |
| **03** | **Ditto** (Li et al.) | PVLDB 2020/2021 | [megagonlabs/ditto](https://github.com/megagonlabs/ditto) | er_magellan (Amazon-Google, DBLP-ACM) | DOI: 10.14778/3421424.3421431 |

*(Các bài tiếp theo gồm JointBERT PVLDB 2021, LEMON TKDE 2022, DAEM VLDBJ 2022 sẽ được tích hợp theo cùng khuôn mẫu trong Đợt 2).*

---

## 3. Cấu Trúc Thư Mục Thực Nghiệm

```
experiments/
├── 01_tfidf_baseline/
│   ├── data/                      # Dataset benchmark (Amazon-Google CSV)
│   ├── run.py                     # Script huấn luyện và kiểm thử TF-IDF + Logistic Regression
│   ├── results/                   # Log chi tiết, confusion matrix, metrics CSV
│   └── REPORT.md                  # Báo cáo đánh giá 5 mục
│
├── 02_deepmatcher_sigmod2018/
│   ├── paper.pdf                  # Toàn văn bài báo gốc (SIGMOD 2018)
│   ├── code/                      # Clone nguyên bản từ https://github.com/anhaidgroup/deepmatcher
│   ├── data/                      # Tải từ server Wisconsin (train.csv, valid.csv, test.csv)
│   ├── run.py                     # Script thực thi chạy pipeline DeepMatcher
│   ├── results/                   # Log huấn luyện, epoch losses, checkpoint, metrics CSV
│   └── REPORT.md                  # Báo cáo đối soát 5 mục
│
└── 03_ditto_pvldb2020/
    ├── paper.pdf                  # Toàn văn bài báo gốc (PVLDB 2020)
    ├── code/                      # Clone nguyên bản từ https://github.com/megagonlabs/ditto
    ├── data/                      # Dữ liệu serialize "COL ... VAL ..."
    ├── run.py                     # Script fine-tune DistilBERT/RoBERTa
    ├── results/                   # Checkpoint, tensorboard logs, metrics CSV
    └── REPORT.md                  # Báo cáo đối soát 5 mục
```

---

## 4. Khuôn Mẫu Báo Cáo Chuẩn (`REPORT.md` cho từng bài)

Mỗi bài báo sau khi hoàn thành thực nghiệm bắt buộc có `REPORT.md` với đúng 5 mục:

### Mục 1: Bài báo giải quyết vấn đề gì?
- Bản chất bài toán, đầu vào, đầu ra.
- Kiến trúc mô hình đề xuất (ví dụ: BiLSTM + Attribute Attention đối với DeepMatcher; Transformer + Domain Knowledge đối với Ditto).
- Đóng góp khoa học tại thời điểm công bố.

### Mục 2: Kết quả thực nghiệm đối soát (Experimental Reproduction)
- Mô tả môi trường: CPU/GPU, RAM, hệ điều hành, seed ngẫu nhiên (1001).
- Bảng so sánh kết quả thực tế vs kết quả bài báo:
  | Chỉ số | Kết quả bài báo công bố | Kết quả chạy thực tế trên máy | Chênh lệch (Δ) | Phân tích nguyên nhân sai lệch |
  |---|---|---|---|---|
  | Precision | ... | ... | ... | ... |
  | Recall | ... | ... | ... | ... |
  | F1 Score | ... | ... | ... | ... |
  | Thời gian huấn luyện | ... | ... | ... | ... |
  | Tốc độ suy luận (cặp/giây) | ... | ... | ... | ... |

### Mục 3: Ưu điểm của bài báo
- Khả năng xử lý dữ liệu, biểu diễn ngữ nghĩa.
- Kỹ thuật hay đáng kế thừa cho đề tài.

### Mục 4: Khuyết điểm & Rào cản thực tế
- Hạn chế khi áp dụng vào môi trường dữ liệu ERP–CRM doanh nghiệp.
- Vấn đề chi phí gộp sai bất đối xứng ($C_{FP} \gg C_{FN}$).
- Hiện tượng vi phạm ràng buộc cứng (Hard False Merges khi khác Mã số thuế).
- Sự phụ thuộc tài nguyên phần cứng hoặc phiên bản thư viện cũ.

### Mục 5: Bài học rút ra & Đề xuất nâng cấp cho đề tài
- Cần nâng cấp điều gì trong pipeline?
- Khung giải pháp của đề tài (CARE-ER) sẽ vượt trội hơn phương pháp này ở những điểm cụ thể nào?

---

## 5. Quy Trình Triển Khai Kỹ Thuật (3 Bước)

1. **Chuẩn bị môi trường & Dữ liệu:**
   - Tạo cấu trúc thư mục `experiments/`.
   - Viết script tải tự động PDF từ nguồn DOI chính thống.
   - Clone repo gốc qua git.
   - Tải bộ dữ liệu `Amazon-Google` (11.460 cặp có nhãn: 1.167 match) chuẩn bị cho chạy thử nghiệm.
2. **Thực thi lần lượt:**
   - **Bước 2.1:** Chạy `01_tfidf_baseline` (chạy nhanh, làm mốc tối thiểu).
   - **Bước 2.2:** Cài đặt và chạy `02_deepmatcher_sigmod2018`.
   - **Bước 2.3:** Cài đặt và chạy `03_ditto_pvldb2020`.
3. **Tổng hợp kết quả:**
   - Tự động trích xuất metrics ra file `REPORT.md` của từng bài.
   - Lập bảng tổng hợp so sánh chéo 3 phương pháp phục vụ báo cáo tiến độ môn học và luận văn.
