# BaiBao — Bộ sưu tập 13 công trình ERP–CRM / Entity Matching

Thư mục này chứa toàn văn (PDF) và mã nguồn (nếu có) của 13 công trình được khảo sát trong
`docs/BaoCao_10Plus_BaiBao_ERP_CRM_Entity_Matching.docx`.

- Ngày tải: 15/11/2026
- Quy ước: mỗi thư mục `BaiN_<tên>` gồm `paper.pdf` (toàn văn) và `code/` (mã nguồn chính thức, nếu có)
- Danh sách chi tiết ở [`MANIFEST.csv`](MANIFEST.csv)

## Tổng quan

| # | Công trình | Venue | PDF | Code |
|---|-----------|-------|-----|------|
| 1 | Deep Entity Matching with Pre-trained Language Models (Ditto) | PVLDB 2020 | ✅ arXiv | ✅ `megagonlabs/ditto` |
| 2 | Dual-Objective Fine-Tuning of BERT for Entity Matching | PVLDB 2021 | ✅ bản tác giả | ✅ `wbsg-uni-mannheim/jointbert` |
| 3 | Automating Entity Matching Model Development (AutoEM) | ICDE 2021 | ✅ bản tác giả | ❌ chưa thấy repo chính thức |
| 4 | Deep Entity Matching with Adversarial Active Learning (DAEM) | VLDBJ 2022 | ❌ paywall | ✅ `nju-websoft/DAEM` |
| 5 | LEMON: Explainable Entity Matching | IEEE TKDE 2022 | ✅ arXiv | ✅ `NilsBarlaug/lemon` |
| 6 | Privacy-Preserving Deep Learning Based Record Linkage | IEEE TKDE 2023 | ✅ arXiv | ❌ chưa thấy repo chính thức |
| 7 | Entity Matching using Large Language Models | arXiv 2023 | ✅ arXiv | ✅ `wbsg-uni-mannheim/MatchGPT` |
| 8 | Learning from Natural Language Explanations for Generalizable Entity Matching | EMNLP 2024 | ✅ ACL | ❌ chưa thấy repo chính thức |
| 9 | Cost-efficient Prompt Engineering for Unsupervised Entity Resolution | Discover AI 2024 | ✅ Springer OA | ❌ chưa thấy repo chính thức |
| 10 | An Incremental Entity Resolution Approach based on Deep Metric Learning | Research Square 2024 (preprint) | ✅ Research Square | ❌ chưa thấy repo chính thức |
| 11 | On Active Learning of Record Matching Packages | SIGMOD 2010 | ⚠️ chỉ có slide | ❌ |
| 12 | A Bayesian Decision Model for Cost Optimal Record Matching | VLDBJ 2003 | ❌ paywall | ❌ |
| 13 | A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication | IEEE TKDE 2012 | ❌ paywall | ❌ |

## Ghi chú quan trọng

- **Bài 11 — DOI trong báo cáo bị sai.** Báo cáo ghi `10.1145/1807167.1807256`, nhưng DOI này thuộc một bài SIGMOD 2010 khác. DOI đúng của *On Active Learning of Record Matching Packages* (Arasu, Götz, Kaushik) là **`10.1145/1807167.1807252`**. **Lưu ý:** file `paper.pdf` hiện có là **slide trình bày** SIGMOD 2010 (24 trang), không phải bài báo 12 trang — cần bổ sung bản full-text (ACM DL / thư viện trường) trước khi phân tích.
- **Bài 4 — chưa có PDF.** Bài là open access trên Springer nhưng bản trên figshare/RMIT chỉ có metadata, không có file đính kèm; không tìm thấy bản preprint hợp pháp. Đã có mã nguồn đầy đủ trong `Bai4_DAEM_VLDBJ2022/`.
- **Bài 13 — chưa có PDF.** File `paper.pdf` từng nằm trong thư mục là một bài khác (Waikato Journal of Education 13:2007) nên đã bị xóa; xem `Bai13_IndexingSurvey_TKDE2012/LINK_PAPER.md`. Unpaywall báo `is_oa: false`; ANU DSpace trả 401 với tải tự động.
- **Bài 12 — chưa có PDF.** Bài cũ (2003) nằm sau paywall Springer; trang tác giả (`ssp.ece.upatras.gr`) timeout từ mạng hiện tại. Có thể tải qua thư viện trường nếu cần.
- **Bài 1, 5, 6** dùng bản arXiv (bản PVLDB/IEEE gốc bị chặn bot hoặc paywall). Nội dung khoa học tương đương bản tác giả; khi trích dẫn hãy dùng đúng venue/DOI trong báo cáo.
- **Bài 10** là preprint Research Square, chưa peer-review — không nên dùng để khẳng định SOTA.
- **Bài 9** là bản published open access trên Springer (Discover Artificial Intelligence).
- Các repo code đã clone nông (`--depth 1`); muốn cập nhật: `git -C <thư mục> pull`.

## Cấu trúc

```
BaiBao/
├── README.md
├── MANIFEST.csv
├── Bai1_Ditto_PVLDB2020/
│   ├── paper_arxiv.pdf
│   └── code/ditto/
├── Bai2_DualObjective_BERT_PVLDB2021/
│   ├── paper.pdf
│   └── code/jointbert/
├── ...
├── Bai4_DAEM_VLDBJ2022/
│   ├── DAEM-main/          # code + datasets + scripts
│   ├── DAEM-main.zip
│   └── LINK_PAPER.md
├── Bai12_BayesianCost_RecordMatching_VLDBJ2003/
│   └── LINK_PAPER.md
└── Bai13_IndexingSurvey_TKDE2012/
    └── LINK_PAPER.md
```
