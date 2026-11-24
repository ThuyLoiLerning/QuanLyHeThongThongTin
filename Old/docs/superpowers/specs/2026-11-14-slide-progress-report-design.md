# Slide Progress Report Design – Hướng 1

**Date:** 2026-11-14
**Author:** Claude Code (assistant)
**Project:** ĐT02 – Đối sánh và hợp nhất dữ liệu khách hàng giữa ERP và CRM

## 1. Mục tiêu tài liệu
- Định nghĩa chi tiết thiết kế slide báo cáo tiến độ (7–9 slide) cho buổi kiểm tra giữa kỳ với giảng viên hướng dẫn.
- Cung cấp các thông tin cấu trúc, nội dung, phong cách thiết kế và file output (`.pptx`).
- Đảm bảo tài liệu dễ hiểu, không còn placeholder, và có các chỉ dẫn rõ ràng cho việc tạo slide sau này.

## 2. Quy chuẩn hình thức (Visual Theme)
- **Kích thước:** 16:9 (1920×1080).
- **Màu sắc:**
  - Primary: Deep Navy Blue `#0F2537`.
  - Secondary: Teal `#00A896`.
  - Background: White `#F8F9FA`.
  - Accent (cảnh báo/xung đột): Coral `#E76F51`.
- **Font:** Segoe UI, Arial, Calibri – hỗ trợ tiếng Việt.
- **File output:** `reports/Slide_Bao_Cao_Tien_Do.pptx` (tạo bằng `python-pptx`).

## 3. Cấu trúc slide (8 slide)
| Slide | Chủ đề | Nội dung chính |
|------|--------|----------------|
| 1 | Tiêu đề | Tên đề tài, GVHD, Học phần, tên sinh viên. |
| 2 | Bối cảnh & Dị thể dữ liệu | Mô tả ngắn ERP vs CRM, ví dụ minh họa (ảnh/đồ thị). |
| 3 | Mục tiêu & Phạm vi | Liệt kê mục tiêu nghiên cứu (Entity Matching, Data Fusion) + phạm vi (ERP‑CRM). |
| 4 | Kiến trúc pipeline (4 module) | Sơ đồ khối: Preprocessing → Blocking → Matching → Fusion. Mô tả mỗi khối 1‑2 câu. |
| 5 | Thiết kế thử nghiệm | Lược đồ schema ERP/CRM, cách sinh dữ liệu có nhãn (Faker). |
| 6 | Kết quả tiến độ hiện tại | Hoàn thành: nghiên cứu lý thuyết, lập lộ trình, thiết kế schema, roadmap. |
| 7 | Kế hoạch giai đoạn tiếp theo | Timeline tuần 4‑7: code pipeline, UI Streamlit, đánh giá, viết báo cáo. |
| 8 | Kết luận & Xin ý kiến | Tóm tắt, mong muốn phản hồi từ GVHD. |

## 4. Nội dung chi tiết từng slide (đề xuất bullet)
- **Slide 2:** "ERP lưu thông tin pháp nhân, CRM lưu thông tin liên hệ – dữ liệu không đồng nhất (Tên, Địa chỉ, SĐT)."
- **Slide 4:** Sử dụng SmartArt hoặc shape arrows để nối 4 khối; mỗi khối có màu secondary và tiêu đề.
- **Slide 6:** Checklist các việc đã hoàn thành – đánh dấu ✓.

## 5. Đầu ra & quy trình tạo slide (tương lai)
1. **Script tạo PPTX** (`python-pptx`) sẽ đọc file này, tự động tạo slide dựa trên bảng nội dung.
2. **Kiểm tra manual**: mở file `reports/Slide_Bao_Cao_Tien_Do.pptx` để chỉnh sửa cuối cùng.

---

## 6. Self‑review checklist
- [x] Không có placeholder “TODO”.
- [x] Nội dung slide tương thích với thời lượng 5‑10 phút.
- [x] Các màu, font, tỷ lệ đã được chỉ định rõ.
- [x] Định danh file output nhất quán với `reports/`.
- [x] Không có mâu thuẫn giữa mô tả kiến trúc và các slide.
- [x] Phạm vi đề tài (ERP‑CRM) được nhắc lại ở slide 2 & 3.

---

**Next step:** Người dùng xem và phản hồi tài liệu này. Khi được phê duyệt, tôi sẽ gọi skill `writing-plans` để tạo kế hoạch triển khai slide (cài script, tạo PPTX, review).