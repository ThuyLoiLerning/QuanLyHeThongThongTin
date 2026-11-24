# Slide Progress Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng script Python sinh file thuyết trình PowerPoint (`reports/Slide_Bao_Cao_Tien_Do.pptx`) 8 slide chuẩn 16:9 báo cáo tiến độ đề tài ĐT02 với GVHD TS. Nguyễn Khắc Chiến.

**Architecture:** Sử dụng thư viện `python-pptx` để lập trình sinh slide theo hướng module hóa: định nghĩa bảng màu (Navy Blue, Teal, White, Coral), layout tỷ lệ 16:9, các hàm helper tạo card khối nội dung, vẽ sơ đồ pipeline 4 module dạng shape/arrow, và xuất file `.pptx` tự động kèm bộ kiểm thử `pytest`.

**Tech Stack:** Python 3.10+, `python-pptx`, `pytest`.

## Global Constraints

- Slide aspect ratio: 16:9 (`slide_width = Inches(13.333)`, `slide_height = Inches(7.5)`).
- Palette: Primary `#0F2537` (RGB 15, 37, 55), Secondary `#00A896` (RGB 0, 168, 150), Accent `#E76F51` (RGB 231, 111, 81), Background `#F8F9FA` (RGB 248, 249, 250), Card Fill `#FFFFFF` (RGB 255, 255, 255), Text Muted `#555555` (RGB 85, 85, 85).
- Fonts: `Segoe UI` (Primary), fallback `Calibri`, `Arial`.
- Ngôn ngữ nội dung: Tiếng Việt có dấu chuẩn xác.
- Output file path: `reports/Slide_Bao_Cao_Tien_Do.pptx`.

---

### Task 1: Thiết lập Dependencies & Cấu trúc Thư mục

**Files:**
- Create: `requirements.txt`
- Create: `reports/.gitkeep`
- Create: `tests/__init__.py`
- Create: `scripts/__init__.py`

**Interfaces:**
- Consumes: None.
- Produces: Môi trường cài đặt sẵn `python-pptx`, `pytest` và các thư mục cần thiết.

- [ ] **Step 1: Viết file requirements.txt**

```text
python-pptx>=0.6.21
pytest>=7.0.0
```

- [ ] **Step 2: Tạo các file định vị thư mục**

Tạo `reports/.gitkeep`, `tests/__init__.py`, `scripts/__init__.py`.

- [ ] **Step 3: Cài đặt dependencies qua pip**

Run: `python -m pip install -r requirements.txt`
Expected: `Successfully installed python-pptx... pytest...` (hoặc `Requirement already satisfied`)

- [ ] **Step 4: Kiểm tra import thư viện thành công**

Run: `python -c "import pptx; import pytest; print('OK')"`
Expected: `OK`

---

### Task 2: Core Helpers & Theme Builder

**Files:**
- Create: `scripts/slide_helpers.py`
- Test: `tests/test_slide_helpers.py`

**Interfaces:**
- Consumes: `pptx`, `pptx.util`, `pptx.dml.color`, `pptx.enum.text`.
- Produces:
  - Theme constants: `COLOR_PRIMARY`, `COLOR_SECONDARY`, `COLOR_ACCENT`, `COLOR_BG`, `COLOR_WHITE`, `COLOR_MUTED`, `FONT_MAIN`.
  - Helper functions:
    - `init_presentation() -> Presentation` (khởi tạo presentation 16:9, layout blank)
    - `add_slide_header(slide, title_text: str, subtitle_text: str = "")`
    - `add_card(slide, left, top, width, height, bg_rgb=COLOR_WHITE, border_rgb=None)`
    - `add_bullet_points(tf, points: list[str], font_size=16, text_color=COLOR_PRIMARY)`

- [ ] **Step 1: Viết test cho helper khởi tạo presentation và màu sắc**

```python
# tests/test_slide_helpers.py
from pptx.util import Inches
from scripts.slide_helpers import (
    init_presentation,
    add_slide_header,
    add_card,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_BG
)

def test_init_presentation_dimensions():
    prs = init_presentation()
    assert abs(prs.slide_width.inches - 13.333) < 0.01
    assert abs(prs.slide_height.inches - 7.5) < 0.01

def test_add_slide_with_header():
    prs = init_presentation()
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    add_slide_header(slide, "Tiêu Đề Thử Nghiệm", "Phụ đề thử nghiệm")
    assert len(slide.shapes) >= 1
    assert "Tiêu Đề Thử Nghiệm" in slide.shapes[0].text_frame.text

def test_add_card():
    prs = init_presentation()
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    shape = add_card(slide, Inches(1), Inches(1), Inches(4), Inches(3))
    assert shape is not None
```

- [ ] **Step 2: Chạy test để xác nhận test FAIL (chưa có code)**

Run: `pytest tests/test_slide_helpers.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.slide_helpers'`

- [ ] **Step 3: Triển khai scripts/slide_helpers.py**

```python
# scripts/slide_helpers.py
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Theme Colors
COLOR_PRIMARY = RGBColor(15, 37, 55)     # #0F2537 Deep Navy
COLOR_SECONDARY = RGBColor(0, 168, 150)  # #00A896 Teal
COLOR_ACCENT = RGBColor(231, 111, 81)    # #E76F51 Coral
COLOR_BG = RGBColor(248, 249, 250)       # #F8F9FA Soft White/Gray
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_MUTED = RGBColor(100, 110, 120)

FONT_MAIN = "Segoe UI"
FONT_TITLE = "Segoe UI"

def init_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs

def set_slide_background(slide, color=COLOR_BG):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_slide_header(slide, title_text: str, subtitle_text: str = ""):
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_TITLE
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY

    if subtitle_text:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_MAIN
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = COLOR_MUTED
        p_sub.space_before = Pt(4)

    # Accent divider line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(1.65), Inches(2.0), Inches(0.04)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_SECONDARY
    line.line.color.rgb = COLOR_SECONDARY

def add_card(slide, left, top, width, height, bg_rgb=COLOR_WHITE, border_rgb=COLOR_SECONDARY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_rgb
    if border_rgb:
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape
```

- [ ] **Step 4: Chạy test xác nhận PASS**

Run: `pytest tests/test_slide_helpers.py`
Expected: `3 passed in ...s`

---

### Task 3: Triển khai Slide 1 đến Slide 4

**Files:**
- Modify: `scripts/slide_helpers.py` (bổ sung builders cho Slide 1-4)
- Create: `scripts/slide_builders_part1.py`
- Test: `tests/test_slide_part1.py`

**Interfaces:**
- Consumes: `init_presentation`, `add_slide_header`, `add_card`, color constants.
- Produces:
  - `build_slide_1_title(prs)`: Slide bìa tên đề tài, GVHD, HV
  - `build_slide_2_heterogeneity(prs)`: Bối cảnh & Dị thể dữ liệu ERP - CRM
  - `build_slide_3_objectives(prs)`: Mục tiêu nghiên cứu & Phạm vi
  - `build_slide_4_architecture(prs)`: Sơ đồ Pipeline 4 khối Preprocessing → Blocking → Matching → Fusion

- [ ] **Step 1: Viết test cho Slide 1-4**

```python
# tests/test_slide_part1.py
from scripts.slide_helpers import init_presentation
from scripts.slide_builders_part1 import (
    build_slide_1_title,
    build_slide_2_heterogeneity,
    build_slide_3_objectives,
    build_slide_4_architecture
)

def test_slides_1_to_4_creation():
    prs = init_presentation()
    build_slide_1_title(prs)
    build_slide_2_heterogeneity(prs)
    build_slide_3_objectives(prs)
    build_slide_4_architecture(prs)

    assert len(prs.slides) == 4
    # Slide 1: ĐT02
    assert "ĐT02" in prs.slides[0].shapes[0].text_frame.text or "ERP" in prs.slides[0].shapes[0].text_frame.text
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_slide_part1.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.slide_builders_part1'`

- [ ] **Step 3: Triển khai scripts/slide_builders_part1.py**

```python
# scripts/slide_builders_part1.py
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from scripts.slide_helpers import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_BG, COLOR_WHITE, COLOR_MUTED,
    FONT_MAIN, FONT_TITLE, set_slide_background, add_slide_header, add_card
)

def build_slide_1_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_PRIMARY)

    # Accent decorative box
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(5.0))
    banner.fill.solid()
    banner.fill.fore_color.rgb = COLOR_SECONDARY
    banner.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.2), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU ĐỀ TÀI (ĐỢT 1)"
    p0.font.name = FONT_MAIN
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SECONDARY

    p1 = tf.add_paragraph()
    p1.text = "Đối Sánh Và Hợp Nhất Dữ Liệu Khách Hàng\nGiữa ERP Và CRM Với Dữ Liệu Không Đồng Nhất"
    p1.font.name = FONT_TITLE
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "Mã đề tài: ĐT02 | Học phần: Quản trị Hệ thống Thông tin Thông minh (CSE631)"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_BG
    p2.space_before = Pt(16)

    p3 = tf.add_paragraph()
    p3.text = "Giảng viên hướng dẫn: TS. Nguyễn Khắc Chiến\nTrường Đại học Thủy Lợi — Khoa Công nghệ Thông tin"
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(14)
    p3.font.color.rgb = COLOR_SECONDARY
    p3.space_before = Pt(28)

def build_slide_2_heterogeneity(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "1. Bối Cảnh & Vấn Đề Dị Thể Dữ Liệu ERP – CRM",
        "Dữ liệu khách hàng phân mảnh, cấu trúc không đồng nhất và không có khóa định danh chung"
    )

    # Card 1: ERP View
    add_card(slide, Inches(0.8), Inches(1.9), Inches(3.6), Inches(5.0))
    tb1 = slide.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(3.4), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Hệ Thống ERP (Back-office)"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY

    bullets1 = [
        "Mục đích: Kế toán, hóa đơn, công nợ, xuất hàng.",
        "Đặc trưng dữ liệu: Định dạng pháp nhân chính quy, có Mã số thuế, địa chỉ trụ sở.",
        "Ví dụ mẫu:",
        "  • Tên: CÔNG TY TNHH CÔNG NGHỆ Á CHÂU",
        "  • MST: 0102345678",
        "  • ĐT: 024-3856-xxxx (Số bàn bàn phòng KT)",
        "  • Đ/c: Tầng 5, Tòa Keangnam, P. Mễ Trì, Q. Nam Từ Liêm, Hà Nội"
    ]
    for b in bullets1:
        p_b = tf1.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_PRIMARY
        p_b.space_before = Pt(6)

    # Card 2: CRM View
    add_card(slide, Inches(4.7), Inches(1.9), Inches(3.6), Inches(5.0))
    tb2 = slide.shapes.add_textbox(Inches(4.8), Inches(2.0), Inches(3.4), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Hệ Thống CRM (Front-office)"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_PRIMARY

    bullets2 = [
        "Mục đích: Tìm kiếm lead, sales pipeline, marketing.",
        "Đặc trưng: Gõ tự do, viết tắt, chú trọng liên hệ cá nhân, thường khuyết MST.",
        "Ví dụ mẫu:",
        "  • Tên: Cty A Chau Tech (Anh Minh)",
        "  • MST: (Trống / None)",
        "  • ĐT: 0912.345.xxx (Di động cá nhân)",
        "  • Đ/c: Keangnam Landmark, Từ Liêm, HN"
    ]
    for b in bullets2:
        p_b = tf2.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_PRIMARY
        p_b.space_before = Pt(6)

    # Card 3: Thách thức kỹ thuật
    add_card(slide, Inches(8.6), Inches(1.9), Inches(3.9), Inches(5.0), border_rgb=COLOR_ACCENT)
    tb3 = slide.shapes.add_textbox(Inches(8.7), Inches(2.0), Inches(3.7), Inches(4.7))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "Thách Thức Trọng Tâm"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = COLOR_ACCENT

    bullets3 = [
        "Không có Khóa chung: Không thể JOIN bảng theo ID hay Khóa chính thông thường.",
        "Dị thể cấu trúc & giá trị: Lỗi gõ tiếng Việt, viết tắt, hoán đổi thứ tự từ.",
        "Bùng nổ tổ hợp so sánh: N bản ghi ERP x M bản ghi CRM = O(NxM). 10k x 10k = 100 triệu phép so sánh.",
        "Hợp nhất xung đột: Quyết định giá trị chuẩn khi 2 nguồn lệch nhau (Data Fusion)."
    ]
    for b in bullets3:
        p_b = tf3.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_PRIMARY
        p_b.space_before = Pt(8)

def build_slide_3_objectives(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "2. Mục Tiêu Nghiên Cứu & Phạm Vi Đề Tài",
        "Xây dựng pipeline tự động hóa quá trình nhận diện và hợp nhất bản ghi khách hàng"
    )

    items = [
        ("Mục Tiêu Nghiên Cứu Kỹ Thuật", [
            "Làm chủ quy trình Entity Resolution (ER) chuyên sâu cho dữ liệu tiếng Việt có cấu trúc hành chính và doanh nghiệp.",
            "Triển khai cơ chế Blocking (chặn phân cụm) hiệu quả nhằm giảm tối đa không gian so sánh O(NxM).",
            "So sánh định lượng 2 hướng tiếp cận: Rule-based kết hợp hàm khoảng cách chuỗi vs Machine Learning Classifier.",
            "Hiện thực hóa module Data Fusion giải quyết xung đột thuộc tính để xuất bản ghi chuẩn (Golden Record)."
        ], Inches(0.8), Inches(1.9), Inches(5.7)),
        ("Phạm Vi & Sản Phẩm Đầu Ra", [
            "Phạm vi dữ liệu: Bảng thông tin khách hàng doanh nghiệp (B2B) và khách hàng cá nhân (B2C) giữa ERP và CRM.",
            "Bộ dữ liệu thực nghiệm: Sinh tự động có Ground Truth kiểm chứng độ chính xác.",
            "Mã nguồn hoàn chỉnh: Module xử lý Python + Ứng dụng Web Demo tương tác (Streamlit).",
            "Báo cáo học phần: Đánh giá chi tiết các chỉ số Precision, Recall, F1-Score và thời gian thực thi."
        ], Inches(6.8), Inches(1.9), Inches(5.7))
    ]

    for title, bullets, left, top, width in items:
        add_card(slide, left, top, width, Inches(5.0))
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(4.6))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(18)
        p.font.color.rgb = COLOR_PRIMARY

        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"✔ {b}"
            p_b.font.size = Pt(12.5)
            p_b.font.color.rgb = COLOR_PRIMARY
            p_b.space_before = Pt(12)

def build_slide_4_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "3. Kiến Trúc Pipeline Giải Pháp Đề Xuất",
        "Quy trình xử lý 4 giai đoạn khép kín từ dữ liệu thô dị thể đến Master Customer Record"
    )

    modules = [
        ("Module 1:\nTiền Xử Lý Chuỗi", "Làm sạch text, unidecode, chuẩn hóa viết tắt (Cty, TNHH), format SĐT E.164, chuẩn hóa địa chỉ.", COLOR_PRIMARY),
        ("Module 2:\nChặn Phân Cụm\n(Blocking)", "Lọc cặp tiềm năng theo MST / Tỉnh-Thành + First-letter. Giảm 95%+ tổ hợp so sánh O(NxM).", COLOR_SECONDARY),
        ("Module 3:\nĐo Tương Đồng\n& Phân Lớp", "Trích xuất vector tương đồng (Jaro-Winkler, Levenshtein, TF-IDF). Phân lớp Match / Non-match.", COLOR_PRIMARY),
        ("Module 4:\nHợp Nhất Dữ Liệu\n(Data Fusion)", "Giải quyết xung đột giá trị theo độ tin cậy nguồn (ERP ưu tiên thuế/pháp nhân, CRM ưu tiên liên hệ).", COLOR_SECONDARY)
    ]

    card_w = Inches(2.7)
    card_h = Inches(4.8)
    start_x = Inches(0.8)
    spacing = Inches(0.25)

    for i, (m_title, m_desc, border_c) in enumerate(modules):
        x = start_x + i * (card_w + spacing)
        add_card(slide, x, Inches(2.0), card_w, card_h, border_rgb=border_c)
        tb = slide.shapes.add_textbox(x + Inches(0.15), Inches(2.15), card_w - Inches(0.3), card_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = f"BƯỚC {i+1}"
        p0.font.bold = True
        p0.font.size = Pt(11)
        p0.font.color.rgb = COLOR_ACCENT

        p1 = tf.add_paragraph()
        p1.text = m_title
        p1.font.bold = True
        p1.font.size = Pt(16)
        p1.font.color.rgb = COLOR_PRIMARY
        p1.space_before = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = m_desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_MUTED
        p2.space_before = Pt(16)

        # Arrow indicator between steps (except last)
        if i < len(modules) - 1:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                x + card_w + Inches(0.04), Inches(4.2), Inches(0.18), Inches(0.25)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = COLOR_SECONDARY
            arrow.line.fill.background()
```

- [ ] **Step 4: Chạy test xác nhận PASS**

Run: `pytest tests/test_slide_part1.py`
Expected: `1 passed in ...s`

---

### Task 4: Triển khai Slide 5 đến Slide 8

**Files:**
- Create: `scripts/slide_builders_part2.py`
- Test: `tests/test_slide_part2.py`

**Interfaces:**
- Consumes: `init_presentation`, `add_slide_header`, `add_card`, color constants.
- Produces:
  - `build_slide_5_experiment(prs)`: Thiết kế thử nghiệm & Lược đồ dữ liệu
  - `build_slide_6_progress(prs)`: Kết quả tiến độ hiện tại (checklist hoàn thành)
  - `build_slide_7_roadmap(prs)`: Kế hoạch thực hiện các giai đoạn tiếp theo (Timeline)
  - `build_slide_8_conclusion(prs)`: Kết luận & Xin ý kiến GVHD

- [ ] **Step 1: Viết test cho Slide 5-8**

```python
# tests/test_slide_part2.py
from scripts.slide_helpers import init_presentation
from scripts.slide_builders_part2 import (
    build_slide_5_experiment,
    build_slide_6_progress,
    build_slide_7_roadmap,
    build_slide_8_conclusion
)

def test_slides_5_to_8_creation():
    prs = init_presentation()
    build_slide_5_experiment(prs)
    build_slide_6_progress(prs)
    build_slide_7_roadmap(prs)
    build_slide_8_conclusion(prs)

    assert len(prs.slides) == 4
    # Slide 8 conclusion text verification
    assert "Ý Kiến" in prs.slides[3].shapes[0].text_frame.text or "Kết Luận" in prs.slides[3].shapes[0].text_frame.text
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_slide_part2.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.slide_builders_part2'`

- [ ] **Step 3: Triển khai scripts/slide_builders_part2.py**

```python
# scripts/slide_builders_part2.py
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from scripts.slide_helpers import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_BG, COLOR_WHITE, COLOR_MUTED,
    FONT_MAIN, FONT_TITLE, set_slide_background, add_slide_header, add_card
)

def build_slide_5_experiment(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "4. Thiết Kế Thử Nghiệm & Lược Đồ Dữ Liệu",
        "Phương án thiết kế schema và cơ chế sinh tập dữ liệu đối sánh có nhãn chân thực"
    )

    # Left: Schema comparison table/card
    add_card(slide, Inches(0.8), Inches(1.9), Inches(5.7), Inches(5.0))
    tb_l = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Lược Đồ Thuộc Tính Thực Nghiệm"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY

    schema_details = [
        "Lược đồ ERP: [erp_id, tax_code, company_name, representative, billing_address, phone, created_date]",
        "Lược đồ CRM: [crm_id, lead_name, contact_person, email, mobile_phone, shipping_address, last_contact]",
        "Đặc điểm:",
        "  • Không có khóa ngoại trực tiếp giữa 2 bảng.",
        "  • Tỷ lệ trùng thực tế được thiết lập: ~40% - 60% số bản ghi.",
        "  • Có bảng Ground Truth: (erp_id, crm_id, is_match) đóng vai trò nhãn chuẩn đánh giá thuật toán."
    ]
    for item in schema_details:
        p_item = tf_l.add_paragraph()
        p_item.text = item
        p_item.font.size = Pt(12)
        p_item.font.color.rgb = COLOR_PRIMARY
        p_item.space_before = Pt(8)

    # Right: Data generation & Noise injection
    add_card(slide, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0))
    tb_r = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Cơ Chế Gieo Nhiễu Thực Tế (Noise Injection)"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = COLOR_PRIMARY

    noises = [
        "Sinh dữ liệu gốc bằng Faker (tiếng Việt vi_VN) cho tập khách hàng cơ sở.",
        "Nhiễu gõ phím tiếng Việt: Sai dấu Telex/VNI (Á Châu ↔ As Châu, Thủy Lợi ↔ Thuyr Lợi).",
        "Nhiễu viết tắt & từ đồng nghĩa: Cty ↔ Công ty, TNHH ↔ Trách nhiệm hữu hạn, CP ↔ Cổ phần.",
        "Nhiễu cấu trúc địa chỉ: Lược bớt tiền tố 'Phường/Quận/TP', hoán đổi vị trí số nhà.",
        "Nhiễu số điện thoại: Biến thiên +84 / 09xx / dấu chấm phân tách / khoảng trắng."
    ]
    for n in noises:
        p_n = tf_r.add_paragraph()
        p_n.text = f"• {n}"
        p_n.font.size = Pt(12)
        p_n.font.color.rgb = COLOR_PRIMARY
        p_n.space_before = Pt(8)

def build_slide_6_progress(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "5. Kết Quả Tiến Độ Hiện Tại",
        "Những nội dung đã hoàn thành phục vụ cơ sở khoa học và phương pháp luận của đề tài"
    )

    done_items = [
        ("Nghiên Cứu Lý Thuyết Chuyên Sâu", [
            "Hệ thống hóa bài toán Entity Resolution trong ngữ cảnh doanh nghiệp ERP & CRM.",
            "Khảo sát các kỹ thuật đo khoảng cách chuỗi phù hợp đặc thù tiếng Việt (Jaro-Winkler, Levenshtein, N-gram).",
            "Phân tích cơ chế giải quyết xung đột dữ liệu (Data Fusion Strategies)."
        ]),
        ("Lập Lộ Trình Nghiên Cứu Chi Tiết", [
            "Hoàn thiện văn bản lộ trình kỹ thuật 6 giai đoạn (LO_TRINH_NGHIEN_CUU.md).",
            "Phân bổ mốc thời gian bám sát đề cương học phần CSE631 và yêu cầu nộp bài.",
            "Xác định rõ ràng tiêu chí đánh giá định lượng (Precision, Recall, F1, Reduction Ratio)."
        ]),
        ("Thiết Kế Kiến Trúc & Kịch Bản Thực Nghiệm", [
            "Hoàn tất đặc tả Schema dữ liệu mô phỏng hai hệ thống ERP và CRM.",
            "Thiết kế thuật toán tạo dữ liệu có nhiễu chủ đích và bảng nhãn kiểm chứng (Ground Truth).",
            "Định hình cấu trúc module hóa mã nguồn (`src/`) và giao diện tương tác Streamlit."
        ])
    ]

    card_w = Inches(3.7)
    card_h = Inches(5.0)
    for i, (title, points) in enumerate(done_items):
        x = Inches(0.8) + i * Inches(4.0)
        add_card(slide, x, Inches(1.9), card_w, card_h, border_rgb=COLOR_SECONDARY)
        tb = slide.shapes.add_textbox(x + Inches(0.15), Inches(2.05), card_w - Inches(0.3), card_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = "✓ HOÀN THÀNH"
        p0.font.bold = True
        p0.font.size = Pt(11)
        p0.font.color.rgb = COLOR_SECONDARY

        p1 = tf.add_paragraph()
        p1.text = title
        p1.font.bold = True
        p1.font.size = Pt(16)
        p1.font.color.rgb = COLOR_PRIMARY
        p1.space_before = Pt(4)

        for pt in points:
            p_pt = tf.add_paragraph()
            p_pt.text = f"• {pt}"
            p_pt.font.size = Pt(12)
            p_pt.font.color.rgb = COLOR_PRIMARY
            p_pt.space_before = Pt(10)

def build_slide_7_roadmap(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_BG)
    add_slide_header(
        slide,
        "6. Kế Hoạch Thực Hiện Giai Đoạn Tiếp Theo",
        "Lộ trình triển khai cụ thể các tuần tới nhằm hoàn tất mã nguồn, thực nghiệm và báo cáo"
    )

    steps = [
        ("Tuần 3", "Sinh Dữ Liệu Thực Nghiệm", "Viết script Python sinh bộ dữ liệu ERP & CRM mẫu có nhiễu và file nhãn chuẩn Ground Truth."),
        ("Tuần 4", "Phát Triển Core Pipeline", "Hiện thực hóa 4 module Python: Preprocessor, Blocking, Matcher (Rule + ML) và Data Fusion."),
        ("Tuần 5 - 6", "Đánh Giá Thực Nghiệm & Web App", "Đo đạc Precision/Recall/F1; phát triển giao diện trực quan hóa kết quả bằng Streamlit."),
        ("Tuần 7", "Hoàn Thiện Báo Cáo & Slide Bảo Vệ", "Viết báo cáo chuyên đề Word theo quy chuẩn ĐHTL; hoàn thiện Slide và kịch bản vấn đáp.")
    ]

    for i, (time_tag, step_title, step_desc) in enumerate(steps):
        y = Inches(1.9) + i * Inches(1.25)
        card = add_card(slide, Inches(0.8), y, Inches(11.733), Inches(1.1), border_rgb=COLOR_SECONDARY)

        tb = slide.shapes.add_textbox(Inches(1.0), y + Inches(0.1), Inches(11.3), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = f"{time_tag}  |  {step_title}"
        p0.font.bold = True
        p0.font.size = Pt(16)
        p0.font.color.rgb = COLOR_PRIMARY

        p1 = tf.add_paragraph()
        p1.text = step_desc
        p1.font.size = Pt(12.5)
        p1.font.color.rgb = COLOR_MUTED
        p1.space_before = Pt(3)

def build_slide_8_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, COLOR_PRIMARY)

    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.0), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "KẾT LUẬN & XIN Ý KIẾN HƯỚNG DẪN"
    p0.font.name = FONT_TITLE
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_WHITE

    p1 = tf.add_paragraph()
    p1.text = "Đề tài đã xác định rõ nét bài toán thực tế, cơ sở lý thuyết vững chắc và cấu trúc kỹ thuật khả thi."
    p1.font.name = FONT_MAIN
    p1.font.size = Pt(15)
    p1.font.color.rgb = COLOR_SECONDARY
    p1.space_before = Pt(14)

    questions_header = tf.add_paragraph()
    questions_header.text = "Các nội dung xin ý kiến chỉ đạo từ Thầy TS. Nguyễn Khắc Chiến:"
    questions_header.font.name = FONT_MAIN
    questions_header.font.size = Pt(16)
    questions_header.font.bold = True
    questions_header.font.color.rgb = COLOR_WHITE
    questions_header.space_before = Pt(24)

    qs = [
        "1. Quy mô tập dữ liệu thực nghiệm đề xuất (1.000 vs 10.000 bản ghi) để vừa đảm bảo tính khoa học vừa tối ưu thời gian chạy trên máy cá nhân.",
        "2. Đánh giá tính phù hợp của việc kết hợp mô hình học máy (Random Forest / Logistic Regression) so với phương pháp heuristic trọng số truyền thống.",
        "3. Các tiêu chí bổ sung cho quy tắc hợp nhất dữ liệu (Data Fusion) đối với đặc thù doanh nghiệp Việt Nam."
    ]
    for q in qs:
        p_q = tf.add_paragraph()
        p_q.text = q
        p_q.font.name = FONT_MAIN
        p_q.font.size = Pt(13)
        p_q.font.color.rgb = COLOR_BG
        p_q.space_before = Pt(8)

    p_thanks = tf.add_paragraph()
    p_thanks.text = "EM XIN TRÂN TRỌNG CẢM ƠN THẦY!"
    p_thanks.font.name = FONT_TITLE
    p_thanks.font.size = Pt(18)
    p_thanks.font.bold = True
    p_thanks.font.color.rgb = COLOR_ACCENT
    p_thanks.space_before = Pt(24)
```

- [ ] **Step 4: Chạy test xác nhận PASS**

Run: `pytest tests/test_slide_part2.py`
Expected: `1 passed in ...s`

---

### Task 5: Script Điều Phối Tổng Thể & Xuất File PPTX

**Files:**
- Create: `scripts/generate_slides.py`
- Test: `tests/test_generate_slides_e2e.py`
- Output: `reports/Slide_Bao_Cao_Tien_Do.pptx`

**Interfaces:**
- Consumes: `init_presentation`, `build_slide_1_title` -> `build_slide_8_conclusion`.
- Produces: File `reports/Slide_Bao_Cao_Tien_Do.pptx` hoàn chỉnh 8 slide.

- [ ] **Step 1: Viết test End-to-End cho file output**

```python
# tests/test_generate_slides_e2e.py
import os
from pptx import Presentation
from scripts.generate_slides import generate_all_slides, OUTPUT_PATH

def test_generate_all_slides_file_created():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    output_file = generate_all_slides()
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 15000  # lớn hơn 15KB

    # Kiểm tra load lại file pptx thành công
    prs = Presentation(output_file)
    assert len(prs.slides) == 8
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_generate_slides_e2e.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_slides'`

- [ ] **Step 3: Triển khai scripts/generate_slides.py**

```python
# scripts/generate_slides.py
import os
from scripts.slide_helpers import init_presentation
from scripts.slide_builders_part1 import (
    build_slide_1_title,
    build_slide_2_heterogeneity,
    build_slide_3_objectives,
    build_slide_4_architecture
)
from scripts.slide_builders_part2 import (
    build_slide_5_experiment,
    build_slide_6_progress,
    build_slide_7_roadmap,
    build_slide_8_conclusion
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Slide_Bao_Cao_Tien_Do.pptx")

def generate_all_slides(output_path=OUTPUT_PATH) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prs = init_presentation()
    print("-> Đang tạo Slide 1: Tiêu đề...")
    build_slide_1_title(prs)

    print("-> Đang tạo Slide 2: Bối cảnh & Dị thể dữ liệu ERP - CRM...")
    build_slide_2_heterogeneity(prs)

    print("-> Đang tạo Slide 3: Mục tiêu & Phạm vi...")
    build_slide_3_objectives(prs)

    print("-> Đang tạo Slide 4: Kiến trúc Pipeline 4 Module...")
    build_slide_4_architecture(prs)

    print("-> Đang tạo Slide 5: Thiết kế thử nghiệm...")
    build_slide_5_experiment(prs)

    print("-> Đang tạo Slide 6: Kết quả tiến độ hiện tại...")
    build_slide_6_progress(prs)

    print("-> Đang tạo Slide 7: Kế hoạch các giai đoạn tiếp theo...")
    build_slide_7_roadmap(prs)

    print("-> Đang tạo Slide 8: Kết luận & Xin ý kiến GVHD...")
    build_slide_8_conclusion(prs)

    prs.save(output_path)
    print(f"✓ Đã xuất file thành công: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_all_slides()
```

- [ ] **Step 4: Chạy script trực tiếp để sinh file slide**

Run: `python scripts/generate_slides.py`
Expected: `✓ Đã xuất file thành công: ...reports\Slide_Bao_Cao_Tien_Do.pptx`

- [ ] **Step 5: Chạy toàn bộ test suite để xác nhận 100% PASS**

Run: `pytest tests/`
Expected: `5 passed in ...s`
