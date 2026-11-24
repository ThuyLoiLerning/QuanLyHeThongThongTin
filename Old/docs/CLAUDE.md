# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

- **Học phần:** CSE631 - Quản Trị Hệ Thống Thông Tin Thông Minh (Trình độ Thạc sĩ - Đại học Thủy Lợi)
- **GVHD:** TS. Nguyễn Khắc Chiến
- **Mã đề tài:** ĐT02
- **Tên đề tài:** Đối sánh và hợp nhất dữ liệu khách hàng giữa ERP và CRM với dữ liệu không đồng nhất (ERP - CRM Entity Matching / Data Integration)
- **Trọng tâm:** ERP – CRM, Entity Resolution / Record Linkage, Data Heterogeneity

## Current State

Đã hoàn thành chuẩn bị tài liệu báo cáo tiến độ và mã nguồn sinh slide:
- `LO_TRINH_NGHIEN_CUU.md`: Lộ trình nghiên cứu chi tiết 6 giai đoạn
- `docs/superpowers/specs/2026-11-14-slide-progress-report-design.md`: Bản thiết kế chi tiết 8 slide báo cáo tiến độ
- `docs/superpowers/plans/2026-11-14-slide-progress-report.md`: Kế hoạch triển khai kỹ thuật
- `scripts/`: Chứa mã nguồn sinh slide bằng `python-pptx` (`generate_slides.py`, `slide_helpers.py`, `slide_builders_part1.py`, `slide_builders_part2.py`)
- `tests/`: Bộ test kiểm thử cho các module sinh slide
- `reports/Slide_Bao_Cao_Tien_Do.md`: Bản nội dung slide định dạng Marp Markdown (chuẩn 16:9, màu Navy/Teal)

## Commands

- Cài đặt thư viện: `pip install -r requirements.txt`
- Sinh file PowerPoint: `python scripts/generate_slides.py` (tạo `reports/Slide_Bao_Cao_Tien_Do.pptx`)
- Chạy unit test: `pytest tests/`

## When Code Is Added

Update this file with:
- Build/run/test commands
- Architecture overview
- Key dependencies and conventions

