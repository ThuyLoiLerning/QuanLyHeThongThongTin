"""
Crawl 1,000 real Vietnamese enterprises from public directory (doanhnghiep.biz).
Collects: Tax ID (MST), Company Name, Representative, Address.
Uses concurrent requests for speed.
"""
import sys
import os
import re
import csv
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

TARGET_COUNT = 1000
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "enterprises_raw.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


import time

def fetch_page(page_num, retries=5):
    url = f"https://doanhnghiep.biz/?p={page_num}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                return page_num, html
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return page_num, None


def parse_page(html):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    records = []

    mst_links = soup.find_all("a", title=re.compile(r"Tra cứu Số ĐKKD/MST"))
    for l in mst_links:
        parent = l.find_parent("div")
        if not parent:
            continue

        mst = l.text.strip()
        if not re.match(r"^\d{10}(-\d{3})?$", mst):
            continue

        h6 = parent.find("h6")
        name = h6.text.strip() if h6 else ""
        if not name:
            continue

        rep_tag = parent.find("a", title=re.compile(r"giám đốc"))
        rep = rep_tag.text.strip() if rep_tag else ""

        addr_tag = parent.find("address")
        addr = addr_tag.text.strip() if addr_tag else ""
        addr = re.sub(r"\s+", " ", addr)

        records.append({
            "tax_id": mst,
            "company_name": name,
            "representative": rep,
            "address": addr
        })
    return records


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_records = []
    seen_mst = set()

    total_pages = 40  # 40 pages * 30 items = 1,200 candidates
    print(f"Đang thu thập dữ liệu song song qua {total_pages} trang...")

    pages_html = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_page, p): p for p in range(1, total_pages + 1)}
        for future in as_completed(futures):
            p_num, html = future.result()
            if html:
                pages_html[p_num] = html

    # Process in page order
    for p_num in sorted(pages_html.keys()):
        items = parse_page(pages_html[p_num])
        for it in items:
            if it["tax_id"] not in seen_mst:
                seen_mst.add(it["tax_id"])
                all_records.append(it)
                if len(all_records) >= TARGET_COUNT:
                    break
        if len(all_records) >= TARGET_COUNT:
            break

    print(f"Thu thập thành công {len(all_records)} doanh nghiệp thật.")

    # Save CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tax_id", "company_name", "representative", "address"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Đã lưu vào: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
