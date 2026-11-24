"""
Generate realistic ERP and CRM datasets with ground truth from 1,000 real VN enterprises.
Simulates real-world heterogeneity:
  - Schema heterogeneity
  - Value heterogeneity (abbreviations, typos, missing tax IDs, address shortening)
  - Hard negative pairs (similar name but different Tax ID)
Seed: 1001 (reproducible protocol)
"""
import sys
import os
import re
import random
import csv
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 1001
random.seed(SEED)

RAW_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "enterprises_raw.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ERP_FILE = os.path.join(PROCESSED_DIR, "erp_customers.csv")
CRM_FILE = os.path.join(PROCESSED_DIR, "crm_leads.csv")
GROUND_TRUTH_FILE = os.path.join(PROCESSED_DIR, "ground_truth.csv")


def corrupt_company_name(name):
    """Simulate sales/CRM shorthand, prefix omission, title case, and minor typos."""
    corrupted = name

    # Prefix replacements
    prefixes = [
        ("CÔNG TY TNHH MTV", ["Cty TNHH 1TV", "TNHH MTV", "Công ty TNHH MTV", "Cty"]),
        ("CÔNG TY TNHH", ["Cty TNHH", "TNHH", "Công ty TNHH", "Cty"]),
        ("CÔNG TY CỔ PHẦN", ["Cty CP", "Công ty CP", "CTCP", "Cty Cổ phần"]),
        ("HỢP TÁC XÃ", ["HTX", "Hợp tác xã"]),
        ("DOANH NGHIỆP TƯ NHÂN", ["DNTN", "Doanh nghiệp tư nhân"])
    ]
    for old, reps in prefixes:
        if corrupted.startswith(old):
            chosen = random.choice(reps)
            corrupted = corrupted.replace(old, chosen, 1)
            break

    # 40% change to Title Case
    if random.random() < 0.4:
        corrupted = corrupted.title()

    # 15% drop legal prefix completely (e.g. 'Hải Sản Hồng Nhất' instead of 'Cty TNHH Hải Sản Hồng Nhất')
    if random.random() < 0.15:
        tokens = corrupted.split()
        if len(tokens) > 3:
            corrupted = " ".join(tokens[2:])

    # 10% small typo (duplicate character or missing space)
    if random.random() < 0.1 and len(corrupted) > 10:
        idx = random.randint(3, len(corrupted) - 2)
        if corrupted[idx] != " ":
            corrupted = corrupted[:idx] + corrupted[idx] + corrupted[idx:]

    return corrupted.strip()


def corrupt_address(addr):
    """Simulate address abbreviation and truncation in CRM."""
    c_addr = addr
    replacements = [
        (r"\bPhường\b", "P."),
        (r"\bQuận\b", "Q."),
        (r"\bThành phố\b", "TP"),
        (r"\bTp Hồ Chí Minh\b", "TP.HCM"),
        (r"\bTp Hà Nội\b", "HN"),
        (r"\bHuyện\b", "H."),
        (r"\bXã\b", "X."),
        (r"\bThị xã\b", "TX.")
    ]
    for pat, rep in replacements:
        c_addr = re.sub(pat, rep, c_addr, flags=re.IGNORECASE)

    # 25% truncate: take only first 2 parts (street and ward)
    if random.random() < 0.25 and "," in c_addr:
        parts = [p.strip() for p in c_addr.split(",")]
        if len(parts) >= 3:
            c_addr = ", ".join(parts[:2])

    return c_addr.strip()


def generate_phone(is_crm=False):
    prefix = random.choice(["090", "091", "098", "097", "093", "088", "079"])
    digits = f"{random.randint(1000000, 9999999)}"
    raw = prefix + digits
    if not is_crm:
        return raw
    # CRM formatting variance
    fmt = random.choice(["raw", "intl", "dots", "spaces"])
    if fmt == "intl":
        return f"+84{raw[1:]}"
    elif fmt == "dots":
        return f"{raw[:4]}.{raw[4:7]}.{raw[7:]}"
    elif fmt == "spaces":
        return f"{raw[:4]} {raw[4:7]} {raw[7:]}"
    return raw


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_raw = pd.read_csv(RAW_FILE)
    print(f"Đọc {len(df_raw)} bản ghi từ {RAW_FILE}")

    erp_rows = []
    crm_rows = []
    ground_truth = []

    # 1. Build ERP table (800 records)
    # Using 800 for ERP, so we have records that match and records that don't
    n_erp = 800
    df_erp_base = df_raw.iloc[:n_erp].copy()

    for i, row in df_erp_base.iterrows():
        erp_id = f"ERP_{i+1:04d}"
        erp_rows.append({
            "erp_id": erp_id,
            "company_name": str(row["company_name"]).strip(),
            "tax_id": str(row["tax_id"]).strip(),
            "representative": str(row["representative"]).strip() if pd.notna(row["representative"]) else "",
            "billing_address": str(row["address"]).strip(),
            "phone": generate_phone(is_crm=False)
        })

    # 2. Build CRM table (800 records total: 600 True Matches + 100 Hard Negatives + 100 Unmatched CRM leads)
    crm_counter = 1

    # Group A: 600 True Matches (same underlying entity, corrupted attributes)
    for i in range(600):
        erp_item = erp_rows[i]
        crm_id = f"CRM_{crm_counter:04d}"
        crm_counter += 1

        # Corrupt name and address
        c_name = corrupt_company_name(erp_item["company_name"])
        c_addr = corrupt_address(erp_item["billing_address"])

        # Contact person: 30% add title like "Anh / Chị"
        rep = erp_item["representative"]
        if rep and random.random() < 0.35:
            rep = random.choice(["Anh ", "Chị "]) + rep.split()[-1]

        # Tax ID: 35% MISSING (realistic sales scenario), 65% retained
        if random.random() < 0.35:
            c_tax = ""
        else:
            c_tax = erp_item["tax_id"]

        crm_rows.append({
            "crm_id": crm_id,
            "lead_name": c_name,
            "tax_id": c_tax,
            "contact_person": rep,
            "shipping_address": c_addr,
            "mobile_phone": generate_phone(is_crm=True)
        })

        ground_truth.append({
            "erp_id": erp_item["erp_id"],
            "crm_id": crm_id,
            "label": 1,
            "relation_type": "true_match_missing_tax" if c_tax == "" else "true_match"
        })

    # Group B: 100 Hard Negatives (Company names share strong tokens, but DIFFERENT tax IDs)
    # E.g. pair ERP_i with CRM having modified name from ERP_i, but DIFFERENT Tax ID from another firm!
    for i in range(600, 700):
        erp_item = erp_rows[i]
        crm_id = f"CRM_{crm_counter:04d}"
        crm_counter += 1

        # Different enterprise for tax ID
        diff_enterprise = df_raw.iloc[800 + (i - 600)]

        # Name is very similar (e.g. branch or subsidiary)
        tokens = erp_item["company_name"].split()
        if len(tokens) > 3:
            fake_branch_name = " ".join(tokens[:-1]) + " - CHI NHÁNH " + random.choice(["HÀ NỘI", "MIỀN BẮC", "SÀI GÒN", "ĐÀ NẴNG"])
        else:
            fake_branch_name = erp_item["company_name"] + " PHÁT TRIỂN"

        crm_rows.append({
            "crm_id": crm_id,
            "lead_name": corrupt_company_name(fake_branch_name),
            "tax_id": str(diff_enterprise["tax_id"]).strip(),  # VALID BUT DIFFERENT TAX ID!
            "contact_person": str(diff_enterprise["representative"]).strip() if pd.notna(diff_enterprise["representative"]) else "",
            "shipping_address": corrupt_address(str(diff_enterprise["address"])),
            "mobile_phone": generate_phone(is_crm=True)
        })

        # Ground truth: MUST NOT MATCH! label = 0
        ground_truth.append({
            "erp_id": erp_item["erp_id"],
            "crm_id": crm_id,
            "label": 0,
            "relation_type": "hard_negative_different_tax"
        })

    # Group C: 100 Unmatched CRM leads (independent real companies from index 900-999)
    for i in range(900, 1000):
        raw_ent = df_raw.iloc[i]
        crm_id = f"CRM_{crm_counter:04d}"
        crm_counter += 1

        crm_rows.append({
            "crm_id": crm_id,
            "lead_name": corrupt_company_name(str(raw_ent["company_name"])),
            "tax_id": str(raw_ent["tax_id"]).strip() if random.random() > 0.3 else "",
            "contact_person": str(raw_ent["representative"]).strip() if pd.notna(raw_ent["representative"]) else "",
            "shipping_address": corrupt_address(str(raw_ent["address"])),
            "mobile_phone": generate_phone(is_crm=True)
        })

    # Save ERP
    df_erp = pd.DataFrame(erp_rows)
    df_erp.to_csv(ERP_FILE, index=False, encoding="utf-8-sig")

    # Save CRM
    df_crm = pd.DataFrame(crm_rows)
    df_crm.to_csv(CRM_FILE, index=False, encoding="utf-8-sig")

    # Save Ground Truth
    df_gt = pd.DataFrame(ground_truth)
    df_gt.to_csv(GROUND_TRUTH_FILE, index=False, encoding="utf-8-sig")

    print(f"Hoàn thành sinh dữ liệu chuẩn:")
    print(f"  - ERP: {len(df_erp)} bản ghi -> {ERP_FILE}")
    print(f"  - CRM: {len(df_crm)} bản ghi -> {CRM_FILE}")
    print(f"  - Ground Truth: {len(df_gt)} cặp gán nhãn kiểm chứng -> {GROUND_TRUTH_FILE}")
    print(f"    + True Matches: {(df_gt['label'] == 1).sum()} cặp")
    print(f"    + Hard Negatives (trùng tên khác MST): {(df_gt['relation_type'] == 'hard_negative_different_tax').sum()} cặp")


if __name__ == "__main__":
    main()
