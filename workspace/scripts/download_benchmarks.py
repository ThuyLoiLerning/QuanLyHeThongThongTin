# workspace/scripts/download_benchmarks.py
import os
import sys
import urllib.request
import zipfile
import shutil

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DATA_DIR = os.path.join(ROOT_DIR, "experiments", "shared_data")
PAPERS_DIR = os.path.join(ROOT_DIR, "experiments", "papers")

AMAZON_GOOGLE_ZIP_URL = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_exp_data.zip"
DBLP_ACM_ZIP_URL = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip"

PAPER_URLS = {
    "Mudgal_SIGMOD2018.pdf": "https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf",
    "Ditto_PVLDB2020.pdf": "https://arxiv.org/pdf/2004.00584.pdf"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[SKIP] Already exists: {target_path}")
        return True
    print(f"[DOWNLOAD] {url} -> {target_path}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return True


def setup_amazon_google():
    ag_dir = os.path.join(SHARED_DATA_DIR, "Amazon-Google")
    os.makedirs(ag_dir, exist_ok=True)
    zip_path = os.path.join(ag_dir, "amazon_google_exp_data.zip")

    download_file(AMAZON_GOOGLE_ZIP_URL, zip_path)
    print(f"[EXTRACT] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ag_dir)

    # DeepMatcher archives extract into exp_data/ subfolder
    exp_sub = os.path.join(ag_dir, "exp_data")
    if os.path.exists(exp_sub):
        for f in os.listdir(exp_sub):
            shutil.move(os.path.join(exp_sub, f), os.path.join(ag_dir, f))
        os.rmdir(exp_sub)
    print(f"[OK] Amazon-Google dataset prepared at {ag_dir}")


def setup_papers():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    for name, url in PAPER_URLS.items():
        target = os.path.join(PAPERS_DIR, name)
        try:
            download_file(url, target)
        except Exception as e:
            print(f"[WARN] Failed to download {name} from {url}: {e}")
            # Fallback to local copy from Old/ if available
            if name == "Ditto_PVLDB2020.pdf":
                local_ditto = os.path.join(ROOT_DIR, "Old", "BaiBao", "Bai1_Ditto_PVLDB2020", "paper_arxiv.pdf")
                if os.path.exists(local_ditto):
                    shutil.copyfile(local_ditto, target)
                    print(f"[FALLBACK] Copied local Ditto paper to {target}")


def main():
    print("=== TASK 1: DOWNLOADING BENCHMARKS & PAPERS ===")
    setup_amazon_google()
    setup_papers()
    print("=== TASK 1 COMPLETE ===")


if __name__ == "__main__":
    main()
