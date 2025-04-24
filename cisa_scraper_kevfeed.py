import requests
import json
import os
import hashlib
from datetime import datetime

# File paths
OUTPUT_FILE = r"D:\cisa_kev.json"
HASH_CACHE_FILE = r"D:\cisa_kev_seen.json"

# CISA KEV JSON feed
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

def load_seen_hashes():
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_hashes(hashes):
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(list(hashes), f)

def hash_entry(entry):
    # Hash by CVE ID + vendor + product for deduplication
    identifier = f"{entry.get('cveID', '')}_{entry.get('vendorProject', '')}_{entry.get('product', '')}"
    return hashlib.sha256(identifier.encode()).hexdigest()

def fetch_kev(limit=None):
    print("\n📡 Fetching CISA Known Exploited Vulnerabilities catalog...")

    try:
        response = requests.get(KEV_URL)
        response.raise_for_status()
        kev_data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch KEV feed: {e}")
        return []

    vulns = kev_data.get("vulnerabilities", [])
    if limit:
        vulns = vulns[:limit]

    seen_hashes = load_seen_hashes()
    new_hashes = set()
    new_vulns = []

    for vuln in vulns:
        hash_key = hash_entry(vuln)
        if hash_key in seen_hashes:
            continue
        vuln["date_scraped"] = datetime.utcnow().isoformat()
        new_vulns.append(vuln)
        new_hashes.add(hash_key)

    if new_vulns:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(new_vulns, f, indent=2)
        save_seen_hashes(seen_hashes.union(new_hashes))
        print(f"✅ Saved {len(new_vulns)} new vulnerabilities to: {OUTPUT_FILE}")
    else:
        print("⚠️ No new vulnerabilities found.")

    return new_vulns

if __name__ == "__main__":
    fetch_kev(limit=None)  # Or set a limit like limit=50
