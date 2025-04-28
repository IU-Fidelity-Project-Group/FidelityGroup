import json
from datetime import datetime

INPUT_FILE = r"D:\cisa_kev.json"
OUTPUT_FILE = r"D:\cisa_kev_enriched.json"

def build_embedding_text(entry):
    lines = [
        f"{entry.get('cveID', 'Unknown CVE')}: {entry.get('vulnerabilityName', '')}",
        entry.get('shortDescription') or entry.get('description', ''),
        f"Vendor: {entry.get('vendorProject', 'Unknown')}",
        f"Product: {entry.get('product', 'Unknown')}",
        f"Required Action: {entry.get('requiredAction', 'None listed.')}",
        f"Known Ransomware Use: {entry.get('knownRansomwareCampaignUse', 'Unknown')}",
        f"Due Date: {entry.get('dueDate', 'N/A')}",
        f"Notes: {entry.get('notes', 'None.')}",
    ]
    return " ".join(lines).strip()

def enrich_kev(input_path, output_path):
    print(f"📂 Loading KEV entries from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched_data = []
    for entry in data:
        entry["embedding_text"] = build_embedding_text(entry)
        enriched_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"✅ Enriched KEV saved to: {output_path}")
    print(f"📦 Total entries processed: {len(enriched_data)}")

if __name__ == "__main__":
    enrich_kev(INPUT_FILE, OUTPUT_FILE)
