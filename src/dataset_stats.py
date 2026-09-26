import json
from pathlib import Path
from collections import Counter

DATASET_FILE = Path("data/layoutlm_dataset.jsonl")


def analyze_dataset():
    total_samples = 0
    field_found_counts = Counter()  # zählt, in wie vielen Belegen ein Feld mind. 1x vorkommt
    label_type_counts = Counter()   # zählt, wie oft jedes Label insgesamt vorkommt (über alle Wörter)

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            total_samples += 1

            labels_in_sample = set(sample["labels"])

            for field in ["COMPANY", "DATE", "TOTAL"]:
                if f"B-{field}" in labels_in_sample:
                    field_found_counts[field] += 1

            for label in sample["labels"]:
                label_type_counts[label] += 1

    print(f"Gesamtzahl aufbereiteter Belege: {total_samples}\n")

    print("=== Wie viele Belege enthalten mindestens 1x dieses Feld? ===")
    for field in ["COMPANY", "DATE", "TOTAL"]:
        count = field_found_counts[field]
        percentage = (count / total_samples) * 100
        print(f"{field:10s}: {count}/{total_samples} ({percentage:.1f}%)")

    print("\n=== Verteilung aller Labels (über alle Wörter hinweg) ===")
    for label, count in sorted(label_type_counts.items()):
        print(f"{label:15s}: {count}")


if __name__ == "__main__":
    analyze_dataset()