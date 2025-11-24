import json
import pandas as pd
from pathlib import Path
import argparse


def load_json_files(folder: Path):
    """Reads all .json files from a folder and returns list of dictionaries."""
    folder = folder.resolve()

    if not folder.exists():
        raise RuntimeError(f"Folder not found: {folder}")

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in folder: {folder}")

    data = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                obj = json.load(fp)
                obj["_source_file"] = f.name   # track JSON filename
                data.append(obj)
        except Exception as e:
            print(f"[ERROR] Failed to read {f.name}: {e}")

    return data


def convert_to_csv(json_folder: Path, output_csv: Path):
    """Loads JSONs → DataFrame → CSV"""
    records = load_json_files(json_folder)

    df = pd.DataFrame(records)

    # Save CSV only
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✔ CSV created: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON folder to CSV only")
    parser.add_argument("--source", "-s", required=True, help="Folder containing .json files")
    parser.add_argument("--csv", "-c", default="output.csv", help="Output CSV filename")

    args = parser.parse_args()

    convert_to_csv(
        Path(args.source),
        Path(args.csv)
    )


if __name__ == "__main__":
    main()
