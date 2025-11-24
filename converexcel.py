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
                obj["_source_file"] = f.name   # track source file
                data.append(obj)
        except Exception as e:
            print(f"[ERROR] Failed to read {f.name}: {e}")

    return data


def convert_to_excel_and_csv(json_folder: Path, output_excel: Path, output_csv: Path):
    """Loads JSONs → DataFrame → Excel + CSV"""
    records = load_json_files(json_folder)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Save Excel 
    df.to_excel(output_excel, index=False)

    # Save CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✔ Excel created: {output_excel}")
    print(f"✔ CSV created:   {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON folder to Excel + CSV")
    parser.add_argument("--source", "-s", required=True, help="Folder containing .json files")
    parser.add_argument("--excel", "-e", default="output.xlsx", help="Output Excel file name")
    parser.add_argument("--csv", "-c", default="output.csv", help="Output CSV file name")

    args = parser.parse_args()

    convert_to_excel_and_csv(
        Path(args.source),
        Path(args.excel),
        Path(args.csv)
    )


if __name__ == "__main__":
    main()
