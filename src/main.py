import argparse
import json
from data_loader import load_dataset, profile_dataset

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Load and profile a real CSV or Excel dataset."
    )

    parser.add_argument(
        "--file",
        required = True,
        help = "Path to a CSV or Excel dataset",
    )

    parser.add_argument(
        "--sheet",
        default = 0,
        help = "Excel aheet name or sheet index. Ignore for CSV files."
    )

    args = parser.parse_args()

    dataframe = load_dataset(
        file_path=args.file,
        sheet_name=args.sheet,
    )

    profile = profile_dataset(dataframe)

    print("\nDataset loaded successfully.\n")
    print(f"Rows: {profile['row_count']}")
    print(f"Columns: {profile['column_count']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")

    print("\nColumn profile:")
    print(
        json.dumps(
            profile["column_profiles"],
            indent=2,
        )
    )

    print("\nFirst five rows:")
    print(dataframe.head().to_string())


if __name__ == "__main__":
    main()