from pathlib import Path

from src.data_loader import load_dataset, profile_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "email.csv"


def test_real_dataset_can_be_loaded():
    dataframe = load_dataset(str(DATASET_PATH))

    assert not dataframe.empty
    assert "Category" in dataframe.columns
    assert "Message" in dataframe.columns


def test_real_dataset_profile_contains_expected_information():
    dataframe = load_dataset(str(DATASET_PATH))
    profile = profile_dataset(dataframe)

    assert profile["row_count"] > 0
    assert profile["column_count"] >= 2
    assert profile["duplicate_rows"] >= 0
    assert "Category" in profile["column_profiles"]
    assert "Message" in profile["column_profiles"]
