import pytest
import pandas as pd
from data_preprocessing import load_data

@pytest.fixture
def sample_csv(tmp_path):
    # Create a sample CSV file for testing
    csv_content = "Id,Entity,Year,Records,Organization type,Method\n1,Entity1,2020,100,Type1,Method1\n2,Entity2,2021,200,Type2,Method2"
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(csv_content)
    return csv_file

def test_load_data_existing_file(sample_csv):
    # Test loading existing file
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['Id', 'Entity', 'Year', 'Records', 'Organization type', 'Method']

def test_load_data_non_existing_file():
    # Test loading non-existing file
    filepath = "non_existing_file.csv"
    df = load_data(filepath)
    assert df is None
