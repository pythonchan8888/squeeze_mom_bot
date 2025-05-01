import pytest
from pathlib import Path


def test_data_directory_exists():
    """Test that the data directory exists."""
    data_dir = Path("data")
    assert data_dir.exists(), "Data directory doesn't exist"


def test_sample():
    """A passing test example."""
    assert True
