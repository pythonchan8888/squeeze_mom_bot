import pandas as pd
from pathlib import Path
import pytest
import io
import tempfile
from unittest.mock import patch, MagicMock
from src.smb.data.hl_archive import fetch_day, load_ticks


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client that returns fake data."""
    mock_client = MagicMock()
    
    # Mock download_fileobj to write sample data to the provided file object
    def fake_download(bucket, key, file_obj):
        # Create sample LZ4 compressed data (we'll mock the decompression)
        file_obj.write(b"mock_lz4_data")
    
    mock_client.download_fileobj.side_effect = fake_download
    return mock_client


@pytest.fixture
def mock_lz4_decompress():
    """Create a mock for lz4.frame.decompress that returns sample JSON data."""
    sample_json = b'{"timestamp": 1672531200000, "price": 100.0, "size": 1.0, "side": "b"}\n'
    return MagicMock(return_value=sample_json)


def test_fetch_day_downloads_and_converts(mock_s3_client, mock_lz4_decompress, tmp_path):
    """Test that fetch_day downloads data, decompresses it, and saves as Parquet."""
    with patch("src.smb.data.hl_archive.get_s3_client", return_value=mock_s3_client), \
         patch("src.smb.data.hl_archive.lz4.frame.decompress", mock_lz4_decompress):
        
        # Call the function with a test date
        date = pd.Timestamp("2024-01-01")
        result_path = fetch_day("BTC", date, tmp_path)
        
        # Check that the S3 client was called with the correct parameters
        mock_s3_client.download_fileobj.assert_called_once()
        _, args, _ = mock_s3_client.download_fileobj.mock_calls[0]
        assert args[0] == "hyperliquid-archive"
        assert "market_data/2024/01/01/trades/btc.lz4" in args[1]
        
        # Check that lz4 decompression was called
        mock_lz4_decompress.assert_called_once()
        
        # Check that the result path is correct
        assert result_path.name.endswith("_trades.parquet")
        assert "btc" in str(result_path)
        assert "2024" in str(result_path)
        assert "01" in str(result_path)


def test_load_ticks_with_no_data():
    """Test that load_ticks returns an empty DataFrame when no data is available."""
    with patch("src.smb.data.hl_archive.fetch_day", side_effect=Exception("S3 error")):
        # Use a non-existent directory
        result = load_ticks("BTC", "2024-01-01", "2024-01-02", Path("nonexistent"))
        
        # Should return an empty DataFrame with the expected columns
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_columns = {"timestamp", "price", "size", "side", "symbol"}
        assert set(result.columns) == expected_columns


def test_load_ticks_merges_multiple_days():
    """Test that load_ticks correctly merges data from multiple days."""
    # Create mock data for two days
    day1_data = pd.DataFrame({
        "timestamp": [1672531200000],
        "price": [100.0],
        "size": [1.0],
        "side": ["b"],
        "symbol": ["BTC"]
    })
    
    day2_data = pd.DataFrame({
        "timestamp": [1672617600000],
        "price": [101.0],
        "size": [2.0],
        "side": ["s"],
        "symbol": ["BTC"]
    })
    
    # Mock reading parquet files to return our test data
    def mock_read_parquet(path):
        if "2024-01-01" in str(path):
            return day1_data
        elif "2024-01-02" in str(path):
            return day2_data
        else:
            raise ValueError(f"Unexpected path: {path}")
    
    # Mock fetch_day to create fake parquet files
    def mock_fetch(symbol, date, out_dir):
        path = out_dir / symbol.lower() / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{date:%Y-%m-%d}_trades.parquet"
    
    with patch("src.smb.data.hl_archive.fetch_day", side_effect=mock_fetch), \
         patch("pandas.read_parquet", side_effect=mock_read_parquet), \
         tempfile.TemporaryDirectory() as tmpdir:
        
        # Call load_ticks
        result = load_ticks("BTC", "2024-01-01", "2024-01-02", Path(tmpdir))
        
        # Should have combined both days
        assert len(result) == 2
        assert result.iloc[0]["price"] == 100.0
        assert result.iloc[1]["price"] == 101.0 