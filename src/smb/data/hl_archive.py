"""
Downloader/parquet converter for Hyperliquid S3 archive.
"""

from __future__ import annotations
import boto3
import lz4.frame
import pandas as pd
import json
import io
import tempfile
from pathlib import Path
from botocore.client import Config
from botocore import UNSIGNED
from loguru import logger


# Initialize S3 client with anonymous access
# (Hyperliquid archive is public read)
def get_s3_client():
    """Get an anonymous S3 client for accessing public buckets."""
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))


def fetch_day(symbol: str, date: pd.Timestamp, out_dir: Path) -> Path:
    """
    Download one day of Hyperliquid trade data and save as Parquet.
    
    Args:
        symbol: Trading pair symbol (e.g. "BTC")
        date: Date to download
        out_dir: Output directory
        
    Returns:
        Path to saved Parquet file
    """
    symbol = symbol.lower()  # Hyperliquid files use lowercase
    
    # Format the S3 key
    key = f"market_data/{date:%Y/%m/%d}/trades/{symbol}.lz4"
    bucket = "hyperliquid-archive"
    
    # Create output directory structure
    local_dir = out_dir / symbol / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
    local_dir.mkdir(parents=True, exist_ok=True)
    out_file = local_dir / f"{date:%Y-%m-%d}_trades.parquet"
    
    # Skip if file already exists
    if out_file.exists():
        logger.info(f"File already exists: {out_file}")
        return out_file
    
    # Get S3 client
    s3 = get_s3_client()
    
    try:
        # Download LZ4 file to memory buffer
        logger.info(f"Downloading s3://{bucket}/{key}")
        tmp = io.BytesIO()
        s3.download_fileobj(bucket, key, tmp)
        tmp.seek(0)
        
        # Decompress LZ4
        logger.info(f"Decompressing LZ4 data")
        decompressed = lz4.frame.decompress(tmp.read())
        
        # Parse JSON
        logger.info(f"Parsing JSON data")
        df = pd.read_json(io.BytesIO(decompressed), lines=True)
        
        # Format columns to match schema
        # Map Hyperliquid-specific fields if needed
        if 'timestamp' not in df.columns and 'ts' in df.columns:
            df['timestamp'] = df['ts']
        
        # Convert timestamp to nanoseconds if it's not already
        if df['timestamp'].dtype != 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64')
        
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol.upper()
        
        # Save as Parquet
        logger.info(f"Saving to {out_file}")
        df.to_parquet(out_file, index=False)
        
        return out_file
        
    except Exception as e:
        logger.error(f"Error downloading {key}: {e}")
        raise


def load_ticks(symbol: str, start_date: str, end_date: str, data_dir: Path = None) -> pd.DataFrame:
    """
    Load trade ticks for a symbol between start and end dates.
    
    Args:
        symbol: Trading pair symbol
        start_date: Start date (inclusive) in "YYYY-MM-DD" format
        end_date: End date (inclusive) in "YYYY-MM-DD" format
        data_dir: Base directory for data files (default: "data/raw")
        
    Returns:
        DataFrame with trade data
    """
    # Set default data directory if not provided
    if data_dir is None:
        data_dir = Path("data/raw")
    
    # Parse dates
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Generate a range of dates
    dates = pd.date_range(start, end, freq='D')
    
    # Create a list to hold DataFrames
    dfs = []
    
    # Download/load each day's data
    for date in dates:
        symbol_dir = data_dir / symbol.lower() / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}"
        parquet_file = symbol_dir / f"{date:%Y-%m-%d}_trades.parquet"
        
        # Download if file doesn't exist
        if not parquet_file.exists():
            try:
                parquet_file = fetch_day(symbol, date, data_dir)
            except Exception as e:
                logger.warning(f"Could not download {date:%Y-%m-%d} for {symbol}: {e}")
                continue
        
        # Load the Parquet file
        try:
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {parquet_file}: {e}")
    
    # Combine all DataFrames
    if not dfs:
        logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
        return pd.DataFrame(columns=["timestamp", "price", "size", "side", "symbol"])
    
    return pd.concat(dfs, ignore_index=True)
