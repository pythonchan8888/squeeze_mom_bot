"""
Hyperliquid websocket consumer that writes live trade data to storage.
"""

from __future__ import annotations
import asyncio
import json
import pandas as pd
import duckdb
from pathlib import Path
import websockets
from typing import AsyncGenerator, Callable, Optional, List, Dict, Any
from loguru import logger
from .schema import Trade


class LiveWriter:
    """
    Writes live trade data to DuckDB and/or Parquet files.
    """
    
    def __init__(self, db_path: Path, parquet_dir: Optional[Path] = None):
        """
        Initialize live writer.
        
        Args:
            db_path: Path to DuckDB database file
            parquet_dir: Directory for Parquet files (optional)
        """
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.conn = self._initialize_db()
        self._running = False
        self._task = None
    
    def _initialize_db(self) -> duckdb.DuckDBPyConnection:
        """Initialize DuckDB database with the trades table."""
        conn = duckdb.connect(str(self.db_path))
        
        # Create trades table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp BIGINT,
                symbol VARCHAR,
                price DOUBLE,
                size DOUBLE,
                side VARCHAR(1),
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on timestamp and symbol
        conn.execute("""
            CREATE INDEX IF NOT EXISTS trades_ts_symbol_idx
            ON trades (timestamp, symbol)
        """)
        
        return conn
    
    async def start(self, symbols: list[str], 
                   on_data: Optional[Callable[[pd.DataFrame], None]] = None):
        """
        Start consuming websocket data and writing to storage.
        
        Args:
            symbols: List of symbols to subscribe to
            on_data: Optional callback for new data
        """
        if self._running:
            logger.warning("LiveWriter is already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._consume_websocket(symbols, on_data))
        
        return self._task
    
    async def stop(self):
        """Stop the websocket consumer."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
        self._task = None
    
    async def _consume_websocket(self, symbols: list[str], 
                               on_data: Optional[Callable[[pd.DataFrame], None]] = None):
        """Consume websocket data and write to storage."""
        symbols_upper = [s.upper() for s in symbols]
        
        while self._running:
            try:
                async for trades_df in connect_websocket(symbols_upper):
                    if trades_df.empty:
                        continue
                    
                    # Save the trades to DuckDB and/or Parquet
                    self.save_ticks(trades_df)
                    
                    # Call the callback if provided
                    if on_data:
                        on_data(trades_df)
            
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                logger.warning(f"Websocket connection closed: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in websocket consumer: {e}. Reconnecting in 10 seconds...")
                await asyncio.sleep(10)
    
    def save_ticks(self, df: pd.DataFrame):
        """
        Save trade ticks to DuckDB and optionally Parquet.
        
        Args:
            df: DataFrame with trade data
        """
        if df.empty:
            return
        
        # Ensure required columns exist
        required_cols = ["timestamp", "symbol", "price", "size", "side"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return
        
        try:
            # Insert into DuckDB
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.execute("INSERT INTO trades SELECT * FROM df")
            self.conn.execute("COMMIT")
            
            # Optionally write to Parquet
            if self.parquet_dir:
                self._write_to_parquet(df)
                
        except Exception as e:
            logger.error(f"Error saving ticks: {e}")
            self.conn.execute("ROLLBACK")
    
    def _write_to_parquet(self, df: pd.DataFrame):
        """Write trades to Parquet files organized by symbol and date."""
        # Group by symbol
        for symbol, group in df.groupby("symbol"):
            # Convert timestamp to datetime for easier path creation
            timestamps = pd.to_datetime(group["timestamp"], unit="ns")
            
            # Group by date
            for date, date_group in group.groupby(timestamps.dt.date):
                # Create path for this symbol and date
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
                symbol_dir = self.parquet_dir / symbol.lower()
                symbol_dir.mkdir(parents=True, exist_ok=True)
                
                parquet_path = symbol_dir / f"{date_str}_trades.parquet"
                
                # Append to existing file or create new one
                if parquet_path.exists():
                    try:
                        existing_df = pd.read_parquet(parquet_path)
                        combined_df = pd.concat([existing_df, date_group])
                        combined_df = combined_df.sort_values("timestamp")
                        combined_df.to_parquet(parquet_path, index=False)
                    except Exception as e:
                        logger.error(f"Error appending to Parquet file {parquet_path}: {e}")
                else:
                    try:
                        date_group.to_parquet(parquet_path, index=False)
                    except Exception as e:
                        logger.error(f"Error writing Parquet file {parquet_path}: {e}")


async def connect_websocket(symbols: list[str]) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Connect to Hyperliquid websocket and yield trade data.
    
    Args:
        symbols: List of symbols to subscribe to (uppercase)
        
    Yields:
        DataFrame with recent trades
    """
    uri = "wss://api.hyperliquid.xyz/ws"
    
    # Prepare subscription message
    subscribe_msg = {
        "method": "subscribe", 
        "subscription": {
            "type": "trades",
            "symbols": symbols
        }
    }
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to trade feed
        await websocket.send(json.dumps(subscribe_msg))
        
        # Process incoming messages
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Process based on message type
                if "data" in data and isinstance(data["data"], list) and data["data"]:
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(data["data"])
                        
                        # Normalize column names if needed
                        if "ts" in df.columns and "timestamp" not in df.columns:
                            df["timestamp"] = df["ts"]
                        
                        # Convert timestamp to nanoseconds if not already
                        if df["timestamp"].dtype != "int64":
                            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64")
                        
                        # Return the DataFrame with trades
                        yield df
                    except Exception as e:
                        logger.error(f"Error processing trade data: {e}")
                        yield pd.DataFrame()
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Websocket error: {e}")
                raise  # Re-raise to trigger reconnection
