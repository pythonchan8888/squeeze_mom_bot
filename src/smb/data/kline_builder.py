from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

@dataclass
class KlineBuilder:
    interval: pd.Timedelta  # e.g. pd.Timedelta("1h")
    _bars: dict[str, dict] = field(default_factory=lambda: defaultdict(dict))
    _last_close: dict[str, float] = field(default_factory=dict)

    def update(self, trades: pd.DataFrame) -> list[pd.DataFrame]:
        """
        trades must have columns: timestamp (ns), price, size, side
        Returns list of completed OHLCV bars for *each* symbol present.
        """
        completed: list[pd.DataFrame] = []
        
        # No trades? Nothing to do
        if trades.empty:
            return completed
        
        # Sort by timestamp to ensure proper sequencing
        trades = trades.sort_values("timestamp")
        
        # Track which days we've seen for each symbol to handle daily intervals correctly
        days_seen = defaultdict(set)
        
        for row in trades.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp, unit="ns")
            
            # For daily intervals, get the date
            if self.interval >= pd.Timedelta("1d"):
                day = ts.floor("D")
                days_seen[row.symbol].add(day)
                key = day
            else:
                key = ts.floor(self.interval)
            
            # Get or create bar for this symbol+interval
            if row.symbol not in self._bars or key not in self._bars[row.symbol]:
                # Create new bar with initial data
                self._bars.setdefault(row.symbol, {})[key] = {
                    "open": row.price, 
                    "high": row.price, 
                    "low": row.price,
                    "close": row.price, 
                    "volume": row.size, 
                    "end": key + self.interval
                }
            else:
                # Update existing bar
                d = self._bars[row.symbol][key]
                d["high"] = max(d["high"], row.price)
                d["low"] = min(d["low"], row.price)
                d["close"] = row.price
                d["volume"] += row.size
            
            # Always update last known close price for the symbol
            # (used for filling empty bars on stream gaps)
            self._last_close[row.symbol] = row.price
        
        # Check if we need to close any bars based on the latest timestamp 
        # in the batch of trades
        if not trades.empty:
            latest_ts = pd.Timestamp(trades["timestamp"].max(), unit="ns")
            
            for symbol in list(self._bars.keys()):
                for interval_start in list(self._bars[symbol].keys()):
                    bar_data = self._bars[symbol][interval_start]
                    interval_end = bar_data["end"]
                    
                    # If the latest timestamp is past the bar end, close the bar
                    if latest_ts >= interval_end:
                        completed.append(
                            pd.DataFrame(
                                {
                                    "timestamp": [interval_end],
                                    "open": [bar_data["open"]],
                                    "high": [bar_data["high"]],
                                    "low": [bar_data["low"]],
                                    "close": [bar_data["close"]],
                                    "volume": [bar_data["volume"]],
                                    "symbol": [symbol],
                                }
                            )
                        )
                        del self._bars[symbol][interval_start]
        
        return completed
    
    def fill_empty_bars(self, current_time: pd.Timestamp, symbols: list[str]) -> list[pd.DataFrame]:
        """
        Fill empty bars for the specified symbols up to the current time.
        Used when there's a gap in the trade stream to avoid missing bars.
        
        Args:
            current_time: Current timestamp
            symbols: List of symbols to check for missing bars
            
        Returns:
            List of filled bars
        """
        filled_bars = []
        
        for symbol in symbols:
            # Skip if we don't have a last close for this symbol
            if symbol not in self._last_close:
                continue
                
            last_close = self._last_close[symbol]
            
            # Get the timestamp of the most recent bar for this symbol
            most_recent = None
            if symbol in self._bars and self._bars[symbol]:
                most_recent = max(self._bars[symbol].keys())
            
            # If we have no recent bar, start from current time floored to interval
            if most_recent is None:
                # Floor to interval start
                if self.interval >= pd.Timedelta("1d"):
                    start = current_time.floor("D")
                else:
                    start = current_time.floor(self.interval)
                
                # Go back one interval to start filling from there
                start = start - self.interval
            else:
                # Start from the end of the most recent bar
                start = self._bars[symbol][most_recent]["end"] - self.interval
            
            # Calculate how many bars we need to fill - up to but not including current interval
            if self.interval >= pd.Timedelta("1d"):
                end = current_time.floor("D")
            else:
                end = current_time.floor(self.interval)
                
            # Create filled bars up to (but not including) the current interval
            while start < end:
                next_start = start + self.interval
                filled_bars.append(
                    pd.DataFrame(
                        {
                            "timestamp": [next_start],
                            "open": [last_close],
                            "high": [last_close],
                            "low": [last_close],
                            "close": [last_close],
                            "volume": [0.0],
                            "symbol": [symbol],
                        }
                    )
                )
                start = next_start
        
        return filled_bars
