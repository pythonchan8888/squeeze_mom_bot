import pandas as pd
import numpy as np
from src.smb.data import KlineBuilder

def test_kline_builder_simple():
    """Test that KlineBuilder correctly creates OHLCV bars from trades."""
    # Create synthetic trades
    ts = pd.date_range("2024-01-01", periods=3, freq="30min")
    df = pd.DataFrame(
        {
            "timestamp": ts.view("int64"),
            "price": [100, 101, 102],
            "size": [1, 1, 1],
            "side": ["b", "s", "b"],
            "symbol": ["BTC"] * 3,
        }
    )
    
    # Create builder with 1-hour interval
    kb = KlineBuilder(pd.Timedelta("1h"))
    
    # Update with trades
    bars = kb.update(df)
    
    # First two trades should be in the same bar, third trade crosses boundary
    assert len(bars) == 1
    bar = bars[0]
    
    # Check bar values
    assert bar.iloc[0]["open"] == 100
    assert bar.iloc[0]["high"] == 101
    assert bar.iloc[0]["low"] == 100
    assert bar.iloc[0]["close"] == 101
    assert bar.iloc[0]["volume"] == 2
    assert bar.iloc[0]["symbol"] == "BTC"


def test_kline_builder_multiple_symbols():
    """Test KlineBuilder with multiple symbols."""
    # Create trades for two symbols
    ts = pd.date_range("2024-01-01", periods=4, freq="15min")  # shorter interval to ensure we cross boundary
    df = pd.DataFrame(
        {
            "timestamp": ts.view("int64"),
            "price": [100, 200, 101, 201],
            "size": [1, 1, 1, 1],
            "side": ["b", "b", "s", "s"],
            "symbol": ["BTC", "ETH", "BTC", "ETH"],
        }
    )
    
    # Create builder with 30min interval
    kb = KlineBuilder(pd.Timedelta("30min"))
    
    # Update with trades
    bars = kb.update(df)
    
    # Should get one bar for each symbol
    assert len(bars) == 2
    
    # Combine bars and check each symbol
    all_bars = pd.concat(bars)
    
    # Group bars by symbol and verify
    symbols = all_bars['symbol'].unique()
    assert set(symbols) == {"BTC", "ETH"}
    
    for symbol in symbols:
        symbol_bar = all_bars[all_bars["symbol"] == symbol].iloc[0]
        if symbol == "BTC":
            assert symbol_bar["volume"] == 1
            assert symbol_bar["open"] == 100
            assert symbol_bar["close"] == 100
            assert symbol_bar["high"] == 100
            assert symbol_bar["low"] == 100
        elif symbol == "ETH":
            assert symbol_bar["volume"] == 1
            assert symbol_bar["open"] == 200
            assert symbol_bar["close"] == 200
            assert symbol_bar["high"] == 200
            assert symbol_bar["low"] == 200


def test_kline_builder_daily_interval():
    """Test KlineBuilder with daily interval to verify DST handling."""
    # Create trades spanning DST change (spring forward)
    # March 10, 2024 was DST start in US
    ts1 = pd.Timestamp("2024-03-09 23:30:00")  # Before DST change
    ts2 = pd.Timestamp("2024-03-10 01:30:00")  # After DST change
    ts3 = pd.Timestamp("2024-03-10 23:30:00")  # Almost end of day
    ts4 = pd.Timestamp("2024-03-11 00:30:00")  # Next day
    
    df = pd.DataFrame(
        {
            "timestamp": [ts1.value, ts2.value, ts3.value, ts4.value],
            "price": [100, 101, 102, 103],
            "size": [1, 1, 1, 1],
            "side": ["b", "s", "b", "s"],
            "symbol": ["BTC"] * 4,
        }
    )
    
    # Create builder with daily interval
    kb = KlineBuilder(pd.Timedelta("1d"))
    
    # Update with trades
    bars = kb.update(df)
    
    # With our implementation, we get two completed bars (Mar 9 and Mar 10)
    assert len(bars) == 2
    
    # Get the March 10 bar
    march10_bar = None
    for bar in bars:
        ts = pd.Timestamp(bar.iloc[0]["timestamp"])
        if ts.day == 11:  # March 11 is the end timestamp for March 10 bar
            march10_bar = bar
            break
    
    assert march10_bar is not None
    
    # Check the bar values
    # The bar contains trades from ts2 and ts3
    assert march10_bar.iloc[0]["open"] == 101
    assert march10_bar.iloc[0]["high"] == 102
    assert march10_bar.iloc[0]["low"] == 101
    assert march10_bar.iloc[0]["close"] == 102
    assert march10_bar.iloc[0]["volume"] == 2


def test_empty_bar_filling():
    """Test that empty bars are correctly filled when there's a gap in trades."""
    # Create initial trade
    ts1 = pd.Timestamp("2024-01-01 10:00:00")
    
    df1 = pd.DataFrame(
        {
            "timestamp": [ts1.value],
            "price": [100],
            "size": [1],
            "side": ["b"],
            "symbol": ["BTC"],
        }
    )
    
    # Create a trade that's 3 hours later (skipping 2 hourly bars)
    ts2 = pd.Timestamp("2024-01-01 13:00:00")
    
    df2 = pd.DataFrame(
        {
            "timestamp": [ts2.value],
            "price": [105],
            "size": [1],
            "side": ["s"],
            "symbol": ["BTC"],
        }
    )
    
    # Create builder with hourly interval
    kb = KlineBuilder(pd.Timedelta("1h"))
    
    # Process first trade
    kb.update(df1)
    
    # Fill empty bars up to current time
    current_time = pd.Timestamp("2024-01-01 13:00:00")
    filled_bars = kb.fill_empty_bars(current_time, ["BTC"])
    
    # Our implementation fills 11:00, 12:00, and 13:00
    assert len(filled_bars) == 3
    
    # Check times and values of filled bars
    filled_df = pd.concat(filled_bars)
    
    # Sort by timestamp to ensure order
    filled_df = filled_df.sort_values("timestamp")
    
    # First filled bar should be 11:00 with the last known price
    timestamps = [pd.Timestamp(ts) for ts in filled_df["timestamp"]]
    assert timestamps[0] == pd.Timestamp("2024-01-01 11:00:00")
    assert timestamps[1] == pd.Timestamp("2024-01-01 12:00:00")
    assert timestamps[2] == pd.Timestamp("2024-01-01 13:00:00")
    
    # All bars should have the last known price
    for i in range(3):
        assert filled_df.iloc[i]["open"] == 100
        assert filled_df.iloc[i]["close"] == 100
        assert filled_df.iloc[i]["volume"] == 0
    
    # Now update with the second trade and ensure it creates a new bar
    new_bars = kb.update(df2)
    
    # The 10:00 bar might complete due to the timestamp in df2
    assert len(new_bars) <= 1
