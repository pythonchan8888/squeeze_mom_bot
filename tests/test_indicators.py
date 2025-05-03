import numpy as np
import pandas as pd
from src.smb.indicators import bollinger, keltner, momentum, true_range


def test_bollinger_basic():
    """Test basic Bollinger Band calculation."""
    s = pd.Series(np.arange(30, dtype=float))
    bb = bollinger(s, length=20)

    # Check shape and column names
    assert bb.shape == (30, 3)
    assert set(bb.columns) == {"mid", "upper", "lower"}

    # Check values once window is filled
    assert np.isclose(bb["mid"].iloc[19], s.iloc[:20].mean())
    assert np.isclose(bb["upper"].iloc[19], bb["mid"].iloc[19] + 2.0 * s.iloc[:20].std())
    assert np.isclose(bb["lower"].iloc[19], bb["mid"].iloc[19] - 2.0 * s.iloc[:20].std())

    # NaN values before window is filled
    assert pd.isna(bb["mid"].iloc[18])


def test_keltner_basic():
    """Test basic Keltner Channel calculation."""
    high = pd.Series(np.arange(30, 60, dtype=float))
    low = pd.Series(np.arange(10, 40, dtype=float))
    close = pd.Series(np.arange(20, 50, dtype=float))

    kc = keltner(high, low, close, length=10, atr_mult=1.5)

    # Check shape and column names
    assert kc.shape == (30, 3)
    assert set(kc.columns) == {"mid", "upper", "lower"}

    # Check that upper is always above mid and lower is always below mid
    assert (kc["upper"].iloc[10:] > kc["mid"].iloc[10:]).all()
    assert (kc["lower"].iloc[10:] < kc["mid"].iloc[10:]).all()

    # First few values may be populated with ewm in pandas implementation
    # So we're not checking for NaNs here


def test_momentum_basic():
    """Test basic momentum calculation."""
    # Simple test with consecutive integers
    s = pd.Series([1, 2, 3, 4, 5])
    mom = momentum(s, length=1)

    # With length=1, should be current value minus previous
    # Check values, not exact equality due to floating point
    assert np.allclose(mom.iloc[1:].values, np.array([1, 1, 1, 1]))

    # First value should be NaN since there's no previous value
    assert pd.isna(mom.iloc[0])

    # Test with a different length
    mom2 = momentum(s, length=2)
    assert np.allclose(mom2.iloc[2:].values, np.array([2, 2, 2]))


def test_true_range_basic():
    """Test basic true range calculation."""
    high = pd.Series([10, 12, 15, 14, 13])
    low = pd.Series([8, 9, 10, 9, 8])
    close = pd.Series([9, 11, 14, 12, 10])

    tr = true_range(high, low, close)

    # First TR can be calculated as high-low (no previous close)
    assert tr.iloc[0] == 2.0

    # Calculate expected values for remaining items
    expected = [
        max(12 - 9, abs(12 - 9), abs(9 - 9)),  # = 3
        max(15 - 10, abs(15 - 11), abs(10 - 11)),  # = 5
        max(14 - 9, abs(14 - 14), abs(9 - 14)),  # = 5
        max(13 - 8, abs(13 - 12), abs(8 - 12)),  # = 5
    ]

    # Check values
    assert np.allclose(tr.iloc[1:].values, expected)


def test_squeeze_detection_reversed():
    """Test detecting when BB is outside KC."""
    # Create synthetic data where BB is outside KC
    # This is the opposite case of a squeeze
    close = pd.Series(np.linspace(10, 20, 50))
    # Add significant volatility to close to make BB wider
    np.random.seed(42)  # For reproducible results
    close = close + np.random.normal(0, 2.0, 50)

    # Create high and low with small range for narrow KC
    high = close + 0.2
    low = close - 0.2

    # Calculate indicators
    bb = bollinger(close, length=20, std=2.0)
    kc = keltner(high, low, close, length=20, atr_mult=1.5)

    # Wait for both indicators to be populated
    idx = 25

    # With high volatility in close but small high-low range,
    # BB should be outside KC (BB upper > KC upper or BB lower < KC lower)
    assert (
        bb["upper"].iloc[idx] > kc["upper"].iloc[idx]
        or bb["lower"].iloc[idx] < kc["lower"].iloc[idx]
    ), "Expected BB to be outside KC with high volatility in close"
