import pandas as pd
from beartype import beartype


@beartype
def get_rsi(stock_data: pd.DataFrame, span: int=14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)"""
    change_in_price = stock_data["Close"].diff()

    up_df, down_df = change_in_price.copy(), change_in_price.copy()

    up_df[up_df < 0] = 0
    down_df[down_df > 0] = 0
    down_df = down_df.abs()

    # EWMA (Exponential Weighted Moving Average)
    ewma_up = up_df.ewm(span=span).mean()
    ewma_down = down_df.ewm(span=span).mean()

    relative_strength = ewma_up / ewma_down
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    return relative_strength_index


@beartype
def get_stochastic_oscillator(stock_data: pd.DataFrame, window: int=14) -> pd.Series:
    """Calculate the Stochastic Oscillator (k_percent)"""
    low, high = stock_data['Low'].copy(), stock_data['High'].copy()

    rolling_low = low.rolling(window=window).min()
    rolling_high = high.rolling(window=window).max()

    # Calculate the Stochastic Oscillator
    k_percent = 100 * ((stock_data['Close'] - rolling_low) / (rolling_high - rolling_low))
    return k_percent


@beartype
def get_williams(stock_data: pd.DataFrame, window: int=14) -> pd.Series:
    """Calculate the Williams %R (r_percent)"""
    low, high = stock_data['Low'].copy(), stock_data['High'].copy()

    rolling_low = low.rolling(window=window).min()
    rolling_high = high.rolling(window=window).max()

    # Calculate Williams %R
    r_percent = ((rolling_high - stock_data['Close']) / (rolling_high - rolling_low)) * -100

    return r_percent


@beartype
def get_macd(stock_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate the Moving Average Convergence Divergence (MACD)"""
    ema_26 = stock_data['Close'].ewm(span=26).mean()
    ema_12 = stock_data['Close'].ewm(span=12).mean()

    macd = ema_12 - ema_26

    # EMA of MACD
    macd_ema_9 = macd.ewm(span=9).mean()

    return macd, macd_ema_9


@beartype
def get_obv(stock_data: pd.DataFrame) -> pd.Series:
    """Calculate the On-Balance Volume (OBV)"""
    volume_series = stock_data['Volume']
    change_in_price_series = stock_data['change_in_price']

    obv_values = volume_series.copy()
    obv_values[(change_in_price_series == 0) | change_in_price_series.isna()] = 0
    obv_values[change_in_price_series < 0] = -volume_series[change_in_price_series < 0]

    return obv_values.cumsum()
