import pandas as pd
import pandas_ta as ta
from typing import Optional, List


def prepare_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Генерирует новые признаки для датафрейма.

    Args:
        df: Исходный датафрейм с колонками ["UID", "UTC", "open", "close", "high", "low", "volume"].
        features: Список признаков для генерации.

    Return:
        Датафрейм с новыми признаками и список добавленных признаков.
    """

    df = df.sort_values(["UID", "UTC"])

    df["return"] = (df["close"] - df["open"]) / df["open"]
    df["amplitude"] = (df["high"] - df["low"]) / df["low"]
    
    if not features:
        return df, ["return", "amplitude"]

    available_features = {
        "sma_3": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.sma(x, length=3)),
        "sma_5": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.sma(x, length=5)),
        "sma_7": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.sma(x, length=7)),
        "sma_10": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.sma(x, length=10)),        
        "ema_3": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.ema(x, length=3)),
        "ema_5": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.ema(x, length=5)),
        "ema_7": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.ema(x, length=7)),
        "ema_10": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.ema(x, length=10)),        
        "rsi_7": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.rsi(x, length=7)),
        "rsi_14": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.rsi(x, length=14)),
        "rsi_28": lambda df: df.groupby("UID")["close"].transform(lambda x: ta.rsi(x, length=28)),
        "atr_7": lambda df: df.groupby("UID").apply(lambda x: ta.atr(x["high"], x["low"], x["close"], length=7)).reset_index(level=0, drop=True),
        "atr_14": lambda df: df.groupby("UID").apply(lambda x: ta.atr(x["high"], x["low"], x["close"], length=14)).reset_index(level=0, drop=True),
        "atr_28": lambda df: df.groupby("UID").apply(lambda x: ta.atr(x["high"], x["low"], x["close"], length=28)).reset_index(level=0, drop=True),
        "volume_ratio_10": lambda df: df["volume"] / df.groupby("UID")["volume"].transform(lambda x: x.rolling(10).mean()),
        "volume_ratio_20": lambda df: df["volume"] / df.groupby("UID")["volume"].transform(lambda x: x.rolling(20).mean()),
        "volume_ratio_30": lambda df: df["volume"] / df.groupby("UID")["volume"].transform(lambda x: x.rolling(30).mean()),
        "amplitude_mean_10": lambda df: df.groupby("UID")["amplitude"].transform(lambda x: x.rolling(10).mean()),
        "amplitude_mean_20": lambda df: df.groupby("UID")["amplitude"].transform(lambda x: x.rolling(20).mean()),
        "amplitude_mean_30": lambda df: df.groupby("UID")["amplitude"].transform(lambda x: x.rolling(30).mean()),
        "return_lag_3": lambda df: df.groupby("UID")["return"].shift(3),
        "return_lag_5": lambda df: df.groupby("UID")["return"].shift(5),
        "return_lag_7": lambda df: df.groupby("UID")["return"].shift(7),
        "return_lag_10": lambda df: df.groupby("UID")["return"].shift(10),
    }

    main_features = ["return", "amplitude"]
    

    for feature in features:
        if feature in available_features:
            df[feature] = available_features[feature](df)
            main_features.append(feature)
        else:
            raise ValueError(f"Признак '{feature}' отсутствует среди доступных.")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df, main_features