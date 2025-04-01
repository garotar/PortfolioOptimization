import pandas as pd
import pandas_ta as ta
from typing import Optional, List, Tuple


def prepare_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Генерирует новые признаки для датафрейма.

    Args:
        df: Исходный датафрейм с колонками ["UID", "UTC", "open", "close", "high", "low", "volume"].
        features: Список признаков для генерации в формате: "признак_значение" (например, ["sma_5", "ema_10", "rsi_14", "atr_14"]).

    Return:
        Датафрейм с новыми признаками и список добавленных признаков.
    """
    df = df.sort_values(["UID", "UTC"])

    df["return"] = (df["close"] - df["open"]) / df["open"]
    df["amplitude"] = (df["high"] - df["low"]) / df["low"]

    if not features:
        return df, ["return", "amplitude"]

    main_features = ["return", "amplitude"]

    def add_feature(df, feat_type, param):
        if feat_type == "sma":
            return df.groupby("UID")["close"].transform(lambda x: ta.sma(x, length=param))
        elif feat_type == "ema":
            return df.groupby("UID")["close"].transform(lambda x: ta.ema(x, length=param))
        elif feat_type == "rsi":
            return df.groupby("UID")["close"].transform(lambda x: ta.rsi(x, length=param))
        elif feat_type == "atr":
            return df.groupby("UID").apply(lambda x: ta.atr(x["high"], x["low"], x["close"], length=param)).reset_index(level=0, drop=True)
        elif feat_type == "volume_ratio":
            return df["volume"] / df.groupby("UID")["volume"].transform(lambda x: x.rolling(param).mean())
        elif feat_type == "amplitude_mean":
            return df.groupby("UID")["amplitude"].transform(lambda x: x.rolling(param).mean())
        elif feat_type == "return_lag":
            return df.groupby("UID")["return"].shift(param)
        else:
            raise ValueError(f"Признак '{feat_type}' не поддерживается.")

    for feat in features:
        try:
            feat_type, param = feat.rsplit("_", 1)
            param = int(param)
            df[feat] = add_feature(df, feat_type, param)
            main_features.append(feat)
        except Exception as e:
            raise ValueError(f"Ошибка генерации признака '{feat}': {e}")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return (df, main_features)