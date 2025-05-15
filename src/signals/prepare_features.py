import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional, List, Tuple


def prepare_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Генерирует новые признаки для датафрейма.

    Args:
        df: Исходный датафрейм с колонками
            ["ticker", "date", "open", "close", "high", "low", "volume"].
        features: Список признаков для генерации в формате:
            "признак_значение" (например, ["sma_5", "ema_10", "rsi_14"]).

    Returns:
        Датафрейм с новыми признаками и список добавленных признаков.
    """
    df = df.sort_values(["ticker", "date"])

    df["return"] = df.groupby("ticker")["close"].pct_change().fillna(0)
    df["amplitude"] = np.where(
        df["low"] != 0, (df["high"] - df["low"]) / df["low"], df["high"]
    )

    if not features:
        main_features = ["return", "amplitude"]
        return df, main_features

    main_features = ["return", "amplitude", "anomaly_score", "anomaly"]

    def add_feature(df, feat_type, param):
        if feat_type == "sma":
            return df.groupby("ticker")["close"].transform(
                lambda x: ta.sma(x, length=param)
            )

        elif feat_type == "ema":
            return df.groupby("ticker")["close"].transform(
                lambda x: ta.ema(x, length=param)
            )

        elif feat_type == "rsi":
            return df.groupby("ticker")["close"].transform(
                lambda x: ta.rsi(x, length=param)
            )

        elif feat_type == "atr":
            return df.groupby("ticker").apply(
                lambda x: ta.atr(x["high"], x["low"], x["close"], length=param)
            ).reset_index(level=0, drop=True)

        elif feat_type == "volume_ratio":
            volume_ma = df.groupby("ticker")["volume"].transform(
                lambda x: x.rolling(param).mean()
            )
            return df["volume"] / volume_ma

        elif feat_type == "amplitude_mean":
            return df.groupby("ticker")["amplitude"].transform(
                lambda x: x.rolling(param).mean()
            )

        elif feat_type == "return_lag":
            return df.groupby("ticker")["return"].shift(param)

        elif feat_type == "std":
            return df.groupby("ticker")["return"].transform(
                lambda x: x.rolling(param).std()
            )

        elif feat_type == "momentum":
            return df.groupby("ticker")["close"].transform(
                lambda x: x.pct_change(param)
            )

        elif feat_type == "bollinger_bands":
            ma = df.groupby("ticker")["close"].transform(
                lambda x: x.rolling(param).mean()
            )
            std = df.groupby("ticker")["close"].transform(
                lambda x: x.rolling(param).std()
            )
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            return (df["close"] - lower_band) / (upper_band - lower_band)

        elif feat_type == "bollinger_width":
            ma = df.groupby("ticker")["close"].transform(
                lambda x: x.rolling(param).mean()
            )
            std = df.groupby("ticker")["close"].transform(
                lambda x: x.rolling(param).std()
            )
            return 4 * std / ma

        else:
            raise ValueError(f"Признак '{feat_type}' не поддерживается.")

    for feat in features:
        if feat in ["return", "amplitude", "anomaly_score", "anomaly"]:
            continue
        try:
            feat_type, param_str = feat.rsplit("_", 1)
            param = int(param_str)
            df[feat] = add_feature(df, feat_type, param)
            main_features.append(feat)
        except Exception as e:
            raise ValueError(f"Ошибка генерации признака '{feat}': {e}")

    df.update(df.groupby("ticker").ffill())
    df.update(df.groupby("ticker").bfill())

    return (df, main_features)
