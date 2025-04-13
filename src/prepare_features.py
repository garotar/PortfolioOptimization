import pandas as pd
import pandas_ta as ta
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
    df["amplitude"] = (df["high"] - df["low"]) / df["low"]

    if not features:
        return df, ["return", "amplitude"]

    main_features = ["return", "amplitude"]

    def add_feature(
        df: pd.DataFrame,
        feat_type: str,
        param: str
    ):
        if feat_type == "sma":
            return (
                df.groupby("ticker")["close"]
                  .transform(lambda x: ta.sma(x, length=param))
            )
        elif feat_type == "ema":
            return (
                df.groupby("ticker")["close"]
                  .transform(lambda x: ta.ema(x, length=param))
            )
        elif feat_type == "rsi":
            return (
                df.groupby("ticker")["close"]
                  .transform(lambda x: ta.rsi(x, length=param))
            )
        elif feat_type == "atr":
            return (
                df.groupby("ticker")
                  .apply(lambda x: ta.atr(x["high"], x["low"], x["close"],
                                          length=param))
                  .reset_index(level=0, drop=True)
            )
        elif feat_type == "volume_ratio":
            return (
                df["volume"] / df.groupby("ticker")["volume"]
                .transform(lambda x: x.rolling(param).mean())
            )
        elif feat_type == "amplitude_mean":
            return (
                df.groupby("ticker")["amplitude"]
                  .transform(lambda x: x.rolling(param).mean())
            )
        elif feat_type == "return_lag":
            return df.groupby("ticker")["return"].shift(param)
        else:
            raise ValueError(f"Признак '{feat_type}' не поддерживается.")

    for feat in features:
        try:
            feat_type, param_str = feat.rsplit("_", 1)
            param = int(param_str)
            df[feat] = add_feature(df, feat_type, param)
            main_features.append(feat)
        except Exception as e:
            raise ValueError(f"Ошибка генерации признака '{feat}': {e}")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df, main_features
