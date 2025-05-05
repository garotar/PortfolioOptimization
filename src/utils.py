import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict


def plot_close_prices(
    df: pd.DataFrame,
    figsize: tuple = (14, 6)
):
    """
    Рисует цены закрытия исходного датафрейма.

    Args:
        df: Исходный датафрейм.
        figsize: Размер графика.
    """
    plt.figure(figsize=figsize)

    for ticker, group in df.groupby("ticker"):
        plt.plot(group["date"], group["close"], label=ticker)

    plt.title("Цены закрытия для всех тикеров")
    plt.xlabel("Дата")
    plt.ylabel("Цена закрытия")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_equity(
    equity_series: Dict[str, pd.Series],
    title: str = "",
    ylabel: str = "",
    last_point: bool = True,
    figsize: tuple = (14, 6)
):
    """
    Рисует перфоманс портфеля.

    Args:
        equity_series: Словарь, где ключ - наименование портфеля/стратегии,
                                значение - динамика портфеля.
        title: Наименование графика.
        ylabel: Наименование оси Y.
        last_point: Точка с финальной стоимостью портфеля.
        figsize: Размер графика.
    """
    plt.figure(figsize=figsize)

    for label, series in equity_series.items():
        plt.plot(series.index, series.values, label=label)

        if last_point:
            x_last = series.index[-1]
            y_last = round(series.iloc[-1], 2)
            plt.scatter(x_last, y_last, color="red")
            plt.annotate(
                y_last,
                xy=(x_last, y_last),
                xytext=(5, -5),
                textcoords="offset points",
                fontsize=6,
                color="red",
                fontweight="bold"
            )

    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel(ylabel)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_losses(
    train_losses: np.array,
    val_losses: np.array,
    figsize: tuple = (14, 6)
):
    """
    Рисует train/val лоссы.

    Args:
        train_losses: Массив с train лоссами.
        val_losses: Массив с val лоссами.
        figsize: Размер графика.
    """
    plt.figure(figsize=figsize)

    plt.plot(train_losses, label="Train лосс")
    plt.plot(val_losses, label="Val лосс")

    plt.title("Сравнение лоссов")
    plt.xlabel("Эпоха", fontsize=14)
    plt.ylabel("Лосс", fontsize=14)

    plt.legend()
    plt.grid()
    plt.show()


def plot_lstm_predictions(
    df: pd.DataFrame,
    dataloader,
    ticker: str,
    preds: np.array,
    targets: np.array,
    lower_bounds: np.array,
    upper_bounds: np.array,
    figsize: tuple = (14, 6)
):
    """
    Рисует сравнение прогноза доходности с истинными значениями.

    Args:
        df: Исходный датафрейм. Нужен для получения уникальных тикеров.
        dataloader: Даталоадер.
        ticker: Тикер.
        preds: Массив с прогнозами доходности.
        targets: Массив с истинной доходностью.
        lower_bounds: Нижняя граница доверительного интервала для прогноза.
        upper_bounds: Верхняя граница доверительного интервала для прогноза.
        figsize: Размер графика.
    """
    dates = dataloader.dataset.dates[dataloader.dataset.window:]

    plt.figure(figsize=figsize)

    idx = list(df.ticker.unique()).index(ticker)

    plt.plot(dates, targets[:, idx], label="Таргет", color="blue", linewidth=2)
    plt.plot(dates, preds[:, idx], label="Предикт", color="orange", linewidth=2)

    plt.fill_between(
        dates,
        lower_bounds[:, idx],
        upper_bounds[:, idx],
        color="orange",
        alpha=0.2,
        label="Доверительный интервал"
    )

    plt.title(f"Сравнение таргета с предиктом для тикера {ticker}")
    plt.xlabel("Дата")
    plt.ylabel("Доходность")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_portfolio_weights(
    weights_df,
    figsize: tuple = (14, 6)
):
    plt.figure(figsize=figsize)
    weights_df.plot.area(ax=plt.gca(), alpha=0.7)

    plt.title("Веса портфеля")
    plt.ylabel("Вес")
    plt.xlabel("Дата")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
