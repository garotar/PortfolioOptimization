import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.portfolio.backtest import Backtester
from src.dnn.prepare_dataloaders import FinTSDataloaders
from src.dnn.trainer import Trainer
from src.dnn.lstm_forecast import LSTMForecaster
from src.dnn.gru_forecast import GRUForecaster
from src.dnn.attention_lstm import AttentionLSTMForecaster
from src.dnn.attention_gru import AttentionGRUForecaster


@st.cache_data
def load_data():
    df = pd.read_csv("./data/moex_may_report.csv", parse_dates=["date"])
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


@st.cache_data
def load_idxs():
    mcftr = pd.read_csv("./data/MCFTR_may_report.csv", sep=";", parse_dates=["date"])
    rgbitr = pd.read_csv("./data/RGBITR_may_report.csv", sep=";", parse_dates=["date"])
    return (mcftr, rgbitr)


@st.cache_data
def load_signals():
    signals_df = pd.read_csv("./data/signals.csv", parse_dates=["date"])
    return signals_df


@st.cache_data
def load_imoex():
    df = pd.read_csv("./data/IMOEX_may_report.csv", sep=";", parse_dates=["date"])
    return df


def get_baseline_performance(
    data,
    init_cash,
    fees,
    backtest_date
):
    baseline_backtester = Backtester(data, init_cash, fees, backtest_date)
    bh_portfolio = baseline_backtester.run_buy_and_hold()
    return (bh_portfolio, baseline_backtester)


def get_index_portfolios(
    mcftr_df,
    rgbitr_df,
    init_cash,
    fees,
    backtest_date
):
    mcftr_bt = Backtester(mcftr_df, init_cash, fees, backtest_date)
    mcftr_pf = mcftr_bt.run_buy_and_hold()
    rgbitr_bt = Backtester(rgbitr_df, init_cash, fees, backtest_date)
    rgbitr_pf = rgbitr_bt.run_buy_and_hold()
    return (mcftr_pf, rgbitr_pf)


def prepare_equity_with_indexes(
    portfolio,
    mcftr_portfolio,
    rgbitr_portfolio
):
    portfolio_equity = portfolio.value().rename("Портфель")
    mcftr_equity = mcftr_portfolio.value().rename("Индекс полной доходности MCFTR")
    rgbitr_equity = rgbitr_portfolio.value().rename("Индекс гос. облигаций RGBITR")

    combined_df = pd.concat([portfolio_equity, mcftr_equity, rgbitr_equity], axis=1)
    combined_df.reset_index(inplace=True)
    return combined_df


def plot_equity_comparison(combined_df):
    fig = px.line(
        combined_df,
        x="date",
        y=["Портфель", "Индекс полной доходности MCFTR", "Индекс гос. облигаций RGBITR"],
        labels={"value": "Капитал", "date": "Дата", "variable": "Актив"},
        title="Сравнение доходности портфеля с индексами"
    )
    fig.update_layout(template="plotly_white")
    return fig


def get_portfolio_metrics(portfolio):
    return {
        "Коэффициент Шарпа": portfolio.sharpe_ratio(),
        "Коэффициент Сортино": portfolio.sortino_ratio(),
        "Максимальная просадка, %": portfolio.drawdowns.max_drawdown() * 100,
        "Итоговая доходность, %": portfolio.total_return() * 100,
    }


def prepare_metrics_comparison(
    portfolio,
    mcftr_portfolio,
    rgbitr_portfolio
):
    portfolio_metrics = get_portfolio_metrics(portfolio)
    mcftr_metrics = get_portfolio_metrics(mcftr_portfolio)
    rgbitr_metrics = get_portfolio_metrics(rgbitr_portfolio)

    comparison_df = pd.DataFrame({
        "Метрика": portfolio_metrics.keys(),
        "Портфель": portfolio_metrics.values(),
        "MCFTR": mcftr_metrics.values(),
        "RGBITR": rgbitr_metrics.values()
    })

    comparison_df.set_index("Метрика", inplace=True)
    comparison_df = comparison_df.round(4)
    return comparison_df


def sidebar_portfolio_params(data):
    st.sidebar.header("Параметры портфеля")
    init_cash = st.sidebar.number_input(
        "Начальный капитал, руб.",
        value=100_000,
        min_value=10_000
    )
    backtest_date = st.sidebar.date_input(
        "Дата начала бэктеста",
        value=datetime(2024, 10, 17),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )
    fees = st.sidebar.number_input(
        "Комиссия брокера, %",
        value=0.01,
        min_value=0.0,
        max_value=0.5,
        step=0.01
    )
    return (init_cash, fees, backtest_date)


def sidebar_signal_strategy_params(data):
    st.sidebar.header("Параметры сигнальной стратегии")
    init_cash = st.sidebar.number_input(
        "Начальный капитал, руб.", value=100_000, min_value=10_000
    )
    backtest_date = st.sidebar.date_input(
        "Дата начала бэктеста",
        value=datetime(2024, 10, 17),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )
    fees = st.sidebar.number_input(
        "Комиссия брокера, %",
        value=0.01,
        min_value=0.0,
        max_value=0.5,
        step=0.01
    )

    size_type = st.sidebar.selectbox(
        "Тип выделения капитала на тикер:",
        options=["value", "amount", "percent"],
        help="Сумма денег на тикер (value), количество акций по тикеру (amount) или процент от капитала на тикер (percent)."
    )

    if size_type == "percent":
        size = st.sidebar.slider("% от капитала на тикер:", 1.0, 100.0, 12.5, step=0.5)
    else:
        size = st.sidebar.number_input(
            "Размер позиции:",
            value=12500,
            min_value=1
        )

    short_window = st.sidebar.number_input(
        "EMA короткое окно",
        min_value=1,
        max_value=50,
        value=10
    )
    long_window = st.sidebar.number_input(
        "EMA длинное окно",
        min_value=5,
        max_value=200,
        value=20
    )

    return (
        init_cash, fees, backtest_date,
        size, size_type,
        short_window, long_window
        )


def get_ema_portfolio(
    data,
    init_cash,
    fees,
    backtest_date,
    short_window,
    long_window,
    size,
    size_type
):
    backtester = Backtester(
        df=data,
        init_cash=init_cash,
        fees=fees,
        test_start_date=backtest_date
    )
    ema_portfolio = backtester.run_ema_crossover(
        short_window=short_window,
        long_window=long_window,
        size=size,
        size_type=size_type
    )
    return (ema_portfolio, backtester)


def get_signals_portfolio(
    data,
    signals_df,
    init_cash,
    fees,
    backtest_date,
    size,
    size_type
):
    backtester = Backtester(
        df=data,
        init_cash=init_cash,
        fees=fees,
        test_start_date=backtest_date
    )
    signals_portfolio = backtester.run_signal_backtest(
        signals_df=signals_df,
        size=size,
        size_type=size_type
    )
    return (signals_portfolio, backtester)


def load_pretrained_model(
    model_class,
    model_path,
    model_params,
    device="cuda"
):
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


MODEL_CLASSES = {
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
    "LSTM + Attention": AttentionLSTMForecaster,
    "GRU + Attention": AttentionGRUForecaster,
}

MODEL_PATHS = {
    "LSTM": "./models/lstm_best.pth",
    "GRU": "./models/gru_best.pth",
    "LSTM + Attention": "./models/att_lstm_best.pth",
    "GRU + Attention": "./models/att_gru_best.pth"
}


@st.cache_resource
def get_model_predictions(
    data,
    model_name,
    model_params,
    device="cpu"
):
    loader_creator = FinTSDataloaders(
        df=data,
        window=60,
        forecast_horizon=14,
        batch_size=256
    )
    _, _, test_loader, _, target_scaler = loader_creator.get_loaders()

    X_batch, _, y_batch = next(iter(test_loader))
    feature_dim = X_batch.shape[2]
    num_tickers = y_batch.shape[1]

    model_class = MODEL_CLASSES[model_name]
    model_path = MODEL_PATHS[model_name]

    model = load_pretrained_model(
        model_class=model_class,
        model_path=model_path,
        model_params={"feature_dim": feature_dim, "num_tickers": num_tickers, **model_params},
        device=device
    )

    trainer = Trainer(
        model=model,
        criterion=torch.nn.L1Loss(),
        optimizer=torch.optim.Adam(params=model.parameters(), lr=0.0005, weight_decay=0.001),
        device=device)

    preds, targets, lb, ub = trainer.predict(
        dataloader=test_loader,
        target_scaler=target_scaler,
        train_loader=test_loader,
        ci_coef=1
    )

    return (preds, targets, lb, ub, test_loader)


def run_bl_backtest(
    data,
    optimal_weights,
    init_cash,
    fees,
    backtest_date,
    rebalance_freq="Q"
):
    bl_backtester = Backtester(
        df=data,
        init_cash=init_cash,
        fees=fees,
        test_start_date=backtest_date
    )
    bl_portfolio = bl_backtester.run_buy_and_hold_rebalanced(
        weights=optimal_weights,
        rebalance_freq=rebalance_freq
    )
    return (bl_portfolio, bl_backtester)


def plot_dnn_predictions(
    df,
    dataloader,
    ticker,
    preds,
    targets,
    lower_bounds,
    upper_bounds
):
    dates = dataloader.dataset.dates[dataloader.dataset.window:]
    idx = list(df.ticker.unique()).index(ticker)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=targets[:, idx],
        mode="lines",
        name="Таргет",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=preds[:, idx],
        mode="lines",
        name="Предикт",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([upper_bounds[:, idx], lower_bounds[::-1, idx]]),
        fill="toself",
        fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="rgba(255,165,0,0)"),
        hoverinfo="skip",
        name="Доверительный интервал"
    ))

    fig.update_layout(
        title=f"Сравнение таргета с предиктом для тикера {ticker}",
        xaxis_title="Дата",
        yaxis_title="Доходность",
        template="plotly_white"
    )

    return fig


def plot_historical_and_test_weights(
    data,
    init_cash,
    fees,
    backtest_date,
    test_portfolio
):
    historical_end_date = pd.to_datetime(backtest_date) - pd.Timedelta(days=1)

    historical_bt = Backtester(
        df=data,
        init_cash=init_cash,
        fees=fees,
        test_start_date=data["date"].min().strftime("%Y-%m-%d")
    )
    historical_portfolio = historical_bt.run_buy_and_hold()

    historical_asset_value = historical_portfolio.asset_value(group_by=False)
    historical_total_value = historical_portfolio.value()
    historical_weights = historical_asset_value.divide(historical_total_value, axis=0)
    historical_weights.columns = historical_weights.columns.get_level_values(-1)

    test_asset_value = test_portfolio.asset_value(group_by=False)
    test_total_value = test_portfolio.value()
    test_weights = test_asset_value.divide(test_total_value, axis=0)
    test_weights.columns = test_weights.columns.get_level_values(-1)

    full_weights_df = pd.concat([
        historical_weights.loc[historical_weights.index <= historical_end_date],
        test_weights
    ])

    fig = go.Figure()

    for ticker in full_weights_df.columns:
        fig.add_trace(go.Scatter(
            x=full_weights_df.index,
            y=full_weights_df[ticker],
            mode="lines",
            stackgroup="one",
            name=ticker
        ))

    fig.add_shape(
        type="line",
        x0=backtest_date,
        x1=backtest_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )

    fig.add_annotation(
        x=backtest_date,
        y=1,
        yref="paper",
        text="Начало тестового периода",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20
    )

    fig.update_layout(
        title="Исторические и тестовые веса портфеля",
        xaxis_title="Дата",
        yaxis_title="Доля актива",
        template="plotly_white",
        legend_title="Тикеры"
    )

    return fig
