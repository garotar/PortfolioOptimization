import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

from src.baseline import BaselineBacktest


@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.ffill().bfill()
    return df


@st.cache_data
def get_baseline_performance(data, backtest_date, init_cash):
    baseline_backtester = BaselineBacktest(df=data, close_price="close")
    _, total_pf_value, _ = baseline_backtester.buy_and_hold_performance(
        backtest_date=backtest_date,
        test_split_ratio=None,
        init_cash=init_cash
    )
    return total_pf_value


def baseline_performance(data):
    st.header("Перфоманс равновесного портфеля ")

    st.sidebar.header("Параметры портфеля")
    init_cash = st.sidebar.number_input("Начальный капитал в рублях", value=100_000, min_value=10_000)
    backtest_date = st.sidebar.date_input(
        "Дата начала бэктеста",
        value=datetime(2024, 10, 11),
        min_value=data["date"].min(),
        max_value=data["date"].max()
    )

    if st.sidebar.button("Пересчитать портфель"):
        with st.spinner("Расчет перфоманса..."):
            total_pf_value = get_baseline_performance(data, backtest_date.strftime("%Y-%m-%d"), init_cash)

        st.success("Расчет завершен")

        df_plot = total_pf_value.reset_index()
        df_plot.columns = ["date", "portfolio_value"]

        fig = px.line(
            df_plot,
            x="date",
            y="portfolio_value",
            labels={"date": "Дата", "portfolio_value": "Стоимость портфеля"}
        )

        fig.data[0].name = "Стратегия Купи и держи"
        fig.update_layout(hovermode="x unified")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Последние значения портфеля")
        df_to_display = total_pf_value.tail(10).reset_index()
        df_to_display.columns = ["Дата", "Стоимость портфеля"]
        st.dataframe(df_to_display)

    else:
        st.info("Настрой параметры портфеля в сайдбаре и нажми 'Пересчитать портфель'.")


def main():
    st.title("Прогнозирование доходностей и оптимизация портфеля")

    data_path = "./data/moex_data_may.csv"
    data = load_data(data_path)

    selected_tab = st.sidebar.radio("Выбери вкладку:", ["Графики по тикерам", "Стратегия 'Купи и держи'", "Стратегия на сигналах"])

    if selected_tab == "Графики по тикерам":
        st.sidebar.header("Настройки графика")

        selected_tickers = st.sidebar.multiselect(
            "Выбери тикеры:",
            options=list(data["ticker"].unique()),
            default=list(data["ticker"].unique())
        )

        param = st.sidebar.selectbox(
            "Выбери параметр:",
            ["open", "low", "high", "close", "volume"]
        )

        filtered_data = data[data["ticker"].isin(selected_tickers)]

        st.header("Интерактивный график")
        fig = px.line(
            filtered_data,
            x="date",
            y=param,
            color="ticker",
            labels={"date": "Дата", param: param.capitalize(), "ticker": "Тикер"},
            title=f"{param.capitalize()} за весь период по выбранным тикерам"
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_tab == "Стратегия 'Купи и держи'":
        baseline_performance(data)


if __name__ == "__main__":
    main()
