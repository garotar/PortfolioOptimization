import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from src.portfolio.black_litterman_opt import BlackLittermanOptimizer
from src.streamlit.streamlit_utils import (
    load_data, load_idxs, load_signals, load_imoex,
    get_baseline_performance, get_index_portfolios,
    prepare_equity_with_indexes, plot_equity_comparison,
    sidebar_portfolio_params, prepare_metrics_comparison,
    sidebar_signal_strategy_params, get_portfolio_metrics,
    get_signals_portfolio, get_ema_portfolio,
    get_model_predictions, plot_dnn_predictions,
    run_bl_backtest, plot_historical_and_test_weights
)


def main():
    st.set_page_config(layout="wide", page_title="Оптимизация портфеля")
    st.title("Прогнозирование доходностей и оптимизация портфеля")

    data = load_data()

    selected_tab = st.sidebar.radio(
        "Выбери вкладку:",
        ["Графики тикеров", "Бенчмарки", "Сигнальная стратегия", "Оптимизация весов портфеля"]
    )

    if selected_tab == "Графики тикеров":
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

        st.header("График OLHCV")
        fig = px.line(
            data_frame=filtered_data,
            x="date",
            y=param,
            color="ticker",
            labels={"date": "Дата", param: param.capitalize(), "ticker": "Тикер"},
            title=f"{param.capitalize()} за весь период по выбранным тикерам"
        )
        st.plotly_chart(fig, use_container_width=True)

    if selected_tab == "Бенчмарки":
        st.header("Перфоманс равновесного портфеля без ребалансировки")
        init_cash, fees, backtest_date = sidebar_portfolio_params(data)

        if st.sidebar.button("Пересчитать портфель"):
            with st.spinner("Производится расчет..."):
                portfolio, backtester = get_baseline_performance(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )

                mcftr_df, rgbitr_df = load_idxs()
                mcftr_pf, rgbitr_pf = get_index_portfolios(
                    mcftr_df=mcftr_df,
                    rgbitr_df=rgbitr_df,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )

            st.success("Расчет завершен")

            tab1, tab2, tab3 = st.tabs(
                ["Капитал и просадка", "Веса портфеля", "Сравнение с индексами"]
            )

            with tab1:
                eq_fig, dd_fig = backtester.plot_equity_and_drawdown(
                    portfolio=portfolio,
                    mode="plotly"
                )
                st.plotly_chart(eq_fig, use_container_width=True)
                st.plotly_chart(dd_fig, use_container_width=True)

            with tab2:
                allocation_fig = backtester.plot_weights(
                    portfolio=portfolio,
                    mode="plotly"
                )
                st.plotly_chart(allocation_fig, use_container_width=True)

                full_weights_fig = plot_historical_and_test_weights(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d"),
                    test_portfolio=portfolio
                )
                st.plotly_chart(full_weights_fig, use_container_width=True)

            with tab3:
                combined_df = prepare_equity_with_indexes(
                    portfolio=portfolio,
                    mcftr_portfolio=mcftr_pf,
                    rgbitr_portfolio=rgbitr_pf
                )
                comparison_fig = plot_equity_comparison(combined_df)
                st.plotly_chart(comparison_fig, use_container_width=True)

                metrics_df = prepare_metrics_comparison(
                    portfolio=portfolio,
                    mcftr_portfolio=mcftr_pf,
                    rgbitr_portfolio=rgbitr_pf
                )
                st.subheader("Сравнение метрик портфеля и индексов")
                st.table(metrics_df)

        else:
            st.info("Настрой параметры и нажми 'Пересчитать портфель'.")

    if selected_tab == "Сигнальная стратегия":
        st.header("Классификация бычьих, медвежьих и нейтральных сигналов")
        init_cash, fees, backtest_date, size, size_type, short_window, long_window = sidebar_signal_strategy_params(data=data)

        if st.sidebar.button("Пересчитать портфель"):
            with st.spinner("Производится расчет..."):
                signals_df = load_signals()
                mcftr_df, rgbitr_df = load_idxs()

                signals_portfolio, signals_backtester = get_signals_portfolio(
                    data=data,
                    signals_df=signals_df,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d"),
                    size=size,
                    size_type=size_type
                )

                ema_portfolio, _ = get_ema_portfolio(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d"),
                    short_window=short_window,
                    long_window=long_window,
                    size=size,
                    size_type=size_type
                )

                baseline_portfolio, _ = get_baseline_performance(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )

                mcftr_pf, rgbitr_pf = get_index_portfolios(
                    mcftr_df=mcftr_df,
                    rgbitr_df=rgbitr_df,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )

            st.success("Расчет завершен")

            tab1, tab2, tab3 = st.tabs(["Капитал и просадка", "Веса портфеля", "Сравнение с бенчмарками"])

            with tab1:
                eq_fig, dd_fig = signals_backtester.plot_equity_and_drawdown(
                    portfolio=signals_portfolio,
                    mode="plotly"
                )
                st.plotly_chart(eq_fig, use_container_width=True)
                st.plotly_chart(dd_fig, use_container_width=True)

            with tab2:
                allocation_fig = signals_backtester.plot_weights(
                    portfolio=signals_portfolio,
                    mode="plotly"
                )
                st.plotly_chart(allocation_fig, use_container_width=True)

            with tab3:
                signals_eq = signals_portfolio.value().rename("Портфель")
                ema_eq = ema_portfolio.value().rename("EMA-crossover")
                baseline_eq = baseline_portfolio.value().rename("Купи и держи")
                mcftr_eq = mcftr_pf.value().rename("Индекс полной доходности MCFTR")
                rgbitr_eq = rgbitr_pf.value().rename("Индекс гос. облигаций RGBITR")

                comparison_df = pd.concat(
                    [signals_eq, ema_eq, baseline_eq, mcftr_eq, rgbitr_eq],
                    axis=1,
                    join="inner"
                )

                comparison_df = comparison_df / comparison_df.iloc[0] * 100
                comparison_df.reset_index(inplace=True)

                fig_comparison = px.line(
                    comparison_df,
                    x="date",
                    y=["Портфель", "EMA-crossover", "Купи и держи", "Индекс полной доходности MCFTR", "Индекс гос. облигаций RGBITR"],
                    labels={"value": "Капитал", "date": "Дата", "variable": "Стратегия"},
                    title="Сравнение доходности стратегий"
                )
                fig_comparison.update_layout(template="plotly_white")
                st.plotly_chart(fig_comparison, use_container_width=True)

                comparison_metrics_df = pd.DataFrame({
                    "Метрика": get_portfolio_metrics(signals_portfolio).keys(),
                    "Портфель": get_portfolio_metrics(signals_portfolio).values(),
                    "EMA-crossover": get_portfolio_metrics(ema_portfolio).values(),
                    "Купи и держи": get_portfolio_metrics(baseline_portfolio).values(),
                    "Индекс полной доходности MCFTR": get_portfolio_metrics(mcftr_pf).values(),
                    "Индекс гос. облигаций RGBITR": get_portfolio_metrics(rgbitr_pf).values()
                })

                comparison_metrics_df.set_index("Метрика", inplace=True)
                comparison_metrics_df = comparison_metrics_df.round(4)

                st.subheader("Сравнение финансовых метрик")
                st.table(comparison_metrics_df)

        else:
            st.info("Настрой параметры и нажми 'Пересчитать портфель'.")

    if selected_tab == "Оптимизация весов портфеля":
        st.header("Оптимизация весов портфеля через модель Блэка-Литтермана")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        init_cash, fees, backtest_date = sidebar_portfolio_params(data=data)

        model_choice = st.sidebar.selectbox(
            "Выбери DL модель для прогнозирования:",
            ["LSTM", "GRU", "LSTM + Attention", "GRU + Attention"],
            help="Модель должна быть предобучена"
        )

        rebalance_freq = st.sidebar.selectbox(
                "Частота ребалансировки портфеля:",
                ["Q", "M", "D"],
                index=0,
                format_func=lambda x: {"Q": "Квартальная", "M": "Месячная", "D": "Ежедневная"}[x],
                help="Не рекомендуется использовать ежедневную ребалансировку"
            )

        if model_choice == "LSTM":
            model_params = {
                "lstm_hidden": 128,
                "lstm_layers": 2,
                "dropout": 0.3,
                "cnn_embed_dim": 32
            }
        elif model_choice == "GRU":
            model_params = {
                "gru_hidden": 128,
                "gru_layers": 2,
                "dropout": 0.3,
                "cnn_embed_dim": 32
            }
        elif model_choice == "LSTM + Attention":
            model_params = {
                "lstm_hidden": 256,
                "cnn_embed_dim": 32,
                "attention_heads": 16,
                "dropout": 0.5,
                "lstm_layers": 3
            }
        elif model_choice == "GRU + Attention":
            model_params = {
                "gru_hidden": 256,
                "cnn_embed_dim": 32,
                "attention_heads": 16,
                "dropout": 0.5,
                "gru_layers": 3
            }

        tickers = sorted(data["ticker"].unique())

        st.sidebar.header("Ручной ввод уверенностей для каждого тикера:")

        default_confidences = {
            "CHMF": 0.1,
            "GAZP": 0.5,
            "LKOH": 0.5,
            "MGNT": 0.3,
            "NVTK": 0.3,
            "ROSN": 0.1,
            "SBER": 0.1,
            "YDEX": 0.7
        }
        user_confidences = {}
        for ticker in sorted(data["ticker"].unique()):
            confidence = st.sidebar.slider(
                f"Уверенность для {ticker}",
                min_value=0.0,
                max_value=1.0,
                value=default_confidences.get(ticker, 0.5),
                step=0.01
            )
            user_confidences[ticker] = confidence

        if st.sidebar.button("Запустить оптимизацию весов портфеля"):
            with st.spinner(f"Получение прогнозов и оптимизация ({model_choice})..."):
                imoex = load_imoex()

                preds, targets, lb, ub, test_loader = get_model_predictions(
                    data=data,
                    model_name=model_choice,
                    model_params=model_params,
                    device=DEVICE
                )

                weights_optimizer = BlackLittermanOptimizer(
                    historical_prices=data,
                    market_prices=imoex,
                    test_start_date=backtest_date.strftime("%Y-%m-%d")
                )

                preds_df = weights_optimizer.prepare_predictions(
                    df=data,
                    test_start_date=backtest_date.strftime("%Y-%m-%d"),
                    predictions=preds
                )

                views = preds_df.mean(axis=0).to_dict()
                confidences_array = np.array([user_confidences[t] for t in tickers])

                optimal_weights, S, market_prior, posterior_returns = weights_optimizer.optimize_weights(
                    views=views,
                    confidences=confidences_array
                )

                bl_portfolio, bl_backtester = run_bl_backtest(
                    data=data,
                    optimal_weights=optimal_weights,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d"),
                    rebalance_freq=rebalance_freq
                )

                st.session_state.bl_results = {
                    "optimal_weights": optimal_weights,
                    "market_prior": market_prior,
                    "posterior_returns": posterior_returns,
                    "bl_portfolio": bl_portfolio,
                    "bl_backtester": bl_backtester,
                    "preds": preds,
                    "targets": targets,
                    "lb": lb,
                    "ub": ub,
                    "test_loader": test_loader
                }

            st.success("Оптимизация завершена")

        if "bl_results" in st.session_state:
            results = st.session_state.bl_results

            st.subheader("Итоговые веса портфеля")
            weights_df = pd.DataFrame({
                "Тикер": list(results["optimal_weights"].keys()),
                "Вес": [f"{w:.2%}" for w in results["optimal_weights"].values()],
                "Уверенность": [f"{user_confidences[t]:.2f}" for t in results["optimal_weights"].keys()]
            })
            st.table(weights_df)

            tab1, tab2, tab3, tab4 = st.tabs(["Капитал и просадка", "Веса портфеля", "Сравнение с бенчмарками", "Анализ предсказаний"])

            with tab1:
                equity_fig, dd_fig = results["bl_backtester"].plot_equity_and_drawdown(
                    portfolio=results["bl_portfolio"],
                    mode="plotly"
                )
                st.plotly_chart(equity_fig, use_container_width=True)
                st.plotly_chart(dd_fig, use_container_width=True)

            with tab2:
                allocation_fig = results["bl_backtester"].plot_weights(
                    portfolio=results["bl_portfolio"],
                    mode="plotly"
                )
                st.plotly_chart(allocation_fig, use_container_width=True)

                full_weights_fig = plot_historical_and_test_weights(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d"),
                    test_portfolio=results["bl_portfolio"]
                )
                st.plotly_chart(full_weights_fig, use_container_width=True)

            with tab3:
                mcftr_df, rgbitr_df = load_idxs()
                mcftr_pf, rgbitr_pf = get_index_portfolios(
                    mcftr_df=mcftr_df,
                    rgbitr_df=rgbitr_df,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )
                baseline_portfolio, _ = get_baseline_performance(
                    data=data,
                    init_cash=init_cash,
                    fees=fees,
                    backtest_date=backtest_date.strftime("%Y-%m-%d")
                )

                bl_eq = results["bl_portfolio"].value().rename("Портфель")
                baseline_eq = baseline_portfolio.value().rename("Купи и держи")
                mcftr_eq = mcftr_pf.value().rename("Индекс полной доходности MCFTR")
                rgbitr_eq = rgbitr_pf.value().rename("Индекс гос. облигаций RGBITR")

                comparison_df = pd.concat([bl_eq, baseline_eq, mcftr_eq, rgbitr_eq], axis=1).dropna()
                comparison_df = comparison_df / comparison_df.iloc[0] * 100
                comparison_df.reset_index(inplace=True)

                fig_comparison = px.line(
                    comparison_df,
                    x="date",
                    y=["Портфель", "Купи и держи", "Индекс полной доходности MCFTR", "Индекс гос. облигаций RGBITR"],
                    labels={"value": "Капитал", "date": "Дата", "variable": "Стратегия"},
                    title="Сравнение доходности стратегий"
                )
                fig_comparison.update_layout(template="plotly_white")
                st.plotly_chart(fig_comparison, use_container_width=True)

                comparison_metrics_df = pd.DataFrame({
                    "Метрика": get_portfolio_metrics(results["bl_portfolio"]).keys(),
                    "Портфель": get_portfolio_metrics(results["bl_portfolio"]).values(),
                    "Купи и держи": get_portfolio_metrics(baseline_portfolio).values(),
                    "MCFTR": get_portfolio_metrics(mcftr_pf).values(),
                    "RGBITR": get_portfolio_metrics(rgbitr_pf).values()
                }).set_index("Метрика").round(4)

                st.subheader("Сравнение финансовых метрик")
                st.table(comparison_metrics_df)

            with tab4:
                ticker_choice = st.selectbox("Выбери тикер:", tickers)
                preds_fig = plot_dnn_predictions(
                    df=data,
                    dataloader=results["test_loader"],
                    ticker=ticker_choice,
                    preds=results["preds"],
                    targets=results["targets"],
                    lower_bounds=results["lb"],
                    upper_bounds=results["ub"]
                )
                st.plotly_chart(preds_fig, use_container_width=True)

        else:
            st.info("Настрой уверенности и запусти оптимизацию.")


if __name__ == "__main__":
    main()
