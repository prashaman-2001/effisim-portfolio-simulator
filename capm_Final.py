import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.graph_objs import Scatter
from fredapi import Fred


import os
from fredapi import Fred

# Load API key from Streamlit Cloud secrets
FRED_API_KEY = os.environ.get("FRED_API_KEY")

if not FRED_API_KEY:
    import streamlit as st
    st.error("FRED API key not found. Please set it in Streamlit Cloud > Secrets.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

st.title("Efficient Portfolio Simulator")
st.markdown("""
## About This App

This interactive portfolio simulator is inspired by Harry Markowitz's *Modern Portfolio Theory (MPT)*, a foundational framework in finance that emphasizes diversification, risk management, and efficient asset allocation. 

The application allows users to simulate thousands of portfolios using selected stocks, analyzing their performance across key financial metrics such as expected return, standard deviation, beta, Sharpe ratio, and Treynor ratio. By visualizing the efficient frontier and identifying optimal portfolios — like those with maximum Sharpe ratio or minimum volatility — users can make informed, data-driven investment decisions.

In addition to MPT's diversification principles, the app integrates the **Capital Asset Pricing Model (CAPM)** to assess market-related risk (beta) and calculate performance metrics relative to a risk-free benchmark.

This tool is designed to bridge theory with practical application, giving investors, students, and analysts a powerful lens to explore the relationship between risk and return in portfolio construction.
""")

st.markdown("""
### Disclaimer

This tool is a mathematical abstraction. It models an idealized world where asset returns are normally distributed, risk can be neatly measured by standard deviation, and the past somehow resembles the future.

But reality is messier.

Markets are complex, non-linear, and prone to Black Swans — rare events that lie beyond the scope of probabilistic models. While this simulator can be useful for learning and exploring theoretical frameworks like Modern Portfolio Theory and CAPM, **it does not predict the future** or offer financial advice.

If you're risking real money, remember: no model survives first contact with uncertainty. Think critically. Hedge accordingly.
""")

# User Inputs
st.sidebar.header("User Inputs")
tickers = st.sidebar.text_input("Stock tickers (comma-separated, min 3, max 10):", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers.split(",")][:10]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
num_ports = st.sidebar.slider("Number of portfolios to simulate", 1000, 100000, 50000, step=1000)

def fetch_data(tickers, start, end):
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
    try:
        rf_data = fred.get_series("DTB3", observation_start=start_str, observation_end=end_str)
        st.success("✔️ Risk-free rate successfully pulled from FRED.")
        rf_data = rf_data.asfreq('MS').ffill() / 100 / 12
    except Exception as e:
        st.warning("⚠️ Using default risk-free rate (0.03)")
        st.text(f"FRED API error: {e}")
        rf_data = pd.Series(0.03 / 12, index=pd.date_range(start=start_str, end=end_str, freq="MS"))
    data = yf.download(tickers + ["^GSPC"], start=start_str, end=end_str, interval="1mo")
    st.success("📈 Stock data successfully pulled from Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        stock_data = data.xs('Close', level=0, axis=1)
    else:
        stock_data = data[['Close']]
    return stock_data.dropna(), rf_data.dropna()

# Fetch Data
full_prices, rf_rate = fetch_data(tickers, start_date, end_date)

# Validation: Check if tickers returned valid data
invalid_tickers = [t for t in tickers if t not in full_prices.columns]
if len(invalid_tickers) == len(tickers):
    st.error("❌ No valid tickers found. Please check your input.")
    st.stop()

# Separate market data and stock data
market_prices = full_prices["^GSPC"]
stock_prices = full_prices.drop(columns="^GSPC")

# Calculate returns
market_returns = np.log(market_prices / market_prices.shift(1)).dropna()
stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

# Align all series by common index
data_index = stock_returns.index.intersection(market_returns.index).intersection(rf_rate.index)
stock_returns = stock_returns.loc[data_index]
market_returns = market_returns.loc[data_index]
aligned_rf_rate = rf_rate.reindex(stock_returns.index).ffill()
market_risk_premium = market_returns - aligned_rf_rate

# Simulate Portfolios
st.info("🔄 Calculating portfolios... Please wait.")
results = np.zeros((num_ports, 6))
weights_record = []

for i in range(num_ports):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    port_return = np.sum(stock_returns.mean() * weights) * 12
    port_std = np.sqrt(np.dot(weights.T, np.dot(stock_returns.cov() * 12, weights)))

    # ✅ FIXED beta calculation
    beta_vector = []
    for t in tickers:
        aligned_stock, aligned_market = stock_returns[t].align(market_returns, join='inner')
        beta = aligned_stock.cov(aligned_market) / aligned_market.var()
        beta_vector.append(beta)

    beta = np.dot(weights, beta_vector)
    sharpe = (port_return - aligned_rf_rate.mean() * 12) / port_std
    treynor = (port_return - aligned_rf_rate.mean() * 12) / beta
    results[i] = [port_return, port_std, beta, sharpe, treynor, sharpe]
    weights_record.append(weights)

portfolio_df = pd.DataFrame(results, columns=["Return", "Std Dev", "Beta", "Sharpe Ratio", "Treynor Ratio", "Sharpe"])

# Visualizations
st.subheader("Stock Prices")
st.plotly_chart(px.line(stock_prices))

st.subheader("Monthly Returns")
st.markdown("This visualization plots the log returns of each stock over time. It highlights the month-to-month performance and volatility of each asset, useful for detecting consistent gainers or more volatile instruments.")
st.plotly_chart(px.line(stock_returns))

st.subheader("Risk-Free Rate (in %)")
st.markdown("This line chart displays the monthly risk-free rate, sourced from the FRED database. It is used in CAPM calculations to measure excess returns and assess portfolio performance relative to a riskless benchmark.")
fig_rf = px.line(aligned_rf_rate * 100, labels={"value": "Risk-Free Rate (%)", "index": "Date"},
                 title="Monthly 3-Month T-Bill Rate")
st.plotly_chart(fig_rf)

st.subheader("Risk-Free Rate Table")
st.markdown("This table shows the monthly 3-month T-Bill rates (annualized and converted to monthly format) used in CAPM formulas. These rates are essential for calculating excess returns.")
st.dataframe(aligned_rf_rate.rename("Monthly Risk-Free Rate (%)") * 100)

st.subheader("Market Risk Premium")
st.markdown("This chart shows the difference between the S&P 500's returns and the risk-free rate, representing the market's excess return over a riskless investment. It forms the basis for estimating expected returns in CAPM.")
st.plotly_chart(px.line(market_risk_premium * 100, labels={"value": "Market Risk Premium (%)"}))

# Covariance Matrix Table + Heatmap
st.subheader("Covariance Matrix")
st.markdown("This matrix and heatmap display the covariances between each pair of stock returns. Covariance measures how two assets move together, influencing total portfolio risk and diversification.")
st.write(stock_returns.cov())
fig_cov = plt.figure()
sns.heatmap(stock_returns.cov(), annot=True, fmt=".3f", cmap="Blues")
st.pyplot(fig_cov)

# Correlation Matrix Table + Heatmap
st.subheader("Correlation Matrix")
st.markdown("This matrix and heatmap show the correlations between each pair of stock returns. A correlation closer to 1 means they move together, while closer to -1 indicates inverse movement — key for diversification.")
st.write(stock_returns.corr())
fig_corr = plt.figure()
sns.heatmap(stock_returns.corr(), annot=True, fmt=".3f", cmap="coolwarm")
st.pyplot(fig_corr)

# Efficient Frontier with Highlights
st.subheader("Simulated Portfolios")
st.markdown("Each point in this scatterplot represents a randomly generated portfolio. The x-axis shows portfolio risk (standard deviation), and the y-axis shows expected return. The color scale represents the Sharpe Ratio. Highlighted points include the Max Sharpe and Min Volatility portfolios.")
fig = px.scatter(portfolio_df, x="Std Dev", y="Return", color="Sharpe Ratio",
                 title="Efficient Frontier",
                 labels={"Std Dev": "Risk (Std Dev)", "Return": "Annualized Return"})
fig.add_scatter(x=[portfolio_df["Std Dev"].iloc[portfolio_df['Sharpe Ratio'].idxmax()]],
                y=[portfolio_df["Return"].iloc[portfolio_df['Sharpe Ratio'].idxmax()]],
                mode="markers", marker=dict(size=12, symbol="star", color="red"), name="Max Sharpe")
fig.add_scatter(x=[portfolio_df["Std Dev"].iloc[portfolio_df['Std Dev'].idxmin()]],
                y=[portfolio_df["Return"].iloc[portfolio_df['Std Dev'].idxmin()]],
                mode="markers", marker=dict(size=12, symbol="diamond", color="blue"), name="Min Volatility")
st.plotly_chart(fig)

# Optimal Portfolios
max_sharpe = portfolio_df.iloc[portfolio_df['Sharpe Ratio'].idxmax()]
min_volatility = portfolio_df.iloc[portfolio_df['Std Dev'].idxmin()]
weights_max_sharpe = weights_record[portfolio_df['Sharpe Ratio'].idxmax()]
weights_min_volatility = weights_record[portfolio_df['Std Dev'].idxmin()]

st.subheader("Optimal Portfolios")
st.markdown("These tables display the metrics (Return, Std Dev, Beta, Sharpe Ratio, Treynor Ratio) for the two optimal portfolios:\n- **Max Sharpe Portfolio**: Best return-to-risk ratio.\n- **Min Volatility Portfolio**: Lowest overall risk.")
st.write("Max Sharpe Portfolio:", max_sharpe)
st.write("Min Volatility Portfolio:", min_volatility)

st.subheader("Weights (Max Sharpe Portfolio)")
st.markdown("These tables show the asset allocation (as percentages) for the Max Sharpe and Min Volatility portfolios. It explains how much weight each stock carries in the optimized portfolios.")
st.write(pd.Series(weights_max_sharpe * 100, index=tickers).round(2).astype(str) + " %")

st.subheader("Weights (Min Volatility Portfolio)")
st.write(pd.Series(weights_min_volatility * 100, index=tickers).round(2).astype(str) + " %")

# Pie Chart
st.subheader("Max Sharpe Portfolio Allocation")
fig_pie = px.pie(names=tickers, values=weights_max_sharpe, title="Max Sharpe Portfolio Allocation")
st.plotly_chart(fig_pie)

# Metrics Comparison Bar Chart
metrics_df = pd.DataFrame({
    "Metric": ["Return", "Std Dev", "Beta", "Sharpe Ratio", "Treynor Ratio"],
    "Max Sharpe": max_sharpe[["Return", "Std Dev", "Beta", "Sharpe Ratio", "Treynor Ratio"]].values,
    "Min Volatility": min_volatility[["Return", "Std Dev", "Beta", "Sharpe Ratio", "Treynor Ratio"]].values
})
fig_bar = px.bar(metrics_df.melt(id_vars="Metric", var_name="Portfolio", value_name="Value"),
                 x="Metric", y="Value", color="Portfolio", barmode="group",
                 title="Optimal Portfolio Metrics Comparison")
st.plotly_chart(fig_bar)

# Individual Stock Betas
beta_per_stock = {t: stock_returns[t].cov(market_returns) / market_returns.var() for t in tickers}
st.subheader("Individual Stock Betas")
st.markdown("This bar chart shows the beta of each stock relative to the market. Beta measures sensitivity to market movements — values above 1 indicate higher volatility than the market, while below 1 suggests lower volatility.")
st.bar_chart(pd.Series(beta_per_stock))
