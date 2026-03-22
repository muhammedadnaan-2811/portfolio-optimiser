import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --- APP INTERFACE ---
st.set_page_config(page_title="Muhammed's Portfolio Optimizer", layout="wide")
st.title("📈 AI-Driven Portfolio Optimizer")
st.markdown("Enter up to 10 stock tickers (e.g., AAPL, MSFT, TSLA) to find the mathematically optimal allocation.")

# User Input
user_tickers = st.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, GOOGL, NVDA, TSLA")
tickers = [t.strip().upper() for t in user_tickers.split(",")]

if len(tickers) > 10:
    st.error("Please limit to 10 stocks for performance.")
elif len(tickers) < 2:
    st.warning("Enter at least 2 stocks to optimize.")
else:
    # --- LOGIC (From your script) ---
    @st.cache_data # This makes the app fast by saving data
    def get_data(tickers):
        data = yf.download(tickers, start="2022-01-01")
        # Check if 'Adj Close' exists, otherwise use 'Close'
        if 'Adj Close' in data.columns.levels[0]:
            return data['Adj Close']
        elif 'Close' in data.columns.levels[0]:
            st.warning("Adjusted Close Price not found, using Close Price instead.")
            return data['Close']
        else:
            st.error("Neither Adjusted Close nor Close Price found for the given tickers.")
            return pd.DataFrame() # Return empty DataFrame to avoid further errors

    data = get_data(tickers)
    # Ensure data is not empty before proceeding
    if not data.empty:
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        def portfolio_performance(weights, mean_returns, cov_matrix):
            p_ret = np.dot(weights, mean_returns)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return p_ret, p_std

        def negative_sharpe(weights, mean_returns, cov_matrix):
            p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
            return -p_ret / p_std # Simplified Sharpe

        # Optimization
        num_assets = len(tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        res = minimize(negative_sharpe, num_assets*[1./num_assets], args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

        # --- DISPLAY ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimal Weights")
            weights_df = pd.DataFrame({'Stock': tickers, 'Weight (%)': (res.x * 100).round(2)})
            st.dataframe(weights_df, use_container_width=True)

        with col2:
            st.subheader("Performance Metrics")
            p_ret, p_std = portfolio_performance(res.x, mean_returns, cov_matrix)
            st.metric("Expected Annual Return", f"{p_ret:.2%}")
            st.metric("Annual Volatility (Risk)", f"{p_std:.2%}")
            st.metric("Sharpe Ratio", f"{-(res.fun):.2f}")

        st.bar_chart(weights_df.set_index('Stock'))
    else:
        st.warning("Could not retrieve stock data for the given tickers. Please check the ticker symbols.")