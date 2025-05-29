# stock_price_downloader.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# # Set page title
# st.title("📥 Stock Price Data Downloader")

# # Ticker symbol input and date range selection
# st.subheader("1. Select or Add Ticker Symbols")

def stock_price_downloader():
    if "tickers" not in st.session_state:
        st.session_state.tickers = ["TSLA"]

    # Input to add more tickers
    ticker_input = st.text_input("Add a Ticker Symbol (e.g., AAPL):")
    if st.button("Add Ticker"):
        if ticker_input.strip() and ticker_input.upper() not in st.session_state.tickers:
            st.session_state.tickers.append(ticker_input.upper())

    # Multiselect dropdown for existing tickers
    ticker_selection = st.multiselect("Select Tickers to Download:", st.session_state.tickers, default=st.session_state.tickers)

    st.subheader("2. Select Date Range")
    start_date = st.date_input("Start Date", datetime(2014, 8, 1))
    end_date = st.date_input("End Date", datetime(2024, 11, 9))

    # Proceed button for downloading CSV
    if st.button("📥 Proceed and Download CSV"):
        if not ticker_selection:
            st.warning("Please select at least one ticker.")
        elif start_date >= end_date:
            st.warning("Start date must be before end date.")
        else:
            df = pd.DataFrame()
            for ticker in ticker_selection:
                data = yf.download(ticker, start=start_date, end=end_date)
                if 'Close' in data.columns:
                    df[ticker] = data['Close']
                else:
                    st.warning(f"Warning: No 'Close' data available for {ticker}")

            if not df.empty:
                df.index.name = "Date"
                st.success("Data successfully downloaded!")
                csv = df.to_csv().encode('utf-8')
                st.download_button("⬇️ Download CSV", data=csv, file_name="stock_prices.csv", mime="text/csv")
            else:
                st.error("No valid data was retrieved. Try different tickers or dates.")
