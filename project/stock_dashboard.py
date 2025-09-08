

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go


def get_stock_data(ticker, period='6mo', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def plot_candlestick(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        line=dict(color='blue', width=1),
        name='SMA 20'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_20'],
        line=dict(color='orange', width=1),
        name='EMA 20'
    ))

    fig.update_layout(
        title=f'{ticker.upper()} - Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        line=dict(color='green', width=2),
        name='RSI'
    ))
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        shapes=[
            dict(type='line', y0=70, y1=70, x0=df.index.min(), x1=df.index.max(), line=dict(dash='dash', color='red')),
            dict(type='line', y0=30, y1=30, x0=df.index.min(), x1=df.index.max(), line=dict(dash='dash', color='blue')),
        ]
    )
    return fig

def get_performance_summary(df):
    df = df.sort_index()

    try:
        latest_close = float(df['Close'].dropna().values[-1])
        previous_close = float(df['Close'].dropna().values[-2])
        pct_change = ((latest_close - previous_close) / previous_close) * 100
    except (IndexError, ValueError):
        latest_close = previous_close = pct_change = None

    try:
        rsi = float(df['RSI'].dropna().values[-1])
    except (IndexError, ValueError):
        rsi = None

    return {
        'Latest Close': f"${latest_close:.2f}" if latest_close is not None else "N/A",
        'Daily Change': f"{pct_change:.2f}%" if pct_change is not None else "N/A",
        'RSI': f"{rsi:.2f}" if rsi is not None else "N/A"
    }



st.set_page_config(page_title=" Stock Analysis Dashboard", layout="wide")

st.title(" Stock Analysis Dashboard")
st.markdown("Analyze stock trends with candlestick charts and technical indicators (SMA, EMA, RSI).")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

col1, col2 = st.columns(2)
with col1:
    period = st.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=2)
with col2:
    interval = st.selectbox("Interval", ['1d', '1h', '1wk'], index=0)

if ticker:
    data = get_stock_data(ticker, period, interval)
    data = calculate_indicators(data)

    st.subheader(f" {ticker.upper()} Performance Summary")
    summary = get_performance_summary(data)
    st.metric("Latest Close", summary['Latest Close'])
    st.metric("Daily Change", summary['Daily Change'])
    st.metric("RSI", summary['RSI'])

    st.plotly_chart(plot_candlestick(data, ticker), use_container_width=True)
    st.plotly_chart(plot_rsi(data), use_container_width=True)

    st.subheader(" Historical Data")
    st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI']].tail(30))

    # CSV Export option
    csv = data.to_csv().encode('utf-8')
    st.download_button(
        label=" Download CSV",
        data=csv,
        file_name=f"{ticker}_historical_data.csv",
        mime='text/csv'
    )
