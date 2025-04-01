import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 加载 .env 文件
load_dotenv()

# 从环境变量中读取配置
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# 拼接连接字符串
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 创建 SQLAlchemy 引擎
engine = create_engine(DB_URI)

# ================== 指标计算模块 ==================
def calculate_macd(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEA']
    df['MACD_Signal'] = np.where(df['DIF'] > df['DEA'], '红', '绿')
    return df

def calculate_kdj(df):
    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-8) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['KDJ_Signal'] = np.where(df['K'] > df['D'], '红', '绿')
    return df

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Signal'] = np.where(df['RSI'] > 50, '红', '绿')
    return df

def calculate_lwr(df, period=14):
    highest = df['high'].rolling(period).max()
    lowest = df['low'].rolling(period).min()
    df['LWR'] = (highest - df['close']) / (highest - lowest + 1e-8) * 100
    df['LWR_Signal'] = np.where(df['LWR'] < 20, '红', '绿')
    return df

def calculate_bbi(df):
    df['BBI'] = (df['close'].rolling(3).mean() +
                 df['close'].rolling(6).mean() +
                 df['close'].rolling(12).mean() +
                 df['close'].rolling(24).mean()) / 4
    df['BBI_Signal'] = np.where(df['close'] > df['BBI'], '红', '绿')
    return df

def calculate_zlmm(df):
    df['MainNet'] = (df['close'] - df['open']) * df['volume']
    df['ZLMM_Signal'] = np.where(df['MainNet'] > 0, '红', '绿')
    return df

# ================== 策略逻辑模块 ==================
def six_sword_strategy(df):
    df = df.copy()
    df = (df.pipe(calculate_macd)
          .pipe(calculate_kdj)
          .pipe(calculate_rsi)
          .pipe(calculate_lwr)
          .pipe(calculate_bbi)
          .pipe(calculate_zlmm))

    buy_condition = (df['MACD_Signal'] == '红') & \
                    (df['KDJ_Signal'] == '红') & \
                    (df['RSI_Signal'] == '红') & \
                    (df['LWR_Signal'] == '红') & \
                    (df['BBI_Signal'] == '红') & \
                    (df['ZLMM_Signal'] == '红')

    signal_cols = ['MACD_Signal', 'KDJ_Signal', 'RSI_Signal',
                   'LWR_Signal', 'BBI_Signal', 'ZLMM_Signal']
    sell_condition = (df[signal_cols] == '绿').sum(axis=1) >= 3

    df['raw_buy'] = buy_condition.shift(1).fillna(False)
    df['raw_sell'] = sell_condition.shift(1).fillna(False)
    return df

def filter_signals(df):
    df = df.copy()
    df['clean_buy'] = False
    df['clean_sell'] = False
    hold_position = False

    for i in range(len(df)):
        if df.at[i, 'raw_buy'] and not hold_position:
            df.at[i, 'clean_buy'] = True
            hold_position = True
        if df.at[i, 'raw_sell'] and hold_position:
            df.at[i, 'clean_sell'] = True
            hold_position = False
    return df

# ================== 可视化模块 ==================
def plot_correct_kline(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing={'line': {'color': '#EF5350'}, 'fillcolor': '#EF5350'},
        decreasing={'line': {'color': '#26A69A'}, 'fillcolor': '#26A69A'},
        name='K线'
    ))

    buy_points = df[df['clean_buy']]
    if not buy_points.empty:
        fig.add_trace(go.Scatter(
            x=buy_points['date'],
            y=buy_points['low'] * 0.98,
            mode='markers',
            marker=dict(color='#D32F2F', size=12, symbol='triangle-up'),
            name='买入信号'
        ))

    sell_points = df[df['clean_sell']]
    if not sell_points.empty:
        fig.add_trace(go.Scatter(
            x=sell_points['date'],
            y=sell_points['high'] * 1.02,
            mode='markers',
            marker=dict(color='#388E3C', size=12, symbol='triangle-down'),
            name='卖出信号'
        ))

    fig.update_layout(
        title='六脉神剑策略',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        font=dict(family='Microsoft YaHei')
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== 数据获取模块（SQLAlchemy） ==================
@st.cache_data
def get_stock_data(symbol, start, end):
    try:
        query = text("""
            SELECT `交易日期`, `开盘价`, `最高价`, `最低价`, `收盘价`, `成交量`, `股票代码`
            FROM stock_daily_data
            WHERE `股票代码` = :symbol
              AND `交易日期` BETWEEN :start AND :end
            ORDER BY `交易日期`
        """)
        df = pd.read_sql(query, engine, params={
            "symbol": symbol,
            "start": start,
            "end": end
        }, parse_dates=['交易日期'])

        # 重命名字段为英文以适配策略
        df = df.rename(columns={
            '交易日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume'
        })
        return df
    except Exception as e:
        st.error(f"数据库读取失败: {str(e)}")
        return pd.DataFrame()

# ================== Streamlit主程序 ==================
def main():
    st.set_page_config(page_title="六脉神剑", layout="wide")
    st.title("🗡️ 六脉神剑策略系统")

    with st.sidebar:
        st.header("参数设置")
        symbol = st.text_input("股票代码（如 600000）", "600000")
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=90))
        end_date = st.date_input("结束日期", datetime.now())

    df = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df.empty:
        st.stop()

    df = six_sword_strategy(df)
    df = filter_signals(df)

    st.subheader("K线图与交易信号")
    plot_correct_kline(df)

    col1, col2 = st.columns(2)
    col1.metric("买入信号次数", df['clean_buy'].sum())
    col2.metric("卖出信号次数", df['clean_sell'].sum())

if __name__ == "__main__":
    main()