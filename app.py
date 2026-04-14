# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 網頁配置 ---
st.set_page_config(page_title="量化配對交易系統", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    h1 { color: #000000; font-weight: 900; text-align: center; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 資料載入 ---
@st.cache_data
def load_data():
    target_path = "四家七年.csv" 
    if not os.path.exists(target_path):
        st.error(f"❌ 找不到 {target_path}。")
        return None
    try:
        df = pd.read_csv(target_path, encoding='cp950', skiprows=1)
    except:
        df = pd.read_csv(target_path, encoding='utf-8-sig', skiprows=1)
    
    # 清除欄位名稱的空格與特殊字元
    df.columns = df.columns.str.replace(r'\s+', '', regex=True)
    
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype(str).str.strip()
        pivot_df = df.pivot(index='MDATE', columns='Name', values='CLOSE')
        pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y%m%d')
        return pivot_df.sort_index().interpolate().ffill().bfill()
    return None

data = load_data()

if data is not None:
    # --- 3. 側邊欄 ---
    with st.sidebar:
        if os.path.exists("1223.png"):
            st.image("1223.png", width=200)
        
        st.header("📊 參數設定")
        all_cols = data.columns.tolist()
        
        # 加強版基準線偵測
        benchmark_name = next((c for c in all_cols if any(k in c for k in ["元大台灣50", "0050", "大盤"])), None)
        trading_stocks = [s for s in all_cols if s != benchmark_name]
        
        s1 = st.selectbox("標的股票 A", trading_stocks, index=0)
        s2 = st.selectbox("標的股票 B", trading_stocks, index=min(1, len(trading_stocks)-1))
        threshold = st.slider("開倉門檻", 0.0, 1.0, 0.5, 0.05)
        rf_rate = st.sidebar.slider("無風險利率 (%)", 0.0, 5.0, 1.6, 0.1) / 100
        
        # 確保基準線存在才顯示勾選框
        show_benchmark = False
        if benchmark_name:
            st.write("---")
            show_benchmark = st.checkbox(f"📈 疊加 {benchmark_name} 績效", value=False)

    # --- 4. 策略邏輯 ---
    start_dt = data.index.min()
    split_dt = start_dt + pd.Timedelta(days=730)
    
    daily_ret = data[[s1, s2]].pct_change().fillna(0)
    m_ret = data[[s1, s2]].resample('ME').last().pct_change().fillna(0)

    results = []
    for j in range(1, len(m_ret)):
        curr_dt = m_ret.index[j]
        if curr_dt < split_dt: continue
        
        prev_m_idx = m_ret.index[j-1].strftime('%Y-%m')
        ret_a, ret_b = m_ret.iloc[j][s1], m_ret.iloc[j][s2]
        r1_p, r2_p = m_ret.iloc[j-1][s1], m_ret.iloc[j-1][s2]
        
        prev_corr = daily_ret.loc[prev_m_idx][s1].corr(daily_ret.loc[prev_m_idx][s2])
        prev_corr = 0.0 if pd.isna(prev_corr) else prev_corr
        
        if prev_corr < threshold:
            action, strat_ret = "觀望期", 0.0
        else:
            if r1_p > r2_p:
                action, strat_ret = f"買{s2}/賣{s1}", ret_b - ret_a
            else:
                action, strat_ret = f"買{s1}/賣{s2}", ret_a - ret_b
        
        results.append({
            "月份": curr_dt.strftime('%Y-%m'),
            "相關性": prev_corr,
            f"{s1}報酬": ret_a,
            f"{s2}報酬": ret_b,
            "交易動作": action,
            "策略獲利": strat_ret
        })

    res_df = pd.DataFrame(results)
    res_df['累積報酬'] = (1 + res_df['策略獲利']).cumprod() - 1

    # --- 5. 指標計算 ---
    max_cum_profit = res_df['累積報酬'].max()
    res_df['Equity'] = (1 + res_df['策略獲利']).cumprod()
    res_df['Peak'] = res_df['Equity'].cummax()
    mdd = ((res_df['Equity'] - res_df['Peak']) / res_df['Peak']).min()
    sharpe = ((res_df['策略獲利'] - (rf_rate/12)).mean() / res_df['策略獲利'].std()) * (12**0.5) if res_df['策略獲利'].std() != 0 else 0

    # --- 6. 畫面呈現 ---
    st.title("配對交易量化儀表板")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("累積淨獲利", f"{res_df['累積報酬'].iloc[-1]*100:.2f}%", delta=f"{res_df['策略獲利'].iloc[-1]*100:.2f}%", delta_color="inverse")
    c2.metric("年化夏普比率", f"{sharpe:.2f}")
    c3.metric("歷史最高收益", f"{max_cum_profit*100:.2f}%")
    c4.metric("最大回撤 (MDD)", f"{mdd*100:.2f}%")
    c5.metric("回測月份數", f"{len(res_df)} M")

    tab1, tab2, tab3 = st.tabs(["📈 績效走勢", "📋 交易明細", "📊 相關性觀測"])

    with tab1:
        st.subheader("策略收益曲線")
        fig = go.Figure()
        f_val = res_df['累積報酬'].iloc[-1]
        c_color = '#FF0000' if f_val >= 0 else '#008000'
        fig.add_trace(go.Scatter(x=res_df['月份'], y=res_df['累積報酬']*100, name='策略報酬', line=dict(color=c_color, width=4), fill='tozeroy', fillcolor=f'rgba({int(c_color[1:3],16)}, {int(c_color[3:5],16)}, {int(c_color[5:7],16)}, 0.1)'))
        
        if show_benchmark and benchmark_name:
            b_data = data[benchmark_name].resample('ME').last().pct_change().loc[res_df['月份'].iloc[0]:]
            b_cum = (1 + b_data).cumprod() - 1
            fig.add_trace(go.Scatter(x=res_df['月份'], y=b_cum.values[:len(res_df)]*100, name=benchmark_name, line=dict(color='#64748B', dash='dot')))
        
        fig.update_layout(hovermode="x unified", template="plotly_white", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("詳細交易明細")
        def color_taiwan(val):
            if isinstance(val, (int, float)):
                return f'color: {"#FF0000" if val > 0 else "#008000" if val < 0 else "#000000"}'
            return ''
        
        cols_fmt = [f"{s1}報酬", f"{s2}報酬", '策略獲利', '累積報酬']
        styled_df = res_df[['月份', '相關性', f"{s1}報酬", f"{s2}報酬", '交易動作', '策略獲利', '累積報酬']].style.map(color_taiwan, subset=cols_fmt).format({c: "{:.2%}" for c in cols_fmt}).format({"相關性": "{:.4f}"})
        st.dataframe(styled_df, use_container_width=True, height=600)

    with tab3:
        st.subheader("月度相關係數監測")
        colors = ['#FF0000' if c >= threshold else '#CBD5E1' for c in res_df['相關性']]
        fig_corr = go.Figure(go.Bar(x=res_df['月份'], y=res_df['相關性'], marker_color=colors))
        fig_corr.add_hline(y=threshold, line_dash="dash", line_color="#374151")
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("請確認 CSV 檔案中包含標的名稱及數據。")
