# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 網頁配置與 CSS ---
st.set_page_config(page_title="量化配對交易系統", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    h1 {
        color: #000000;
        font-weight: 900;
        text-align: center;
        padding-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 資料載入 (已修正為適用 GitHub 的相對路徑) ---
@st.cache_data
def load_data():
    # 這裡直接讀取跟 app.py 同資料夾下的 CSV
    target_path = "四家七年.csv" 
    
    if not os.path.exists(target_path):
        st.error(f"❌ 找不到 {target_path}。請確認 CSV 檔案已上傳至 GitHub 倉庫根目錄。")
        return None

    try:
        # 先嘗試繁體中文常用編碼
        df = pd.read_csv(target_path, encoding='cp950', skiprows=1)
    except:
        # 若失敗則嘗試帶有簽名檔的 UTF-8
        df = pd.read_csv(target_path, encoding='utf-8-sig', skiprows=1)
    
    df.columns = df.columns.str.strip()
    df['Name'] = df['Name'].astype(str).str.strip()
    
    # 轉換資料結構
    pivot_df = df.pivot(index='MDATE', columns='Name', values='CLOSE')
    pivot_df.index = pd.to_datetime(pivot_df.index, format='%Y%m%d')
    
    # 資料清洗：排序、線性插補、填補缺失值
    return pivot_df.sort_index().interpolate().ffill().bfill()

data = load_data()

if data is not None:
    # --- 3. 側邊欄 (LOGO: 1223.png) ---
    with st.sidebar:
        if os.path.exists("1223.png"):
            st.image("1223.png", width=200)
        else:
            st.info("💡 提示：若要顯示 LOGO，請上傳 1223.png")
        
        st.header("📊 參數設定")
        all_cols = data.columns.tolist()
        
        # 自動偵測台股大盤或 0050 作為基準
        benchmark_name = next((c for c in all_cols if "元大台灣50" in c or "0050" in c or "大盤" in c), None)
        trading_stocks = [s for s in all_cols if s != benchmark_name]
        
        s1 = st.selectbox("標的股票 A", trading_stocks, index=0)
        s2 = st.selectbox("標的股票 B", trading_stocks, index=min(1, len(trading_stocks)-1))
        threshold = st.slider("開倉門檻 (相關係數)", 0.0, 1.0, 0.5, 0.05)
        
        # 無風險利率設定 (用於計算夏普比率)
        rf_rate = st.sidebar.slider("無風險利率 (%)", 0.0, 5.0, 1.6, 0.1) / 100
        
        st.write("---")
        show_benchmark = st.checkbox(f"📈 疊加 {benchmark_name if benchmark_name else '基準'} 績效", value=False) if benchmark_name else False

    # --- 4. 策略邏輯運算 ---
    start_dt = data.index.min()
    split_dt = start_dt + pd.Timedelta(days=730)  # 訓練期 2 年
    
    daily_ret = data[[s1, s2]].pct_change().fillna(0)
    m_data = data[[s1, s2]].resample('ME').last()
    m_ret = m_data.pct_change().fillna(0)

    results = []
    for j in range(1, len(m_ret)):
        curr_dt = m_ret.index[j]
        if curr_dt < split_dt: continue
        
        m_idx = curr_dt.strftime('%Y-%m')
        prev_m_idx = m_ret.index[j-1].strftime('%Y-%m')
        
        # 計算前一月相關性作為開倉依據
        prev_corr = daily_ret.loc[prev_m_idx][s1].corr(daily_ret.loc[prev_m_idx][s2])
        prev_corr = 0.0 if pd.isna(prev_corr) else prev_corr
        
        r1, r2 = m_ret.iloc[j][s1], m_ret.iloc[j][s2]
        r1_p, r2_p = m_ret.iloc[j-1][s1], m_ret.iloc[j-1][s2]
        
        if prev_corr < threshold:
            action, strat_ret = "觀望期", 0.0
        else:
            # 配對交易邏輯：買入上月表現較差者，賣出表現較佳者
            if r1_p > r2_p:
                action, strat_ret = f"買{s2}/賣{s1}", r2 - r1
            else:
                action, strat_ret = f"買{s1}/賣{s2}", r1 - r2
        
        results.append({
            "月份": m_idx,
            "相關性": prev_corr,
            "交易動作": action,
            "策略獲利": strat_ret
        })

    res_df = pd.DataFrame(results)
    res_df['累積報酬'] = (1 + res_df['策略獲利']).cumprod() - 1

    # --- 5. 專業風控指標計算 ---
    # A. 歷史最高收益
    max_cum_profit = res_df['累積報酬'].max()

    # B. 最大回撤 (MDD)
    res_df['Equity'] = (1 + res_df['策略獲利']).cumprod()
    res_df['Peak'] = res_df['Equity'].cummax()
    res_df['Drawdown'] = (res_df['Equity'] - res_df['Peak']) / res_df['Peak']
    mdd = res_df['Drawdown'].min()

    # C. 年化夏普比率 (Sharpe Ratio)
    monthly_rf = rf_rate / 12
    excess_ret = res_df['策略獲利'] - monthly_rf
    if res_df['策略獲利'].std() != 0:
        sharpe = (excess_ret.mean() / res_df['策略獲利'].std()) * (12**0.5)
    else:
        sharpe = 0.0

    # --- 6. 畫面呈現 ---
    st.title("配對交易量化儀表板")
    
    # 五欄位 KPI 展示
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("累積淨獲利", f"{res_df['累積報酬'].iloc[-1]*100:.2f}%", 
              delta=f"{res_df['策略獲利'].iloc[-1]*100:.2f}% (月)", delta_color="inverse")
    c2.metric("年化夏普比率", f"{sharpe:.2f}")
    c3.metric("歷史最高收益", f"{max_cum_profit*100:.2f}%")
    c4.metric("最大回撤 (MDD)", f"{mdd*100:.2f}%")
    c5.metric("回測月份數", f"{len(res_df)} M")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 績效走勢", "📋 交易明細", "📊 相關性觀測"])

    with tab1:
        st.subheader("累積收益曲線 (漲紅跌綠)")
        fig = go.Figure()
        
        final_val = res_df['累積報酬'].iloc[-1]
        curve_color = '#FF0000' if final_val >= 0 else '#008000'
        fill_color = 'rgba(255, 0, 0, 0.1)' if final_val >= 0 else 'rgba(0, 128, 0, 0.1)'
        
        fig.add_trace(go.Scatter(
            x=res_df['月份'], y=res_df['累積報酬']*100,
            name='策略報酬', line=dict(color=curve_color, width=4),
            fill='tozeroy', fillcolor=fill_color
        ))

        if show_benchmark and benchmark_name:
            # 基準線累積報酬
            b_data = data[benchmark_name].resample('ME').last().pct_change().loc[res_df['月份'].iloc[0]:]
            b_cum = (1 + b_data).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=res_df['月份'], y=b_cum.values[:len(res_df)]*100, 
                name=benchmark_name, line=dict(color='#64748B', width=2, dash='dot')
            ))
        
        fig.update_layout(hovermode="x unified", template="plotly_white", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("詳細交易明細 (紅漲綠跌)")
        def color_taiwan(val):
            if isinstance(val, (int, float)):
                return f'color: {"#FF0000" if val > 0 else "#008000" if val < 0 else "#000000"}'
            return ''
        
        # 只顯示重要欄位，隱藏中間運算過程
        display_df = res_df[['月份', '相關性', '交易動作', '策略獲利', '累積報酬']]
        
        try:
            styled_df = display_df.style.map(color_taiwan, subset=['策略獲利', '累積報酬'])\
                                        .format({"策略獲利": "{:.2%}", "累積報酬": "{:.2%}", "相關性": "{:.4f}"})
        except AttributeError:
            styled_df = display_df.style.applymap(color_taiwan, subset=['策略獲利', '累積報酬'])\
                                        .format({"策略獲利": "{:.2%}", "累積報酬": "{:.2%}", "相關性": "{:.4f}"})
        st.dataframe(styled_df, use_container_width=True, height=600)

    with tab3:
        st.subheader("月度相關係數監測")
        # 相關性長條圖：過門檻顯示紅色，未過顯示灰色
        colors = ['#FF0000' if c >= threshold else '#CBD5E1' for c in res_df['相關性']]
        fig_corr = go.Figure(go.Bar(x=res_df['月份'], y=res_df['相關性'], marker_color=colors))
        fig_corr.add_hline(y=threshold, line_dash="dash", line_color="#374151", annotation_text="開倉門檻")
        fig_corr.update_layout(yaxis_range=[-1, 1], template="plotly_white", height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("請將您的 CSV 資料檔上傳至 GitHub 以供讀取。")
