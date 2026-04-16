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
    h1 { color: #000000; font-weight: 900; text-align: center; padding-bottom: 10px; }
    
    /* 隱藏原生 Metric 的 delta 標籤 */
    [data-testid="stMetricDelta"] { display: none; }
    
    /* 設定指標數值大小，不使用粗體 */
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 400 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 顏色渲染工具函數 ---
def color_metric(label, value, is_percent=True, force_black=False):
    if force_black:
        color = "#000000"
    else:
        color = "#FF0000" if value > 0 else "#008000" if value < 0 else "#000000"
    
    val_display = f"{value*100:.2f}%" if is_percent else f"{value:.2f}"
    
    st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;">
            <p style="color: #64748b; font-size: 14px; margin-bottom: 5px;">{label}</p>
            <p style="color: {color}; font-size: 28px; font-weight: 400; margin: 0;">{val_display}</p>
        </div>
    """, unsafe_allow_html=True)

# --- 2. 資料載入 ---
@st.cache_data
def load_data():
    target_path = "四家七年.csv" 
    if not os.path.exists(target_path):
        st.error(f"❌ 找不到檔案：{target_path}")
        return None
    try:
        df = pd.read_csv(target_path, encoding='cp950', skiprows=1)
    except:
        df = pd.read_csv(target_path, encoding='utf-8-sig', skiprows=1)
    
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()

    df['MDATE'] = pd.to_datetime(df['MDATE'], errors='coerce', format='mixed')
    df = df.dropna(subset=['MDATE'])
    df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
    
    df['搜尋名稱'] = df['COID'] + " " + df['Name']
    pivot_df = df.pivot_table(index='MDATE', columns='搜尋名稱', values='CLOSE', aggfunc='last')
    return pivot_df.sort_index().interpolate().ffill().bfill()

data = load_data()

if data is not None and not data.empty:
    # --- 3. 側邊欄：修正 0050 偵測邏輯 ---
    with st.sidebar:
        if os.path.exists("1223.png"):
            st.image("1223.png", width=200)
        
        st.header("📊 參數設定")
        all_cols = data.columns.tolist()
        
        # 強化版 0050 搜尋邏輯
        benchmark_name = None
        for col in all_cols:
            if any(keyword in col for keyword in ["0050", "元大台灣50", "大盤", "Taiwan 50"]):
                benchmark_name = col
                break
        
        trading_stocks = [s for s in all_cols if s != benchmark_name]

        st.write("🔍 **搜尋標的**")
        s1 = st.selectbox("標的股票 A", trading_stocks, index=0)
        s2 = st.selectbox("標的股票 B", trading_stocks, index=min(1, len(trading_stocks)-1))
        
        threshold = st.slider("開倉門檻", 0.0, 1.0, 0.5, 0.05)
        rf_rate = st.sidebar.slider("無風險利率 (%)", 0.0, 5.0, 1.6, 0.1) / 100
        
        # 強制顯示勾選框區域
        st.write("---")
        if benchmark_name:
            show_benchmark = st.checkbox(f"📈 加入大盤績效", value=True)
        else:
            # 如果還是找不到 0050，顯示提示
            st.warning("⚠️ CSV 內找不到名為 0050 的標的")
            show_benchmark = False

    # --- 4. 策略邏輯 ---
    st.title("配對交易量化儀表板")
    
    try:
        start_dt = data.index.min()
        split_dt = start_dt + pd.Timedelta(days=730)
        
        daily_ret = data[[s1, s2]].pct_change().fillna(0)
        m_ret = data[[s1, s2]].resample('ME').last().pct_change().fillna(0)

        results = []
        for j in range(1, len(m_ret)):
            curr_dt = m_ret.index[j]
            if curr_dt < split_dt: continue
            
            prev_m_idx = m_ret.index[j-1].strftime('%Y-%m')
            m_daily = daily_ret.loc[prev_m_idx]
            prev_corr = m_daily[s1].corr(m_daily[s2]) if not m_daily.empty else 0.0
            prev_corr = 0.0 if pd.isna(prev_corr) else prev_corr
            
            r1, r2 = m_ret.iloc[j][s1], m_ret.iloc[j][s2]
            r1_p, r2_p = m_ret.iloc[j-1][s1], m_ret.iloc[j-1][s2]
            
            if prev_corr < threshold:
                action, strat_ret = "觀望期", 0.0
            else:
                if r1_p > r2_p:
                    action, strat_ret = f"買 {s2} / 賣 {s1}", r2 - r1
                else:
                    action, strat_ret = f"買 {s1} / 賣 {s2}", r1 - r2
            
            results.append({
                "日期": curr_dt, "月份": curr_dt.strftime('%Y-%m'), "相關性": prev_corr,
                f"{s1}報酬": r1, f"{s2}報酬": r2, "交易動作": action, "策略獲利": strat_ret
            })

        res_df = pd.DataFrame(results)

        if not res_df.empty:
            res_df['累積報酬'] = (1 + res_df['策略獲利']).cumprod() - 1
            cum_ret = res_df['累積報酬'].iloc[-1]
            total_months = len(res_df)
            ann_ret = ((1 + cum_ret) ** (12 / total_months)) - 1
            res_df['Equity'] = (1 + res_df['策略獲利']).cumprod()
            mdd = ((res_df['Equity'] - res_df['Equity'].cummax()) / res_df['Equity'].cummax()).min()
            sharpe = (res_df['策略獲利'].mean() / res_df['策略獲利'].std()) * (12**0.5) if res_df['策略獲利'].std() != 0 else 0

            # --- 5. KPI 區塊 ---
            st.subheader("🚩 策略指標")
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: color_metric("累積淨獲利", cum_ret)
            with k2: color_metric("年化報酬率", ann_ret)
            with k3: color_metric("年化夏普比率", sharpe, is_percent=False, force_black=True)
            with k4: color_metric("最高收益", res_df['累積報酬'].max())
            with k5: color_metric("最大回撤 (MDD)", mdd)

            st.subheader("📅 年度報酬統計")
            res_df['年份'] = res_df['日期'].dt.year
            annual_stats = res_df.groupby('年份')['策略獲利'].apply(lambda x: (1 + x).prod() - 1).sort_index(ascending=False)
            y_cols = st.columns(5)
            for i, year in enumerate(annual_stats.index[:5]):
                with y_cols[i]: color_metric(f"{year} 年度", annual_stats[year])

            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📈 績效走勢", "📋 交易明細", "📊 相關性"])

            with tab1:
                fig = go.Figure()
                c_color = '#FF0000' if cum_ret >= 0 else '#008000'
                fig.add_trace(go.Scatter(x=res_df['月份'], y=res_df['累積報酬']*100, name='策略報酬', line=dict(color=c_color, width=4), fill='tozeroy', fillcolor=f'rgba({int(c_color[1:3],16)}, {int(c_color[3:5],16)}, {int(c_color[5:7],16)}, 0.1)'))
                
                if show_benchmark and benchmark_name:
                    b_raw = data[benchmark_name].resample('ME').last()
                    b_start_val = b_raw.loc[:res_df['日期'].iloc[0]].iloc[-1]
                    b_cum = (b_raw.loc[res_df['日期'].iloc[0]:] / b_start_val) - 1
                    fig.add_trace(go.Scatter(x=res_df['月份'], y=b_cum.values[:len(res_df)]*100, name=benchmark_name, line=dict(color='#64748B', dash='dot')))
                
                fig.update_layout(hovermode="x unified", template="plotly_white", height=550)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                def table_color(val):
                    if isinstance(val, (int, float)):
                        if val > 0: return 'color: #FF0000;'
                        if val < 0: return 'color: #008000;'
                    return 'color: #000000;'
                
                cols_to_style = [f"{s1}報酬", f"{s2}報酬", '策略獲利', '累積報酬']
                styled_df = res_df[['月份', '相關性', f"{s1}報酬", f"{s2}報酬", '交易動作', '策略獲利', '累積報酬']].style.map(table_color, subset=cols_to_style).format({c: "{:.2%}" for c in cols_to_style}).format({"相關性": "{:.4f}"})
                st.dataframe(styled_df, use_container_width=True, height=600)

            with tab3:
                colors = ['#FF0000' if c >= threshold else '#CBD5E1' for c in res_df['相關性']]
                fig_corr = go.Figure(go.Bar(x=res_df['月份'], y=res_df['相關性'], marker_color=colors))
                fig_corr.add_hline(y=threshold, line_dash="dash", line_color="#374151")
                st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"❌ 錯誤：{e}")
