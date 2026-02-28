import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
except ImportError:
    VMD_AVAILABLE = False

# 1. 设置网页全局配置
st.set_page_config(page_title="真实交通流 VMD 交互分析", layout="wide")

st.title("🚦 真实交通流 VMD 分解与动态重构 (本地直读+实时解耦版)")
st.markdown("""
**演示说明：** 1. **时间跨度**：您可以在左侧自由切换查看单日 (24h) 或双日 (48h) 的数据。
2. **图层解耦**：勾选图层开关，该模态曲线会**动态加入到红色虚线的重构流量中**。取消勾选相当于将该信号剔除！
3. **动态权重**：拖动滑块可放大或缩小该模态对整体交通流的影响。
---
""")

if not VMD_AVAILABLE:
    st.error("⚠️ 未检测到 vmdpy 库！请在终端运行 `pip install vmdpy` 后刷新页面。")
    st.stop()

# ==========================================
# 缓存函数：极速读取与处理 Excel
# ==========================================
@st.cache_data
def load_and_preprocess(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    dfs = []
    for sheet_name, df in all_sheets.items():
        if '线圈号' in df.columns:
            df = df[df['线圈号'] != 'LoopId']
            dfs.append(df)
            
    if not dfs: return None
    
    full_df = pd.concat(dfs, ignore_index=True)
    cols_to_numeric = ['小客车流量', '大客车流量', '小货车流量', '大货车流量', '拖挂车流量', '摩托车流量', '采集时间']
    for col in cols_to_numeric:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
        
    full_df['Total_Flow'] = (
        full_df['小客车流量'] * 1.0 + full_df['大客车流量'] * 1.5 + 
        full_df['小货车流量'] * 1.0 + full_df['大货车流量'] * 2.0 + 
        full_df['拖挂车流量'] * 3.0 + full_df['摩托车流量'] * 1.0
    )
    
    full_df['Date'] = pd.to_datetime(full_df['采集日期'])
    full_df['Datetime'] = full_df['Date'] + pd.to_timedelta((full_df['采集时间'] - 1) * 5, unit='m')
    
    agg_df = full_df.groupby('Datetime')['Total_Flow'].sum().reset_index()
    full_idx = pd.date_range(start=agg_df['Datetime'].min(), end=agg_df['Datetime'].max(), freq='5T')
    agg_df = agg_df.set_index('Datetime').reindex(full_idx, fill_value=0).reset_index()
    agg_df.columns = ['Datetime', 'Flow']
    
    return agg_df

# ==========================================
# 缓存函数：实时对指定数据跑 VMD
# ==========================================
@st.cache_data
def run_vmd(y_array):
    alpha = 2000
    tau = 0
    K = 6
    DC = 0
    init = 1    
    tol = 1e-7
    u, u_hat, omega = VMD(y_array, alpha, tau, K, DC, init, tol)
    return u

# ==========================================
# 侧边栏：UI 控件
# ==========================================
st.sidebar.header("📁 1. 数据源设置")
year = st.sidebar.number_input("数据年份", value=2017, step=1)
file_path = f"地面交叉口5分钟流量信息_{year}.xlsx"

if not os.path.exists(file_path):
    st.error(f"❌ 找不到文件：`{file_path}`。请确认它已被放入本项目的根目录下！")
    st.stop()

with st.spinner("正在解析交通流数据，请稍候..."):
    df = load_and_preprocess(file_path)
    
if df is not None:
    unique_dates = pd.Series(df['Datetime'].dt.date.unique()).dropna()
    
    # 动态选择时长 (24h or 48h)
    duration_option = st.sidebar.radio("⏱️ 选择数据截取时长", ["24 小时 (单日)", "48 小时 (双日)"], index=1)
    days_to_add = 1 if "24" in duration_option else 2
    
    # 如果选双日，最后一天不能作为起点
    valid_start_dates = unique_dates[:-1] if days_to_add == 2 else unique_dates
    
    selected_date = st.sidebar.selectbox("📅 选择起始日期", valid_start_dates)
    
    start_ts = pd.to_datetime(selected_date)
    end_ts = start_ts + pd.Timedelta(days=days_to_add)
    mask = (df['Datetime'] >= start_ts) & (df['Datetime'] < end_ts)
    target_df = df[mask].copy()
    
    y_real = target_df['Flow'].values
    x_axis = target_df['Datetime']
    
    u = run_vmd(y_real)
    
    # --- UI: 开关与权重控制 ---
    st.sidebar.markdown("---")
    st.sidebar.header("👁️ 2. 核心图层解耦开关")
    st.sidebar.markdown("⚠️ **勾选将激活该图层，并将其注入红色的重构线中。**")
    
    show_orig = st.sidebar.checkbox("⚫ 显示真实原始车流 (黑实线)", True)
    show_recon = st.sidebar.checkbox("🔴 显示动态重构车流 (红虚线)", True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 3. IMF 模态提取与权重池")
    
    show_imf1 = st.sidebar.checkbox("🟦 参与重构并显示: IMF 1 (基准趋势)", True)
    w1 = st.sidebar.slider("IMF 1 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf1 else 0.0
    
    show_imf2 = st.sidebar.checkbox("🟩 参与重构并显示: IMF 2 (昼夜潮汐)", True)
    w2 = st.sidebar.slider("IMF 2 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf2 else 0.0
    
    show_imf3 = st.sidebar.checkbox("🟨 参与重构并显示: IMF 3 (中低频波动)", False)
    w3 = st.sidebar.slider("IMF 3 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf3 else 0.0
    
    show_imf4 = st.sidebar.checkbox("🟧 参与重构并显示: IMF 4 (中频波动)", False)
    w4 = st.sidebar.slider("IMF 4 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf4 else 0.0
    
    show_imf5 = st.sidebar.checkbox("🟥 参与重构并显示: IMF 5 (中高频波动)", False)
    w5 = st.sidebar.slider("IMF 5 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf5 else 0.0
    
    show_imf6 = st.sidebar.checkbox("🟪 参与重构并显示: IMF 6 (高频噪音)", False)
    w6 = st.sidebar.slider("IMF 6 权重倍数", 0.0, 2.0, 1.0, 0.1) if show_imf6 else 0.0

    # ==========================================
    # 数据重构与绘图计算
    # ==========================================
    shows = [show_imf1, show_imf2, show_imf3, show_imf4, show_imf5, show_imf6]
    weights = [w1, w2, w3, w4, w5, w6]
    
    # 物理重构
    imfs_weighted = [u[i] * weights[i] for i in range(6)]
    reconstructed_signal = np.maximum(0, np.sum(imfs_weighted, axis=0))

    # 🌟 修复：必须先设置 style，然后再应用中文字体，否则会被 style 覆盖！
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  
    
    # 设置输出图像 DPI = 200
    fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
    
    colors = ['#1f77b4', '#2ca02c', '#bcbd22', '#ff7f0e', '#d62728', '#9467bd']
    labels = ["IMF 1 (基准趋势)", "IMF 2 (昼夜潮汐)", "IMF 3 (中低频)", "IMF 4 (中频)", "IMF 5 (中高频)", "IMF 6 (高频噪音)"]

    plotted_data = []

    # 黑实线，不加粗
    if show_orig:
        ax.plot(x_axis, y_real, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label="Original Real Traffic (原始真实流量)")
        plotted_data.append(y_real)
        
    for i in range(6):
        if shows[i]:
            ax.plot(x_axis, imfs_weighted[i], color=colors[i], linewidth=1.5, alpha=0.8, label=f"{labels[i]} (权重={weights[i]:.1f})")
            plotted_data.append(imfs_weighted[i])

    # 红虚线，不加粗
    if show_recon:
        ax.plot(x_axis, reconstructed_signal, color='red', linestyle='-', linewidth=1.5, label="Dynamic Reconstructed (动态重构流量)")
        plotted_data.append(reconstructed_signal)

    title_duration = "单日 24h" if days_to_add == 1 else "双日 48h"
    ax.set_title(f"真实交通流 VMD 实时解耦与重构演示 ({selected_date} | {title_duration})", fontsize=16, fontweight='bold')
    ax.set_xlabel("时间 (Date & Time)", fontsize=12)
    ax.set_ylabel("交通流量 (PCU)", fontsize=12)
    
    # 动态 X 轴刻度间距 (单日3小时一标，双日6小时一标)
    interval_hours = 3 if days_to_add == 1 else 6
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    plt.xticks(rotation=45)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, frameon=True, shadow=True)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    
    if plotted_data:
        global_min = min([np.min(arr) for arr in plotted_data])
        global_max = max([np.max(arr) for arr in plotted_data])
        y_range = global_max - global_min if global_max != global_min else 10
        ax.set_ylim(global_min - y_range * 0.1, global_max + y_range * 0.1)

    ax.grid(True, alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # 渲染至网页
    st.pyplot(fig)

