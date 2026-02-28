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

st.title("🚦 真实交通流 VMD 分解与动态重构 (本地直读版)")
st.markdown("""
**演示说明：** 本系统自动读取本地仓库中的真实交通流数据，并允许您选择任意 **连续两天** 进行 VMD 分解。
通过调节右侧各模态的【权重倍数】（默认 1.0），您可以实时观察去除噪音或放大趋势后，重构的真实交通流会发生什么变化。
---
""")

if not VMD_AVAILABLE:
    st.error("⚠️ 未检测到 vmdpy 库！请在终端运行 `pip install vmdpy` 后刷新页面。")
    st.stop()

# ==========================================
# 缓存函数：极速读取与处理 Excel (改为读取本地路径)
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
# 缓存函数：实时对指定的 2 天数据跑 VMD
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
# 动态年份选择，默认读取 2017 年的表
year = st.sidebar.number_input("数据年份", value=2017, step=1)
file_path = f"地面交叉口5分钟流量信息_{year}.xlsx"

if not os.path.exists(file_path):
    st.error(f"❌ 找不到文件：`{file_path}`。请确认它已被放入与本脚本相同的目录下！")
    st.stop()
else:
    st.sidebar.success(f"✅ 成功锁定本地文件：{file_path}")

with st.spinner("正在解析交通流数据，请稍候..."):
    df = load_and_preprocess(file_path)
    
if df is not None:
    # 提取有效日期
    unique_dates = pd.Series(df['Datetime'].dt.date.unique()).dropna()
    valid_start_dates = unique_dates[:-1] # 留出第二天
    
    selected_date = st.sidebar.selectbox("📅 选择起始日期 (将自动截取连续 48 小时)", valid_start_dates)
    
    # 截取两天的真实数据
    start_ts = pd.to_datetime(selected_date)
    end_ts = start_ts + pd.Timedelta(days=2)
    mask = (df['Datetime'] >= start_ts) & (df['Datetime'] < end_ts)
    two_days_df = df[mask].copy()
    
    y_real = two_days_df['Flow'].values
    x_axis = two_days_df['Datetime']
    
    # 实时进行 VMD 分解
    u = run_vmd(y_real)
    
    # --- UI: 权重与图层控制 ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 2. IMF 权重调节 (倍数)")
    st.sidebar.markdown("设为 0 表示剔除该信号，设为 1 表示维持原状。")
    
    w1 = st.sidebar.slider("🟦 IMF 1 (基准趋势) 权重", 0.0, 2.0, 1.0, 0.1)
    w2 = st.sidebar.slider("🟩 IMF 2 (次低频潮汐) 权重", 0.0, 2.0, 1.0, 0.1)
    w3 = st.sidebar.slider("🟨 IMF 3 (中低频波动) 权重", 0.0, 2.0, 1.0, 0.1)
    w4 = st.sidebar.slider("🟧 IMF 4 (中频波动) 权重", 0.0, 2.0, 1.0, 0.1)
    w5 = st.sidebar.slider("🟥 IMF 5 (中高频波动) 权重", 0.0, 2.0, 1.0, 0.1)
    w6 = st.sidebar.slider("🟪 IMF 6 (高频噪音) 权重", 0.0, 2.0, 1.0, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("👁️ 3. 图层显示开关")
    show_orig = st.sidebar.checkbox("⚫ 显示真实原始车流", True)
    show_recon = st.sidebar.checkbox("🔴 显示加权重构后的新车流", True)
    show_imf1 = st.sidebar.checkbox("🟦 显示 IMF 1", True)
    show_imf2 = st.sidebar.checkbox("🟩 显示 IMF 2", True)
    show_imf3 = st.sidebar.checkbox("🟨 显示 IMF 3", False)
    show_imf4 = st.sidebar.checkbox("🟧 显示 IMF 4", False)
    show_imf5 = st.sidebar.checkbox("🟥 显示 IMF 5", False)
    show_imf6 = st.sidebar.checkbox("🟪 显示 IMF 6", False)

    # ==========================================
    # 数据重构与绘图
    # ==========================================
    # 将原始 IMF 乘以上述调节的倍数权重
    imfs_mod = [u[i] * w for i, w in enumerate([w1, w2, w3, w4, w5, w6])]
    # 物理重构并截断负数 (真实车流不能小于0)
    reconstructed_signal = np.maximum(0, np.sum(imfs_mod, axis=0))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  

    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ['#1f77b4', '#2ca02c', '#bcbd22', '#ff7f0e', '#d62728', '#9467bd']

    # 🌟 智能收集需要画在图上的数据，用于计算 Y 轴范围
    plotted_data = []

    if show_orig:
        ax.plot(x_axis, y_real, color='black', linewidth=4, alpha=0.3, label="Original Traffic (原始真实流量)")
        plotted_data.append(y_real)
        
    if show_recon:
        ax.plot(x_axis, reconstructed_signal, color='red', linestyle='--', linewidth=2.5, label="Reconstructed Traffic (重构流量)")
        plotted_data.append(reconstructed_signal)

    shows = [show_imf1, show_imf2, show_imf3, show_imf4, show_imf5, show_imf6]
    labels = ["IMF 1 (基准趋势)", "IMF 2 (昼夜潮汐)", "IMF 3 (中低频)", "IMF 4 (中频)", "IMF 5 (中高频)", "IMF 6 (高频噪音)"]
    
    for i in range(6):
        if shows[i]:
            ax.plot(x_axis, imfs_mod[i], color=colors[i], linewidth=2, alpha=0.8, label=labels[i])
            plotted_data.append(imfs_mod[i])

    ax.set_title(f"真实数据 VMD 物理重构 ({selected_date} 连续两日)", fontsize=16, fontweight='bold')
    ax.set_xlabel("时间 (Date & Time)", fontsize=12)
    ax.set_ylabel("交通流量 (PCU)", fontsize=12)
    
    # 优化 X 轴日期时间显示格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6)) # 每 6 小时显示一个刻度
    plt.xticks(rotation=45)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.02), fontsize=11, frameon=True, shadow=True)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # 🌟 核心修复：根据当前勾选的曲线自动计算 Y 轴安全显示范围，再也不出界！
    if plotted_data:
        global_min = min([np.min(arr) for arr in plotted_data])
        global_max = max([np.max(arr) for arr in plotted_data])
        y_range = global_max - global_min if global_max != global_min else 10
        # 上下各留 10% 的边距防止贴边
        ax.set_ylim(global_min - y_range * 0.1, global_max + y_range * 0.1)

    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    st.pyplot(fig)
    
    st.success("""
    💡 **给答辩老师演示的绝佳场景**：
    1. **数据降噪**：向老师说明“模型其实很难预测噪音”，然后把右侧 **IMF 6 的权重拖到 0**，向老师展示红色虚线（重构流）立刻变得光滑平顺，说明我们剔除了无用噪音！
    2. **周期剥离**：把除了 IMF 2 以外的所有权重都归 0，你会得到一个极其完美的“潮汐起伏曲线”，这能强有力地证明 VMD 是如何精准抓取出通勤规律的。
    """)
