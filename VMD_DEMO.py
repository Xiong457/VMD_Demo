import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # 🌟 引入高级网格排版工具
import matplotlib.font_manager as fm
# 1. 设置软件网页的全局配置
st.set_page_config(page_title="VMD 模态分解交互模拟器", layout="wide")

# 2. 软件标题与说明
st.title("🚦 交通流 VMD 变分模态分解原理演示系统 (紧凑仪表盘版)")
st.markdown("""
**演示说明：** 真实的交通流往往是极其混乱的混合信号。本系统模拟了 VMD 算法的核心思想：
将复杂的原始车流，逆向解耦为 6 个不同频率的本征模态函数 (IMF)。
👉 *请在左侧控制台拖动滑块，实时观察各模态对整体交通流波形的影响。*
---
""")

# 3. 侧边栏：参数控制台 
st.sidebar.header("⚙️ IMF 振幅控制台")
st.sidebar.markdown("调节各项模态的剧烈程度：")

amp1 = st.sidebar.slider("🟦 IMF 1 (主干趋势) 幅度", 10, 100, 60, step=5)
amp2 = st.sidebar.slider("🟩 IMF 2 (次低频) 幅度", 0, 80, 40, step=5)
amp3 = st.sidebar.slider("🟨 IMF 3 (中低频) 幅度", 0, 60, 25, step=5)
amp4 = st.sidebar.slider("🟧 IMF 4 (中频) 幅度", 0, 50, 15, step=5)
amp5 = st.sidebar.slider("🟥 IMF 5 (中高频) 幅度", 0, 40, 10, step=2)
amp6 = st.sidebar.slider("🟪 IMF 6 (高频噪声) 幅度", 0, 30, 8, step=2)

freqs = [0.015, 0.05, 0.12, 0.25, 0.6, 1.5]

# 4. 后台数学计算
x = np.linspace(0, 100, 500)
# 🌟 修改点 1：把基准流量从 150 提高到 250，留出更多向下波动的空间
imf1 = amp1 * np.sin(2 * np.pi * freqs[0] * x) + 250  
imf2 = amp2 * np.sin(2 * np.pi * freqs[1] * x)
imf3 = amp3 * np.sin(2 * np.pi * freqs[2] * x)
imf4 = amp4 * np.sin(2 * np.pi * freqs[3] * x)
imf5 = amp5 * np.sin(2 * np.pi * freqs[4] * x)
imf6 = amp6 * np.sin(2 * np.pi * freqs[5] * x) + np.random.normal(0, amp6/2, len(x))

# 混合叠加
raw_mixed_signal = imf1 + imf2 + imf3 + imf4 + imf5 + imf6
# 🌟 修改点 2：加入物理极限截断，交通流不可能小于 0
mixed_signal = np.maximum(0, raw_mixed_signal)


# ====== 下面只需稍微修改图表的 Y 轴显示范围 ======
# 5. 绘图与可视化 (保持之前的排版设定)
plt.style.use('seaborn-v0_8-whitegrid')
# 🌟 强制加载你刚刚上传到 GitHub 的字体文件 🌟
# 注意：这里的 'simhei.ttf' 必须和你上传的文件名一模一样（区分大小写）
font_path = "simhei.ttf" 
fm.fontManager.addfont(font_path)

# 将全局字体设置为你刚刚加载的黑体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 10))
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.5, 1, 1, 1], hspace=0.45, wspace=0.15)
colors = ['purple', '#1f77b4', '#2ca02c', '#bcbd22', '#ff7f0e', '#d62728', '#9467bd']

# --- 顶部横图 ---
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(x, mixed_signal, color=colors[0], linewidth=2.5, label="Original Mixed Traffic Flow")
ax0.set_title("0. 原始交通流混合信号", fontsize=14, fontweight='bold')
ax0.legend(loc="upper right")
# 🌟 修改点 3：总图的 Y 轴最高拉到 500，适应抬高的基准线
ax0.set_ylim(0, 500)
ax0.grid(True, alpha=0.4)
ax0.set_ylabel("Traffic Vol", fontsize=10)

# --- 下方网格图 ---
imfs = [imf1, imf2, imf3, imf4, imf5, imf6]
titles = [
    "1. IMF 1：低频主干趋势", "2. IMF 2：次低频波动",
    "3. IMF 3：中低频波动",   "4. IMF 4：中频波动",
    "5. IMF 5：中高频波动",   "6. IMF 6：高频随机干扰"
]
positions = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

for i in range(6):
    row, col = positions[i]
    ax = fig.add_subplot(gs[row, col], sharex=ax0)
    ax.plot(x, imfs[i], color=colors[i+1], linewidth=1.5, label=f"IMF {i+1}")
    ax.set_title(titles[i], fontsize=12, fontweight='bold')
    ax.legend(loc="upper right")
    
    # 🌟 修改点 4：适应抬高的 IMF1 基准线
    if i == 0:
        ax.set_ylim(0, 400) 
    else:
        ax.set_ylim(-100, 100)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        
    ax.grid(True, alpha=0.4)
    if col == 0: ax.set_ylabel("Amplitude", fontsize=10)
    if row < 3: plt.setp(ax.get_xticklabels(), visible=False)
    else: ax.set_xlabel("Time Steps", fontsize=11)

# 6. 渲染图表
st.pyplot(fig)


st.success("💡 **排版优势：** 现在所有的图表都集成在了一个紧凑的 14x10 画布中。您可以直接在网页上右键点击这张大图 ->【图片另存为】，把它完美地插入到您的毕业论文里！")
