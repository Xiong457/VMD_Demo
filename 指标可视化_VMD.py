import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置 =================
year = 2018
# 读取包含 VMD-LSTM 的新数据文件
INPUT_FILE = f'all_lanes_model_comparison_{year}_smoothed_vmd.csv'
# =======================================

# 读取数据
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"找不到文件 {INPUT_FILE}，请确认上一步的模型训练脚本已成功运行并生成了该文件。")
    exit()

# 按Loop排序，确保顺序正确
df = df.sort_values('Loop')

# 创建图形 1: 各车道多模型对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Model Performance Comparison Across All Loops {year} (Incl. VMD-LSTM)', fontsize=16, fontweight='bold')

# 获取所有唯一的Loop值并按数字排序
loops = sorted(df['Loop'].unique())
loop_labels = [f'Loop {int(l)}' for l in loops]

# 动态获取模型列表，保证特定顺序，并加上 VMD-LSTM 的专属颜色
models_in_data = df['Model'].unique()
target_order = ['SVR', 'ARIMA', 'LSTM', 'VMD-LSTM']
models = [m for m in target_order if m in models_in_data] # 保证存在的模型才画

model_colors = {
    'SVR': '#1f77b4',      # 蓝色
    'ARIMA': '#ff7f0e',    # 橙色
    'LSTM': '#2ca02c',     # 绿色
    'VMD-LSTM': '#d62728'  # 红色 (新增)
}

# 动态计算柱状图宽度，避免4个模型时柱子重叠
n_models = len(models)
bar_width = 0.8 / n_models 
x = np.arange(len(loops))

# 辅助函数：快速获取数据并对齐
def get_metric_values(model_name, metric_name):
    model_data = df[df['Model'] == model_name]
    return [model_data[model_data['Loop'] == loop][metric_name].values[0] if loop in model_data['Loop'].values else 0 
            for loop in loops]

# X轴刻度居中偏移量
x_tick_offset = x + bar_width * (n_models - 1) / 2

# 1. MAE对比图
ax1 = axes[0, 0]
for i, model in enumerate(models):
    mae_values = get_metric_values(model, 'MAE')
    ax1.bar(x + i * bar_width, mae_values, bar_width, label=model, color=model_colors[model], alpha=0.8)

ax1.set_xlabel('Loop')
ax1.set_ylabel('MAE')
ax1.set_title('Mean Absolute Error (MAE) Comparison')
ax1.set_xticks(x_tick_offset)
ax1.set_xticklabels(loop_labels, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. RMSE对比图
ax2 = axes[0, 1]
for i, model in enumerate(models):
    rmse_values = get_metric_values(model, 'RMSE')
    ax2.bar(x + i * bar_width, rmse_values, bar_width, label=model, color=model_colors[model], alpha=0.8)

ax2.set_xlabel('Loop')
ax2.set_ylabel('RMSE')
ax2.set_title('Root Mean Square Error (RMSE) Comparison')
ax2.set_xticks(x_tick_offset)
ax2.set_xticklabels(loop_labels, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. MAPE对比图
ax3 = axes[1, 0]
for i, model in enumerate(models):
    mape_values = get_metric_values(model, 'MAPE')
    ax3.bar(x + i * bar_width, mape_values, bar_width, label=model, color=model_colors[model], alpha=0.8)

ax3.set_xlabel('Loop')
ax3.set_ylabel('MAPE (%)')
ax3.set_title('Mean Absolute Percentage Error (MAPE) Comparison')
ax3.set_xticks(x_tick_offset)
ax3.set_xticklabels(loop_labels, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. R²对比图
ax4 = axes[1, 1]
for i, model in enumerate(models):
    r2_values = get_metric_values(model, 'R2')
    ax4.bar(x + i * bar_width, r2_values, bar_width, label=model, color=model_colors[model], alpha=0.8)

ax4.set_xlabel('Loop')
ax4.set_ylabel('R²')
ax4.set_title('Coefficient of Determination (R²) Comparison')
ax4.set_xticks(x_tick_offset)
ax4.set_xticklabels(loop_labels, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图表 1
chart1_filename = f'model_comparison_chart{year}_smoothed_vmd.png'
plt.savefig(chart1_filename, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存: {chart1_filename}")
plt.show()

# =====================================================================
# 创建图形 2: 平均指标性能柱状图
# =====================================================================
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))  # 稍微加宽一点适应4个模型
fig2.suptitle(f'Average Performance Metrics by Model {year}', fontsize=14, fontweight='bold')

metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
metric_titles = ['Mean Absolute Error', 'Root Mean Square Error', 'Mean Absolute Percentage Error', 'Coefficient of Determination']

for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes2[idx]
    avg_values = []
    for model in models:
        avg_value = df[df['Model'] == model][metric].mean()
        avg_values.append(avg_value)
    
    bars = ax.bar(models, avg_values, color=[model_colors[m] for m in models], alpha=0.8)
    ax.set_title(f'Average {metric}')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        # 对于R2可以把文字稍微往下放一点防止顶破天际，其他指标放上面
        offset = 0.01 * height if height > 0 else 0.05
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# 保存图表 2
chart2_filename = f'average_model_performance{year}_smoothed_vmd.png'
plt.savefig(chart2_filename, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存: {chart2_filename}")
plt.show()

# =====================================================================
# 打印终端统计信息
# =====================================================================
print("\n" + "=" * 60)
print("📊 Average Performance by Model:")
print("=" * 60)
for model in models:
    model_data = df[df['Model'] == model]
    print(f"{model}:")
    print(f"  MAE:  {model_data['MAE'].mean():.4f}")
    print(f"  RMSE: {model_data['RMSE'].mean():.4f}")
    print(f"  MAPE: {model_data['MAPE'].mean():.4f}%")
    print(f"  R²:   {model_data['R2'].mean():.4f}\n")

print("=" * 60)
print("🏆 Best Model for Each Metric:")
print("=" * 60)
for metric in metrics:
    best_model = None
    best_value = None
    
    if metric == 'R2':  # 对于R²，值越大越好
        best_value = -np.inf
        for model in models:
            avg_value = df[df['Model'] == model][metric].mean()
            if avg_value > best_value:
                best_value = avg_value
                best_model = model
    else:  # 对于MAE、RMSE、MAPE，值越小越好
        best_value = np.inf
        for model in models:
            avg_value = df[df['Model'] == model][metric].mean()
            if avg_value < best_value:
                best_value = avg_value
                best_model = model
    
    print(f"  {metric}: {best_model} ({best_value:.4f})")
print("=" * 60)