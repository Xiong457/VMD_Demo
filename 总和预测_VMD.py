import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import warnings
import os
from joblib import Parallel, delayed

# --- 尝试引入 VMD 分解库 ---
try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 vmdpy 库！请执行 `pip install vmdpy` 以启用 VMD-LSTM 模型。")
    VMD_AVAILABLE = False

# 限制单个 TF 任务的线程数，为多进程并行腾出 CPU 核心
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 忽略警告
warnings.filterwarnings("ignore")
# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 0. VMD-LSTM 全局配置
# ==========================================
VMD_K = 8           # 分解的模态数量 (IMF个数)
VMD_ALPHA = 2000    # 惩罚因子
N_CORES = -1        # 启用所有 CPU 核心并行训练

# 非对称惩罚 MAE 权重
PENALTY_UNDER = 1.5 # 估计低了 (Under-predict) 给 1.5 倍权重
PENALTY_OVER = 1.0  # 估计高了 (Over-predict) 给 1.0 倍权重

# ==========================================
# 1. 数据加载与预处理
# ==========================================
year = 2017
file_path = f'地面交叉口5分钟流量信息_{year}.xlsx' 

print(f"正在读取文件: {file_path} ...")

try:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}。请确认文件名和路径是否正确。")
    exit()

dfs = []
for sheet_name, df in all_sheets.items():
    if '线圈号' in df.columns:
        df = df[df['线圈号'] != 'LoopId']
        dfs.append(df)
    else:
        print(f"跳过工作簿 {sheet_name}: 未找到'线圈号'列")

if not dfs:
    print("未找到有效数据，请检查Excel表头是否正确。")
    exit()

full_df = pd.concat(dfs, ignore_index=True)

cols_to_numeric = ['小客车流量', '大客车流量', '小货车流量', '大货车流量', '拖挂车流量', '摩托车流量', '采集时间']
for col in cols_to_numeric:
    full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

full_df = full_df.fillna(0)

full_df['Total_Flow'] = (
    full_df['小客车流量'] * 1.0 +
    full_df['大客车流量'] * 1.5 +
    full_df['小货车流量'] * 1.0 +
    full_df['大货车流量'] * 2.0 +
    full_df['拖挂车流量'] * 3.0 +
    full_df['摩托车流量'] * 1.0
)

full_df['Date'] = pd.to_datetime(full_df['采集日期'])
full_df['Datetime'] = full_df['Date'] + pd.to_timedelta((full_df['采集时间'] - 1) * 5, unit='m')

agg_df = full_df.groupby('Datetime')['Total_Flow'].sum().reset_index()
agg_df = agg_df.sort_values('Datetime')

full_idx = pd.date_range(start=agg_df['Datetime'].min(), end=agg_df['Datetime'].max(), freq='5T')
agg_df = agg_df.set_index('Datetime').reindex(full_idx, fill_value=0).reset_index()
agg_df.columns = ['Datetime', 'Flow']

print("数据处理完成，开始模型训练...")

# ==========================================
# 2. 模型准备
# ==========================================
data = agg_df.set_index('Datetime')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Flow']])

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

look_back = 12

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, look_back)
X_full, y_full = create_dataset(scaled_data, look_back)

# ==========================================
# 辅助函数：VMD 模态并行训练
# ==========================================
def train_imf_model(k, imf_data, train_sz, lb):
    imf_train = imf_data[:train_sz].reshape(-1, 1)
    imf_full = imf_data.reshape(-1, 1)
    
    X_tr_m, y_tr_m = create_dataset(imf_train, lb)
    X_fl_m, _ = create_dataset(imf_full, lb)
    
    X_tr_m = np.reshape(X_tr_m, (X_tr_m.shape[0], X_tr_m.shape[1], 1))
    X_fl_m = np.reshape(X_fl_m, (X_fl_m.shape[0], X_fl_m.shape[1], 1))
    
    model_m = Sequential([
        Input(shape=(lb, 1)),
        LSTM(50),
        Dense(1)
    ])
    model_m.compile(loss='mean_squared_error', optimizer='adam')
    model_m.fit(X_tr_m, y_tr_m, epochs=20, batch_size=32, verbose=0)
    
    pred_full = model_m.predict(X_fl_m, verbose=0)
    return k, pred_full

# ==========================================
# 3. 训练与预测
# ==========================================
print("   👉 正在训练 LSTM...")
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] , 1))
X_full_lstm = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

tf.random.set_seed(42)
np.random.seed(42)

model_lstm = Sequential([
    Input(shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
lstm_pred = model_lstm.predict(X_full_lstm, verbose=0)

print("   👉 正在训练 SVR...")
model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
model_svr.fit(X_train, y_train)
svr_pred = model_svr.predict(X_full).reshape(-1, 1)

print("   👉 正在训练 ARIMA...")
model_arima_train = ARIMA(train_data.flatten(), order=(5,1,0))
res_train = model_arima_train.fit()
model_arima_full = ARIMA(scaled_data.flatten(), order=(5,1,0))
res_full = model_arima_full.filter(res_train.params)
arima_pred_full_aligned = res_full.predict(start=0, end=len(scaled_data)-1)

print("   👉 正在训练 🌟 VMD-LSTM...")
vmd_lstm_pred = np.zeros_like(lstm_pred)
u = None
imf_preds_dict = {}  

if VMD_AVAILABLE:
    print(f"      🌊 正在对全局序列进行 VMD 分解 (模态数 K={VMD_K}) ...")
    u, _, _ = VMD(scaled_data.flatten(), alpha=VMD_ALPHA, tau=0, K=VMD_K, DC=0, init=1, tol=1e-7)
    
    print(f"      🚀 启动多核并行训练 (并行度: CPU全核心)...")
    imf_results = Parallel(n_jobs=N_CORES)(
        delayed(train_imf_model)(k, u[k, :], train_size, look_back) for k in range(VMD_K)
    )
    
    for k, pred_full in imf_results:
        imf_preds_dict[k] = pred_full
        
    imf_preds_list = [imf_preds_dict[k] for k in range(VMD_K)]
    vmd_lstm_pred = np.sum(imf_preds_list, axis=0)

# ==========================================
# 4. 数据还原与对齐
# ==========================================
lstm_pred_inv = scaler.inverse_transform(lstm_pred)
svr_pred_inv = scaler.inverse_transform(svr_pred)
vmd_lstm_pred_inv = scaler.inverse_transform(vmd_lstm_pred)

arima_pred_sliced = arima_pred_full_aligned[look_back:].reshape(-1, 1)
arima_pred_inv = scaler.inverse_transform(arima_pred_sliced)

actual_inv = scaler.inverse_transform(scaled_data[look_back:])

valid_indices = data.index[look_back:]
result_df = pd.DataFrame(index=valid_indices)
result_df['Actual'] = actual_inv.flatten()
result_df['LSTM'] = np.maximum(0, lstm_pred_inv.flatten())
result_df['SVR'] = np.maximum(0, svr_pred_inv.flatten())
result_df['ARIMA'] = np.maximum(0, arima_pred_inv.flatten())
result_df['VMD-LSTM'] = np.maximum(0, vmd_lstm_pred_inv.flatten())

# ==========================================
# 5. 绘图 (时间序列对比)
# ==========================================
unique_dates = pd.to_datetime(result_df.index.date).unique()
print(f"共生成 {len(unique_dates)} 天的预测结果。")

for date in unique_dates:
    day_str = date.strftime('%Y-%m-%d')
    day_data = result_df[result_df.index.normalize() == date]
    
    if day_data.empty: continue

    plt.figure(figsize=(14, 6))
    plt.plot(day_data.index, day_data['Actual'], label='Actual', color='black', alpha=0.7, linewidth=2)
    
    plt.plot(day_data.index, day_data['VMD-LSTM'], label='VMD-LSTM', color='orange', linewidth=2)
    plt.plot(day_data.index, day_data['LSTM'], label='LSTM', color='blue', linestyle='--', alpha=0.7)
    plt.plot(day_data.index, day_data['SVR'], label='SVR', color='green', linestyle='-.', alpha=0.7)
    plt.plot(day_data.index, day_data['ARIMA'], label='ARIMA', color='red', linestyle=':', linewidth=2, alpha=0.7)
    
    plt.title(f'Traffic Flow Prediction (Includes VMD-LSTM) - {day_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f'上海总和{year}_{day_str}.png'
    plt.savefig(filename, dpi=200)
    plt.close()

# ==========================================
# 6. 性能指标评估
# ==========================================
print("\n" + "="*40)
print("正在计算测试集评估指标...")
print("="*40)

test_start_index = train_size
test_start_date = data.index[test_start_index]
test_df = result_df[result_df.index >= test_start_date].copy()

y_train_actual = data['Flow'].values[:train_size]

def calculate_metrics(y_true, y_pred, y_train_act, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    
    errors = y_true - y_pred
    penalties = np.where(errors > 0, errors * PENALTY_UNDER, -errors * PENALTY_OVER)
    ap_mae = np.mean(penalties)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    naive_mae = mean_absolute_error(y_train_act[1:], y_train_act[:-1])
    mase = mae / naive_mae if naive_mae != 0 else np.nan
    
    r2 = r2_score(y_true, y_pred)
    
    return [model_name, mae, ap_mae, mape, mase, r2]

metrics_data = []
models = ['LSTM', 'SVR', 'ARIMA', 'VMD-LSTM']
colors_map = {'LSTM': 'blue', 'SVR': 'green', 'ARIMA': 'red', 'VMD-LSTM': 'orange'}

for model in models:
    metrics = calculate_metrics(test_df['Actual'], test_df[model], y_train_actual, model)
    metrics_data.append(metrics)

metrics_df = pd.DataFrame(metrics_data, columns=['模型', 'MAE', 'AP-MAE', 'MAPE(%)', 'MASE', 'R2'])
print("\n测试集性能评估结果:")

display_df = metrics_df.copy()
display_df[['MAE', 'AP-MAE', 'MAPE(%)']] = display_df[['MAE', 'AP-MAE', 'MAPE(%)']].round(2)
display_df[['MASE', 'R2']] = display_df[['MASE', 'R2']].round(4)
print(display_df.to_string(index=False))

# ==========================================
# 7. 生成可视化评估图表
# ==========================================
print("\n正在生成可视化评估图表...")

if VMD_AVAILABLE and u is not None:
    print("📈 正在生成 VMD 模态 (IMFs) 对比图...")
    
    # 🌟 图 0-A: 垂直分离对齐版 (包含分解前的原始总流量在最顶端)
    # 高度设置为 2.5 * (VMD_K + 1)
    fig, axes = plt.subplots(VMD_K + 1, 1, figsize=(15, 2.5 * (VMD_K + 1)), sharex=True)

    # ---> 第一个子图：分解前的原始整体信号 <---
    ax_orig = axes[0]
    actual_scaled_test = scaled_data[train_size:].flatten()
    pred_scaled_test = vmd_lstm_pred[train_size - look_back:].flatten()
    
    ax_orig.plot(actual_scaled_test, label='Actual Original Flow', color='black', linewidth=2)
    ax_orig.plot(pred_scaled_test, label='Predicted Original Flow (VMD-LSTM)', color='red', linestyle='--', linewidth=2)
    ax_orig.set_title('Original Total Signal (Before Decomposition)', fontsize=13, fontweight='bold')
    ax_orig.legend(loc='upper right')
    ax_orig.grid(True, alpha=0.3)
    ax_orig.set_ylabel('Amplitude\n(Scaled)')

    # ---> 余下的子图：分解后的各阶 IMF <---
    for k in range(VMD_K):
        ax = axes[k + 1]
        actual_imf_test = u[k, train_size:]
        pred_imf_test = imf_preds_dict[k][train_size - look_back:].flatten()

        ax.plot(actual_imf_test, label=f'Actual IMF {k+1}', color='black', alpha=0.7, linewidth=1.5)
        ax.plot(pred_imf_test, label=f'Predicted IMF {k+1}', color='orange', linestyle='--', linewidth=2)
        
        freq_label = "Low Frequency/Trend" if k == 0 else "High Frequency/Noise" if k == VMD_K-1 else "Mid Frequency"
        ax.set_title(f'IMF {k+1} ({freq_label})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Amplitude')

    plt.xlabel('Time Steps (Test Set)', fontsize=12)
    plt.tight_layout()
    imf_filename = f'总和预测_VMD_IMFs_Prediction_{year}.png'
    plt.savefig(imf_filename, dpi=200)
    plt.close()

    # 🌟 图 0-B: 同一坐标轴叠放版 (原始曲线与所有IMF物理画在一起)
    plt.figure(figsize=(15, 6))
    # 原始曲线用粗的半透明黑线
    plt.plot(actual_scaled_test, label='Original Total Signal', color='black', linewidth=4, alpha=0.5)
    
    # 给 K 条不同的 IMF 分配不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, VMD_K))
    for k in range(VMD_K):
        plt.plot(u[k, train_size:], label=f'IMF {k+1}', color=colors[k], linewidth=1.5, alpha=0.8)

    plt.title(f'Original Signal and All IMFs Overlaid (Test Set) - {year}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Amplitude (Scaled)', fontsize=12)
    
    # 把图例放到图外防止遮挡
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    overlaid_filename = f'总和预测_VMD_IMFs_Overlaid_{year}.png'
    plt.savefig(overlaid_filename, dpi=200)
    plt.close()
    print(f"已保存: {imf_filename} (垂直分离版)")
    print(f"已保存: {overlaid_filename} (同坐标轴重叠版)")

# ------------------------------------------
# 图表 1: 散点回归图 (2x2 布局)
# ------------------------------------------
plt.figure(figsize=(15, 12))
for i, model in enumerate(models):
    plt.subplot(2, 2, i+1)
    plt.scatter(test_df['Actual'], test_df[model], alpha=0.4, color=colors_map[model], label='Data Points')
    
    min_val = min(test_df['Actual'].min(), test_df[model].min())
    max_val = max(test_df['Actual'].max(), test_df[model].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    z = np.polyfit(test_df['Actual'], test_df[model], 1)
    p = np.poly1d(z)
    x_vals = np.sort(test_df['Actual'])
    plt.plot(x_vals, p(x_vals), color='black', linestyle='-', linewidth=1.5, label=f'Fit (y={z[0]:.2f}x+{z[1]:.2f})')
    
    r2_val = r2_score(test_df['Actual'], test_df[model])
    plt.title(f'{model} Regression (R²={r2_val:.4f})', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Flow')
    plt.ylabel('Predicted Flow')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

plt.suptitle(f'Regression Scatter Plot Comparison {year}', fontsize=16)
plt.tight_layout()
plt.savefig(f'evaluation_scatter_plot{year}.png', dpi=200)

# ------------------------------------------
# 图表 2: 残差分布图 (2x2 布局)
# ------------------------------------------
plt.figure(figsize=(15, 12))
for i, model in enumerate(models):
    plt.subplot(2, 2, i+1)
    residuals = test_df['Actual'] - test_df[model]
    sns.histplot(residuals, kde=True, color=colors_map[model], bins=30, alpha=0.6)
    
    plt.title(f'{model} Residual Distribution\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}', fontsize=12, fontweight='bold')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='k', linestyle='--', linewidth=1.5)
    plt.grid(True, alpha=0.3)

plt.suptitle(f'Residual Histogram Comparison {year}', fontsize=16)
plt.tight_layout()
plt.savefig(f'evaluation_residual_plot{year}.png', dpi=200)

# ------------------------------------------
# 🌟 图表 3: 误差指标分离式柱状图 (2x2 布局)
# ------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

plot_metrics = ['MAE', 'AP-MAE', 'MAPE(%)', 'MASE']
metric_titles = [
    'Standard MAE (Lower is Better)',
    f'Asymmetric Penalized MAE (Under: {PENALTY_UNDER}x, Over: {PENALTY_OVER}x)',
    'MAPE (%) (Lower is Better)',
    'MASE (Scale-Free, <1 is good)'
]

for idx, (metric, title) in enumerate(zip(plot_metrics, metric_titles)):
    ax = axes[idx]
    
    sns.barplot(x='模型', y=metric, data=metrics_df, ax=ax, palette=[colors_map[m] for m in metrics_df['模型']])
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
        
    if metric == 'MASE':
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline (Naive = 1.0)')
        ax.legend(loc='upper right')

plt.suptitle(f'Model Error Metrics Comparison (Separated) - {year}', fontsize=16, fontweight='bold')
plt.tight_layout()
output_file = f'evaluation_metrics_bar{year}.png'
plt.savefig(output_file, dpi=200)
print(f"已保存: {output_file} (四合一独立柱状图)")

print("\n🎉 程序运行结束！")