import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# 屏蔽 TensorFlow 底层 Info 提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# --- 🌟 引入 VMD 分解库 ---
try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 vmdpy 库！请执行 `pip install vmdpy` 以启用 VMD-LSTM 模型。")
    VMD_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    tf.config.set_visible_devices([], 'GPU')
    # 限制单个 TF 任务的线程数，为多进程并行腾出 CPU 核心
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 TensorFlow，将跳过 LSTM 训练。")
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# ================= 配置区 =================
year = 2018
EXCEL_FILENAME = f"地面交叉口5分钟流量信息_{year}.xlsx"

TARGET_LOOP = 10
SMOOTH_WINDOW = 3 
LOOK_BACK = 12            
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32

VMD_K = 12         # 模态数量
VMD_ALPHA = 2000   # 惩罚因子

# 🌟 并行核心数（-1 表示使用所有 CPU 线程，即 13600K 的全核心）
N_CORES = -1      

# 非对称惩罚权重
PENALTY_UNDER = 1.5 
PENALTY_OVER = 1.0  
# =======================================

def load_data(filename):
    print(f"📂 1. 正在加载数据: {filename} ...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到文件: {filename}")

    sheets = pd.read_excel(filename, sheet_name=None, header=1)
    dfs = [df for df in sheets.values() if 'LoopId' in df.columns]
    if not dfs: raise ValueError("无有效数据")
    full_df = pd.concat(dfs, ignore_index=True)

    cols = ['Xk', 'Dk', 'Xh', 'Dh', 'Tg', 'Mt']
    for c in cols:
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce').fillna(0)
    
    full_df['TotalFlow'] = (
        full_df['Xk'] * 1.0 + full_df['Dk'] * 1.5 + 
        full_df['Xh'] * 1.0 + full_df['Dh'] * 2.0 + 
        full_df['Tg'] * 3.0 + full_df['Mt'] * 1.0
    )
    full_df['Date'] = pd.to_datetime(full_df['DateId']).dt.date
    return full_df

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def calculate_mase(y_true, y_pred, y_train):
    naive_mae = mean_absolute_error(y_train[1:], y_train[:-1])
    if naive_mae == 0: return np.nan
    return mean_absolute_error(y_true, y_pred) / naive_mae

def calculate_ap_mae(y_true, y_pred):
    errors = y_true - y_pred
    penalties = np.where(errors > 0, errors * PENALTY_UNDER, -errors * PENALTY_OVER)
    return np.mean(penalties)

def train_single_vmd_mode(k, mode_data, train_len, look_back, epochs, batch_size):
    """用于多进程并行的单独模态训练函数"""
    mode_train = mode_data[:train_len]
    mode_test = mode_data[train_len:]
    
    scaler_mode = MinMaxScaler((0, 1))
    mode_train_scaled = scaler_mode.fit_transform(mode_train.reshape(-1, 1))
    
    X_m_train, y_m_train = create_dataset(mode_train_scaled, look_back)
    X_m_train = X_m_train.reshape((X_m_train.shape[0], X_m_train.shape[1], 1))
    
    model_m = Sequential([Input(shape=(look_back, 1)), LSTM(32), Dense(1)])
    model_m.compile(loss='mae', optimizer='adam')
    model_m.fit(X_m_train, y_m_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    
    last_train_m = mode_train[-look_back:]
    extended_test_m = np.concatenate([last_train_m, mode_test])
    test_scaled_ext_m = scaler_mode.transform(extended_test_m.reshape(-1, 1))
    
    X_m_test, _ = create_dataset(test_scaled_ext_m, look_back)
    X_m_test = X_m_test.reshape((X_m_test.shape[0], X_m_test.shape[1], 1))
    
    mode_pred_scaled = model_m.predict(X_m_test, verbose=0)
    mode_pred = scaler_mode.inverse_transform(mode_pred_scaled).flatten()
    
    return k, mode_test, mode_pred

def train_predict(train_data, test_data, look_back):
    print(f"⚙️ 2. 正在训练模型 (SVR, ARIMA, LSTM, 🌟VMD-LSTM)...")
    
    scaler = MinMaxScaler((0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    X_train, y_train = create_dataset(train_scaled, look_back)
    full_data = np.concatenate([train_data, test_data])
    full_scaled = scaler.transform(full_data.reshape(-1, 1))
    train_len = len(train_data)
    
    # 1. SVR
    print("   👉 训练 SVR...")
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr.fit(X_train, y_train)
    X_test_svr = [full_scaled[train_len + i - look_back : train_len + i].flatten() for i in range(len(test_data))]
    svr_pred = scaler.inverse_transform(svr.predict(X_test_svr).reshape(-1,1)).flatten()
    
    # 2. ARIMA
    print("   👉 训练 ARIMA...")
    try:
        model_arima = ARIMA(train_data, order=(3, 1, 0)).fit()
        model_full = ARIMA(full_data, order=(3, 1, 0))
        res_full = model_full.filter(model_arima.params)
        arima_pred = res_full.fittedvalues[-len(test_data):]
    except:
        arima_pred = np.zeros(len(test_data))
        
    # 3. LSTM & 4. VMD-LSTM
    lstm_pred = np.zeros(len(test_data))
    vmd_lstm_pred = np.zeros(len(test_data))
    vmd_details = {'actual': np.zeros((VMD_K, len(test_data))), 'predicted': np.zeros((VMD_K, len(test_data)))}
    
    if TF_AVAILABLE:
        # A. 训练单一 LSTM
        print("   👉 训练 单一 LSTM...")
        last_train = train_data[-look_back:]
        extended_test = np.concatenate([last_train, test_data])
        test_scaled_ext = scaler.transform(extended_test.reshape(-1, 1))
        
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm, _ = create_dataset(test_scaled_ext, look_back)
        X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
        
        model_lstm = Sequential([Input(shape=(look_back, 1)), LSTM(32), Dense(1)])
        model_lstm.compile(loss='mae', optimizer='adam')
        model_lstm.fit(X_train_lstm, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0, shuffle=False)
        lstm_pred = scaler.inverse_transform(model_lstm.predict(X_test_lstm, verbose=0)).flatten()
        
        # B. 🌟 并行训练 VMD-LSTM 🌟
        if VMD_AVAILABLE:
            print(f"   🌊 正在进行 VMD 分解 (提取 {VMD_K} 个模态)...")
            u, _, _ = VMD(full_data, alpha=VMD_ALPHA, tau=0, K=VMD_K, DC=0, init=1, tol=1e-7)
            
            print(f"   🚀 启动多核并行训练 (并行度: 所有可用核心)... CPU 即将起飞！")
            parallel_results = Parallel(n_jobs=N_CORES)(
                delayed(train_single_vmd_mode)(k, u[k, :], train_len, look_back, LSTM_EPOCHS, LSTM_BATCH_SIZE)
                for k in range(VMD_K)
            )
            
            for k, mode_test, mode_pred in parallel_results:
                vmd_details['actual'][k, :] = mode_test
                vmd_details['predicted'][k, :] = mode_pred
                
            vmd_lstm_pred = np.sum(vmd_details['predicted'], axis=0)
            
    return np.maximum(0, svr_pred), np.maximum(0, arima_pred), np.maximum(0, lstm_pred), np.maximum(0, vmd_lstm_pred), vmd_details

# ================= 绘图函数区 =================

def plot_vmd_imfs(vmd_details, output_prefix):
    print(f"📈 正在生成 VMD 模态 (IMFs) 对比图...")
    actual = vmd_details['actual']
    predicted = vmd_details['predicted']
    K = actual.shape[0]
    
    fig, axes = plt.subplots(K, 1, figsize=(15, 2.5 * K), sharex=True)
    if K == 1: axes = [axes]
    
    for k in range(K):
        ax = axes[k]
        ax.plot(actual[k], label=f'Actual IMF {k+1}', color='black', alpha=0.7, linewidth=1.5)
        ax.plot(predicted[k], label=f'Predicted IMF {k+1}', color='orange', linestyle='--', linewidth=2)
        freq_label = "Low Frequency/Trend" if k == 0 else "High Frequency/Noise" if k == K-1 else "Mid Frequency"
        ax.set_title(f'IMF {k+1} ({freq_label})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Amplitude')

    plt.xlabel('Time Steps (Test Set)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_VMD_IMFs_Prediction.png", dpi=200)
    plt.close()

def plot_residual_histogram(df, models, output_prefix):
    print(f"📉 正在生成 残差直方图 (Residual Histogram)...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, model in enumerate(models):
        residuals = df['Actual'] - df[model]
        sns.histplot(residuals, kde=True, ax=axes[i], color=colors[i], bins=30, alpha=0.6)
        
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        
        axes[i].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
        axes[i].set_title(f"{model} Residuals\nMean: {mean_res:.2f}, Std: {std_res:.2f}")
        axes[i].set_xlabel("Prediction Error (Actual - Predicted)")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        
    plt.suptitle(f"Residual Distribution Comparison - Loop {TARGET_LOOP}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Residual_Histogram.png", dpi=200)
    plt.close()

def plot_scatter_regression(df, models, output_prefix):
    print(f"🎯 正在生成 散点回归图 (Scatter Regression)...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    min_val = min(df['Actual'].min(), df[models].min().min()) * 0.9
    max_val = max(df['Actual'].max(), df[models].max().max()) * 1.1
    
    for i, model in enumerate(models):
        r2 = r2_score(df['Actual'], df[model])
        sns.scatterplot(x=df['Actual'], y=df[model], ax=axes[i], color=colors[i], alpha=0.6, s=50)
        
        # 1. 绘制完美的拟合基准线 (y=x)
        axes[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit (y=x)')
        
        # 2. 🌟 绘制实际数据的线性回归拟合线 (y = ax + b)
        z = np.polyfit(df['Actual'], df[model], 1)
        p = np.poly1d(z)
        x_vals = np.sort(df['Actual'])
        axes[i].plot(x_vals, p(x_vals), color='black', linestyle='-', linewidth=2, label=f'Data Fit (y={z[0]:.2f}x+{z[1]:.2f})')
        
        axes[i].set_title(f"{model} Regression (R² = {r2:.4f})")
        axes[i].set_xlabel("Actual Traffic Flow")
        axes[i].set_ylabel("Predicted Traffic Flow")
        axes[i].set_xlim([min_val, max_val])
        axes[i].set_ylim([min_val, max_val])
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(f"Scatter Regression (Actual vs Predicted) - Loop {TARGET_LOOP}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Scatter_Regression.png", dpi=200)
    plt.close()

def plot_advanced_visualizations(df, train_data, output_prefix):
    print(f"📊 正在生成 综合评估指标柱状图 (Metrics Comparison)...")
    models = ['SVR', 'ARIMA', 'LSTM', 'VMD-LSTM']
    colors = {'SVR': '#1f77b4', 'ARIMA': '#ff7f0e', 'LSTM': '#2ca02c', 'VMD-LSTM': '#d62728'}
    
    metrics_data = []
    for model in models:
        y_true = df['Actual']
        y_pred = df[model]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mase = calculate_mase(y_true, y_pred, train_data)  
        ap_mae = calculate_ap_mae(y_true, y_pred)          
        metrics_data.append([model, mae, rmse, mase, ap_mae])
        
    metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'MAE', 'RMSE', 'MASE', 'AP-MAE'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[0,0], palette=[colors[m] for m in metrics_df['Model']])
    axes[0,0].set_title('Standard MAE (Lower is Better)')
    for i, v in enumerate(metrics_df['MAE']): axes[0,0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[0,1], palette=[colors[m] for m in metrics_df['Model']])
    axes[0,1].set_title('Standard RMSE (Lower is Better)')
    for i, v in enumerate(metrics_df['RMSE']): axes[0,1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    sns.barplot(x='Model', y='MASE', data=metrics_df, ax=axes[1,0], palette=[colors[m] for m in metrics_df['Model']])
    axes[1,0].set_title('MASE (Scale-Free, <1 is good)')
    axes[1,0].axhline(1.0, color='red', linestyle='--', label='Baseline (Naive)')
    axes[1,0].legend()
    for i, v in enumerate(metrics_df['MASE']): axes[1,0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    sns.barplot(x='Model', y='AP-MAE', data=metrics_df, ax=axes[1,1], palette=[colors[m] for m in metrics_df['Model']])
    axes[1,1].set_title(f'Asymmetric Penalized MAE (Under-predict Penalty {PENALTY_UNDER}x)')
    for i, v in enumerate(metrics_df['AP-MAE']): axes[1,1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    plt.suptitle(f"Model Metrics Comparison (VMD-LSTM, K={VMD_K}) - Loop {TARGET_LOOP}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Metrics_Comparison_VMD.png",dpi=200)
    plt.close()

def main():
    start_time = time.time()
    
    full_df = load_data(EXCEL_FILENAME)
    loop_df = full_df[full_df['LoopId'] == TARGET_LOOP].sort_values(['Date', 'TimeId']).copy()
    
    if len(loop_df) == 0:
        print(f"❌ 错误: 找不到车道数据。")
        return

    # 滑动平滑
    loop_df['TotalFlow'] = loop_df['TotalFlow'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    
    dates = sorted(loop_df['Date'].unique())
    split_idx = 8 if len(dates) >= 10 else len(dates) - 2
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_data = loop_df[loop_df['Date'].isin(train_dates)]['TotalFlow'].values
    test_df = loop_df[loop_df['Date'].isin(test_dates)].copy()
    test_data = test_df['TotalFlow'].values
    
    # 获取预测结果（并行为13600K全力加速）
    svr_p, arima_p, lstm_p, vmd_lstm_p, vmd_details = train_predict(train_data, test_data, LOOK_BACK)
    
    test_df['SVR'] = svr_p
    test_df['ARIMA'] = arima_p
    test_df['LSTM'] = lstm_p
    test_df['VMD-LSTM'] = vmd_lstm_p
    test_df.rename(columns={'TotalFlow': 'Actual'}, inplace=True)
    
    output_prefix = f"Loop{TARGET_LOOP}"
    
    # 🌟 1. 绘制 VMD 的 IMF 分解图
    if VMD_AVAILABLE:
        plot_vmd_imfs(vmd_details, output_prefix)
        
    # 🌟 2. 绘制时序图 (TimeSeries)
    print(f"📈 正在生成 每日时序预测对比图 (TimeSeries)...")
    for date in test_dates:
        day_data = test_df[test_df['Date'] == date]
        plt.figure(figsize=(14, 6))
        x = day_data['TimeId']
        plt.plot(x, day_data['Actual'], 'k-', label='Actual (Smoothed)', alpha=0.7, linewidth=2)
        plt.plot(x, day_data['VMD-LSTM'], 'orange', label='🌟VMD-LSTM', linewidth=2)
        plt.plot(x, day_data['LSTM'], 'r-', label='LSTM', alpha=0.5)
        plt.plot(x, day_data['ARIMA'], 'b:', label='ARIMA', alpha=0.5)
        
        # 👇 补充的 SVR 曲线 👇
        plt.plot(x, day_data['SVR'], 'g--', label='SVR', alpha=0.5)
        
        plt.title(f"Traffic Flow Comparison (VMD-LSTM) - Loop {TARGET_LOOP} - {date}")
        plt.xlabel("TimeId"); plt.ylabel("Flow"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_prefix}_TimeSeries_{date}.png",dpi=200)
        plt.close()

    # 🌟 3. 绘制残差直方图
    plot_residual_histogram(test_df, ['SVR', 'ARIMA', 'LSTM', 'VMD-LSTM'], output_prefix)
    
    # 🌟 4. 绘制散点回归图 (带实际数据拟合直线)
    plot_scatter_regression(test_df, ['SVR', 'ARIMA', 'LSTM', 'VMD-LSTM'], output_prefix)
    
    # 🌟 5. 绘制高级评估图表
    plot_advanced_visualizations(test_df, train_data, output_prefix)
    
    print("\n" + "="*60)
    print(f"🎉 计算与绘图全部完成！总耗时: {time.time() - start_time:.2f} 秒")
    print("="*60)

if __name__ == "__main__":
    main()