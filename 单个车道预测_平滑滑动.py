import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 屏蔽 TensorFlow 底层 Info 提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 2. 尝试引入 Intel 专属的 Scikit-learn 加速补丁
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# 3. 导入 TensorFlow 并配置纯 CPU 环境
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 TensorFlow，将跳过 LSTM 训练。")
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid') # 设置绘图风格

# ================= 配置区 =================
year = 2018
EXCEL_FILENAME = f"地面交叉口5分钟流量信息_{year}.xlsx"

# 🎯 核心控制：你要预测的车道ID
TARGET_LOOP = 10

# 🌊 方法二核心：滑动平滑窗口大小 (把前后 SMOOTH_WINDOW 个时间点做平均)
SMOOTH_WINDOW = 3 

LOOK_BACK = 12            # 提取过去 2 小时特征
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
# =======================================

def load_data(filename):
    print(f"📂 1. 正在加载数据: {filename} ...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"未找到文件: {filename}")

    sheets = pd.read_excel(filename, sheet_name=None, header=1)
    dfs = [df for df in sheets.values() if 'LoopId' in df.columns]
    if not dfs: raise ValueError("无有效数据")
    full_df = pd.concat(dfs, ignore_index=True)

    # 预处理与 PCU 计算
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

def train_predict(train_data, test_data, look_back):
    print(f"⚙️ 2. 正在训练模型 (SVR, ARIMA, LSTM) ...")
    
    # --- 1. SVR ---
    scaler = MinMaxScaler((0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    X_train, y_train = create_dataset(train_scaled, look_back)
    
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr.fit(X_train, y_train)
    
    full_data = np.concatenate([train_data, test_data])
    full_scaled = scaler.transform(full_data.reshape(-1, 1))
    
    X_test_svr = []
    start_idx = len(train_data)
    for i in range(len(test_data)):
        X_test_svr.append(full_scaled[start_idx + i - look_back : start_idx + i].flatten())
        
    svr_pred = scaler.inverse_transform(svr.predict(X_test_svr).reshape(-1,1)).flatten()
    
    # --- 2. ARIMA ---
    try:
        # 平滑后采用较低阶数防止过拟合
        model_arima = ARIMA(train_data, order=(3, 1, 0)).fit()
        model_full = ARIMA(full_data, order=(3, 1, 0))
        res_full = model_full.filter(model_arima.params)
        arima_pred = res_full.fittedvalues[-len(test_data):]
    except:
        arima_pred = np.zeros(len(test_data))
        
    # --- 3. LSTM ---
    lstm_pred = np.zeros(len(test_data))
    if TF_AVAILABLE:
        last_train = train_data[-look_back:]
        extended_test = np.concatenate([last_train, test_data])
        test_scaled_ext = scaler.transform(extended_test.reshape(-1, 1))
        
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm, _ = create_dataset(test_scaled_ext, look_back)
        X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
        
        model = Sequential([
            Input(shape=(look_back, 1)),
            LSTM(32),
            Dense(1)
        ])
        model.compile(loss='mae', optimizer='adam')
        model.fit(X_train_lstm, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0, shuffle=False)
        lstm_pred = scaler.inverse_transform(model.predict(X_test_lstm, verbose=0)).flatten()
        
    return np.maximum(0, svr_pred), np.maximum(0, arima_pred), np.maximum(0, lstm_pred)

def plot_advanced_visualizations(df, output_prefix=f"Loop{TARGET_LOOP}"):
    """生成高级可视化图表：散点回归、残差分布、指标对比"""
    print(f"📊 3. 正在生成高级分析图表...")
    
    models = ['SVR', 'ARIMA', 'LSTM']
    colors = {'SVR': 'green', 'ARIMA': 'blue', 'LSTM': 'red'}
    
    # --- 图表 1: 散点回归图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    max_val = max(df['Actual'].max(), df[models].max().max())
    
    for i, model in enumerate(models):
        ax = axes[i]
        ax.scatter(df['Actual'], df[model], alpha=0.5, color=colors[model], label='Data Points')
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Fit (y=x)')
        
        m, b = np.polyfit(df['Actual'], df[model], 1)
        ax.plot(df['Actual'], m*df['Actual'] + b, color=colors[model], linewidth=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
        
        ax.set_title(f'{model}: Predicted vs Actual')
        ax.set_xlabel('Actual Flow (Smoothed)')
        ax.set_ylabel('Predicted Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Scatter_Regression(smoothed).png")
    plt.close()
    
    # --- 图表 2: 残差分布图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model in enumerate(models):
        residuals = df['Actual'] - df[model]
        ax = axes[i]
        sns.histplot(residuals, kde=True, ax=ax, color=colors[model], bins=30)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(f'{model} Residuals Distribution')
        ax.set_xlabel('Residual (Actual - Predicted)')
        
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Residual_Histogram(smoothed).png")
    plt.close()
    
    # --- 图表 3: 误差指标对比柱状图 ---
    metrics_data = []
    for model in models:
        y_true = df['Actual']
        y_pred = df[model]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true[y_true!=0] - y_pred[y_true!=0]) / y_true[y_true!=0])) * 100
        r2 = r2_score(y_true, y_pred)
        metrics_data.append([model, mae, rmse, mape, r2])
        
    metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'MAE', 'RMSE', 'MAPE', 'R2'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE
    sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[0,0], palette=[colors[m] for m in metrics_df['Model']])
    axes[0,0].set_title('MAE (Lower is Better)')
    for i, v in enumerate(metrics_df['MAE']): axes[0,0].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    # RMSE
    sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[0,1], palette=[colors[m] for m in metrics_df['Model']])
    axes[0,1].set_title('RMSE (Lower is Better)')
    for i, v in enumerate(metrics_df['RMSE']): axes[0,1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
    # MAPE
    sns.barplot(x='Model', y='MAPE', data=metrics_df, ax=axes[1,0], palette=[colors[m] for m in metrics_df['Model']])
    axes[1,0].set_title('MAPE % (Lower is Better)')
    for i, v in enumerate(metrics_df['MAPE']): axes[1,0].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
    # R2
    sns.barplot(x='Model', y='R2', data=metrics_df, ax=axes[1,1], palette=[colors[m] for m in metrics_df['Model']])
    axes[1,1].set_title('R2 Score (Higher is Better)')
    axes[1,1].set_ylim(0, 1)
    for i, v in enumerate(metrics_df['R2']): axes[1,1].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
    plt.suptitle(f"Model Metrics Comparison (Loop {TARGET_LOOP}, Smoothed)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_Metrics_Comparison(smoothed).png")
    plt.close()
    
    print(f"✅ 图表已生成:\n  - {output_prefix}_Scatter_Regression.png\n  - {output_prefix}_Residual_Histogram.png\n  - {output_prefix}_Metrics_Comparison.png")

def main():
    start_time = time.time()
    
    # 1. 数据加载与平滑处理
    full_df = load_data(EXCEL_FILENAME)
    loop_df = full_df[full_df['LoopId'] == TARGET_LOOP].sort_values(['Date', 'TimeId']).copy()
    
    if len(loop_df) == 0:
        print(f"❌ 错误: 找不到车道 {TARGET_LOOP} 的数据，请检查 TARGET_LOOP 设置。")
        return

    # 🌊 方法二核心逻辑：滑动平均消除毛刺 🌊
    loop_df['TotalFlow'] = loop_df['TotalFlow'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    
    # 2. 数据集划分
    dates = sorted(loop_df['Date'].unique())
    split_idx = 8 if len(dates) >= 10 else len(dates) - 2
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_data = loop_df[loop_df['Date'].isin(train_dates)]['TotalFlow'].values
    test_df = loop_df[loop_df['Date'].isin(test_dates)].copy()
    test_data = test_df['TotalFlow'].values
    
    # 3. 训练与预测
    svr_p, arima_p, lstm_p = train_predict(train_data, test_data, LOOK_BACK)
    
    test_df['SVR'] = svr_p
    test_df['ARIMA'] = arima_p
    test_df['LSTM'] = lstm_p
    test_df.rename(columns={'TotalFlow': 'Actual'}, inplace=True)
    
    # 4. 基础时间序列折线图 (每天一张)
    for date in test_dates:
        day_data = test_df[test_df['Date'] == date]
        plt.figure(figsize=(14, 6))
        x = day_data['TimeId']
        # 注意：这里的 Actual 已经是平滑后的数据
        plt.plot(x, day_data['Actual'], 'k-', label='Actual (Smoothed)', alpha=0.7)
        plt.plot(x, day_data['LSTM'], 'r-', label='LSTM')
        plt.plot(x, day_data['SVR'], 'g--', label='SVR')
        plt.plot(x, day_data['ARIMA'], 'b:', label='ARIMA')
        
        plt.title(f"Traffic Flow Comparison (Smoothed) - Loop {TARGET_LOOP} - {date}")
        plt.xlabel("TimeId"); plt.ylabel("Flow (PCU)"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f"Loop{TARGET_LOOP}_TimeSeries_{date}(smoothed).png")
        plt.close()
        
    # 5. 高级分析图表
    plot_advanced_visualizations(test_df, output_prefix=f"Loop{TARGET_LOOP}")
    
    print("\n" + "="*50)
    print(f"🎉 任务完成！总耗时: {time.time() - start_time:.2f} 秒")
    print(f"👉 重点关注生成的图表: Loop{TARGET_LOOP}_Metrics_Comparison.png")
    print("="*50)

if __name__ == "__main__":
    main()