import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# 文件路径和日期设置
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"

# 假设牛市时间段，用户需根据实际情况设定
start_date = pd.to_datetime("2023-09-10")
end_date = pd.to_datetime("2024-03-09")

# 1. 读取数据并解析日期列
btc_data = pd.read_csv(data_path)
btc_data['Start'] = pd.to_datetime(btc_data['Start'], dayfirst=True)

# 2. 筛选牛市时期数据
btc_data = btc_data[(btc_data['Start'] >= start_date) & (btc_data['Start'] <= end_date)].copy()

# 将日期设为索引并设置频率（假设数据为每日频率）
btc_data.set_index('Start', inplace=True)
btc_data = btc_data.asfreq('D')
btc_data.fillna(method='ffill', inplace=True)

# 3. 定义变量
btc_data['log_volume'] = np.log(btc_data['Volume'])
btc_data['return'] = np.log(btc_data['Close']) - np.log(btc_data['Open'])
btc_data['volatility'] = btc_data['High'] - btc_data['Low']

# 如果需要平稳化，可在此对变量进行差分或其它变换
# 下例假设数据已平稳，可直接使用。如需差分请参考下方注释:
# btc_data['diff_log_volume'] = btc_data['log_volume'].diff()
# btc_data['diff_return'] = btc_data['return'].diff()
# btc_data['diff_volatility'] = btc_data['volatility'].diff()
# btc_data.dropna(inplace=True)
# vars_for_var = ['diff_log_volume', 'diff_return', 'diff_volatility']

# 若无需差分，则直接使用原定义的三个指标进入VAR
vars_for_var = ['log_volume', 'return', 'volatility']
var_data = btc_data[vars_for_var].dropna()

# 4. 使用VAR模型选择最佳滞后阶数
max_lag = 10
model = VAR(var_data)
bic_values = []

for p in range(1, max_lag+1):
    result = model.fit(p)
    bic_values.append((p, result.bic))

# 找到BIC最小值对应的滞后阶数
best_lag_bic = min(bic_values, key=lambda x: x[1])[0]

print("Lag order and BIC values:")
for p, bic in bic_values:
    print(f"Lag: {p}, BIC: {bic:.4f}")

print(f"\nBest lag based on BIC: {best_lag_bic}")

# 5. 使用最佳滞后阶数拟合最终VAR模型
best_model = model.fit(best_lag_bic)

# 输出最终模型的参数矩阵
print("\nFinal Model Parameters:")
print(best_model.params)

# 若需要，可对最终模型进行进一步的诊断，如残差检验、稳定性检验等
if best_model.is_stable():
    print("\nThe VAR model is stable.")
else:
    print("\nThe VAR model is not stable.")
