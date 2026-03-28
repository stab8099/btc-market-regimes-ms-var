import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# 加载数据
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"
btc_data = pd.read_csv(data_path)

# 将 'Start' 列转换为日期格式，并筛选出2017年1月1日之后的数据
btc_data['Start'] = pd.to_datetime(btc_data['Start'])
btc_data = btc_data[btc_data['Start'] >= '2017-01-01'].copy()

# 计算 log_volume、log_return 和 volatility
btc_data['log_volume'] = np.log(btc_data['Volume'])
btc_data['log_return'] = np.log(btc_data['Close']).diff()
btc_data['volatility'] = btc_data['log_return'] ** 2

# 删除包含NA值的行
btc_data.dropna(inplace=True)

# 检查变量的平稳性
def check_stationarity(series, series_name):
    result = adfuller(series)
    print(f'\nADF Statistic for {series_name}: {result[0]:.4f}')
    print(f'p-value for {series_name}: {result[1]:.4f}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value:.4f}')
    if result[1] < 0.05:
        print(f'{series_name} is stationary.')
    else:
        print(f'{series_name} is non-stationary.')

check_stationarity(btc_data['log_volume'], 'log_volume')
check_stationarity(btc_data['log_return'], 'log_return')
check_stationarity(btc_data['volatility'], 'volatility')

# 由于 log_volume 不平稳，进行差分
btc_data['diff_log_volume'] = btc_data['log_volume'].diff()
btc_data['diff_volatility'] = btc_data['volatility'].diff()
btc_data.dropna(inplace=True)

# 准备数据，选择差分后的变量作为分析变量
data = btc_data[['diff_log_volume', 'log_return', 'diff_volatility']]

# 使用 VAR 模型来选择最佳滞后阶数
model = VAR(data)
bic_values = []
aic_values = []
hqic_values = []
max_lag = 10  # 根据数据频率调整

# 计算每个滞后阶数下的 AIC、BIC 和 HQIC 值
for p in range(1, max_lag + 1):
    result = model.fit(p)
    bic_values.append((p, result.bic))
    aic_values.append((p, result.aic))
    hqic_values.append((p, result.hqic))

# 找到不同信息准则下的最佳滞后阶数
best_lag_bic = min(bic_values, key=lambda x: x[1])[0]
best_lag_aic = min(aic_values, key=lambda x: x[1])[0]
best_lag_hqic = min(hqic_values, key=lambda x: x[1])[0]

# 输出最佳滞后阶数
print(f"\nBest lag based on BIC: {best_lag_bic}")
print(f"Best lag based on AIC: {best_lag_aic}")
print(f"Best lag based on HQIC: {best_lag_hqic}")

# 输出所有滞后阶数的 BIC、AIC 和 HQIC 值
print("\nLag order and Information Criterion values:")
for p in range(1, max_lag + 1):
    print(f"Lag: {p}, BIC: {bic_values[p-1][1]:.4f}, AIC: {aic_values[p-1][1]:.4f}, HQIC: {hqic_values[p-1][1]:.4f}")

# 选择基于 BIC 的最佳模型
best_model = model.fit(best_lag_bic)

# 模型诊断：残差自相关检验
for i, col in enumerate(data.columns):
    lb_test = acorr_ljungbox(best_model.resid.iloc[:, i], lags=10, return_df=True)
    print(f'\nResiduals for variable {col}:')
    print(lb_test)

# 稳定性检验
if best_model.is_stable():
    print("\nThe VAR model is stable.")
else:
    print("\nThe VAR model is not stable.")
