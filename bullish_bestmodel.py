import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# 文件路径（请确保该文件存在）
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"

# 读取数据并解析日期列，明确指定日期格式为 %Y-%m-%d
btc_data = pd.read_csv(data_path)
btc_data['Start'] = pd.to_datetime(btc_data['Start'], format='%Y-%m-%d', errors='coerce')

# 牛市数据范围（根据您的需要调整）
start_date = pd.to_datetime("2023-09-10")
end_date = pd.to_datetime("2024-11-05")

# 筛选牛市阶段的数据
btc_data = btc_data[(btc_data['Start'] >= start_date) & (btc_data['Start'] <= end_date)].copy()

# 将日期设为索引，并设置数据频率为日频
btc_data.set_index('Start', inplace=True)
btc_data = btc_data.asfreq('D')
btc_data.fillna(method='ffill', inplace=True)

# 定义变量
btc_data['log_volume'] = np.log(btc_data['Volume'])
# 使用更合理的收益率定义
btc_data['return'] = np.log(btc_data['Close']).diff()
btc_data['volatility'] = btc_data['return'] ** 2

# 构建VAR数据集
var_data = btc_data[['log_volume', 'return', 'volatility']].dropna()

# 使用VAR选择最佳滞后阶数 (以BIC为准)
max_lag = 10
model = VAR(var_data)
bic_values = []
for p in range(1, max_lag+1):
    result = model.fit(p)
    bic_values.append((p, result.bic))

best_lag_bic = min(bic_values, key=lambda x: x[1])[0]
print("Lag order and BIC values:")
for p, bic in bic_values:
    print(f"Lag: {p}, BIC: {bic:.4f}")
print(f"\nBest lag based on BIC: {best_lag_bic}")

# 使用最佳滞后阶数拟合最终VAR模型
best_model = model.fit(best_lag_bic)

# 打印最终模型参数矩阵
print("\nFinal Model Parameters:")
print(best_model.params)

# ---------------------
# 以下为分析状态收益的代码
# 计算长期均衡均值、IRF、FEVD，以及实际数据统计
# ---------------------

variable_names = best_model.names
p = best_model.k_ar
params = best_model.params
nvars = len(variable_names)

# 1. 长期均衡均值
c = params.loc['const'].values
A_matrices = []
for i in range(1, p+1):
    A_i = params.loc[[f'L{i}.{var}' for var in variable_names]].values
    A_matrices.append(A_i)

I = np.eye(nvars)
A_sum = I.copy()
for A in A_matrices:
    A_sum = A_sum - A

try:
    A_inv = np.linalg.inv(A_sum)
    long_run_mean = A_inv @ c
    long_run_df = pd.DataFrame(long_run_mean, index=variable_names, columns=['Long-run mean'])
    print("\nLong-run equilibrium levels:")
    print(long_run_df)
    long_run_return = long_run_df.loc['return', 'Long-run mean']
    print(f"\nLong-run mean of return: {long_run_return:.6f}")
except np.linalg.LinAlgError:
    print("The model might not be stable or invertible, cannot compute long-run means.")
    long_run_return = None

# 2. 脉冲响应函数 (IRF)
irf_horizon = 10
irf = best_model.irf(irf_horizon)
irf_response = irf.irfs[:, variable_names.index('return'), variable_names.index('return')]
irf_df = pd.DataFrame(irf_response, columns=['IRF: return->return'])
print("\nImpulse Response of 'return' to its own positive shock:")
print(irf_df)

# 3. 预测误差方差分解 (FEVD)
fevd = best_model.fevd(irf_horizon)
fevd_return = fevd.decomp[:, variable_names.index('return'), variable_names.index('return')]
fevd_df = pd.DataFrame(fevd_return, columns=['FEVD(return->return)'])
print("\nFEVD of return:")
print(fevd_df)

# 4. 实际数据统计
actual_return_mean = var_data['return'].mean()
actual_return_std = var_data['return'].std()
print("\nActual data stats (in the given state period):")
print(f"Mean return: {actual_return_mean:.6f}")
print(f"Std of return: {actual_return_std:.6f}")

# 通过以上指标（长期均衡、IRF、FEVD、实际数据均值），综合判断该状态下的收益特征。
