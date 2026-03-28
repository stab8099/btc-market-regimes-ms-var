import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# 文件与路径配置
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/eth.csv"

# 1. 读取数据并解析日期列（指定dayfirst=True以解析日/月/年格式）
btc_data = pd.read_csv(data_path)
btc_data['Start'] = pd.to_datetime(btc_data['Start'], dayfirst=True)

# 这里假设您的牛市时期为 2023年09月10日至2024年03月09日（如需更改请调整日期）
start_date = pd.to_datetime("2023-10-13")
end_date = pd.to_datetime("2024-03-09")

# 筛选出牛市时间段的数据
btc_data = btc_data[(btc_data['Start'] >= start_date) & (btc_data['Start'] <= end_date)].copy()

# 将Start设为索引以建立时间序列索引
btc_data.set_index('Start', inplace=True)
# 为数据设定为每日频率，如果数据中缺日期会生成NaN行
btc_data = btc_data.asfreq('D')


# 2. 定义所需变量
btc_data['log_volume'] = np.log(btc_data['Volume'])
# 使用更合理的收益率定义
btc_data['return'] = np.log(btc_data['Close']) - np.log(btc_data['Open'])
btc_data['volatility'] = btc_data['High'] - btc_data['Low']

# 从数据中提取需要用于VAR模型的变量
var_data = btc_data[['log_volume', 'return', 'volatility']]

# 3. 构建VAR模型并拟合 (p=7)
p = 7
model = VAR(var_data)
results = model.fit(p)

# 4. 输出参数估计结果
print("Coefficient matrix (parameters):")
print(results.params)  # 包括截距项和滞后项系数
