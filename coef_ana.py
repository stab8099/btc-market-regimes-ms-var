# 导入必要的库
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 加载数据
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"
btc_data = pd.read_csv(data_path)

# 将 'Start' 列转换为日期格式，并筛选出2017年1月1日之后的数据
btc_data['Start'] = pd.to_datetime(btc_data['Start'])
btc_data = btc_data[btc_data['Start'] >= '2017-01-01'].copy()

# 计算变量
btc_data['log_volume'] = np.log(btc_data['Volume'])
btc_data['return'] = np.log(btc_data['Close']).diff()
btc_data['volatility'] = btc_data['return'] ** 2

# 删除包含NA值的行
btc_data.dropna(inplace=True)

# 定义函数来计算F统计量和p值
def calculate_f_p_value(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    f_stat = model.fvalue
    p_value = model.f_pvalue
    return f_stat, p_value

# 初始化一个字典来存储每对变量的结果
results = {}

# 计算每对变量的F统计量和p值
f_stat, p_value = calculate_f_p_value(btc_data['log_volume'], btc_data['return'])
results['log_volume vs return'] = (f_stat, p_value)

f_stat, p_value = calculate_f_p_value(btc_data['log_volume'], btc_data['volatility'])
results['log_volume vs volatility'] = (f_stat, p_value)

f_stat, p_value = calculate_f_p_value(btc_data['volatility'], btc_data['log_volume'])
results['volatility vs log_volume'] = (f_stat, p_value)

f_stat, p_value = calculate_f_p_value(btc_data['volatility'], btc_data['return'])
results['volatility vs return'] = (f_stat, p_value)

f_stat, p_value = calculate_f_p_value(btc_data['return'], btc_data['log_volume'])
results['return vs log_volume'] = (f_stat, p_value)

f_stat, p_value = calculate_f_p_value(btc_data['return'], btc_data['volatility'])
results['return vs volatility'] = (f_stat, p_value)

# 绘制散点图并保存为图片
pairs = [
    ('log_volume', 'return'),
    ('log_volume', 'volatility'),
    ('volatility', 'log_volume'),
    ('volatility', 'return'),
    ('return', 'log_volume'),
    ('return', 'volatility')
]

for x_var, y_var in pairs:
    plt.figure(figsize=(8, 6))
    plt.scatter(btc_data[x_var], btc_data[y_var], alpha=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'Scatter plot of {x_var} vs {y_var}')
    
    # 获取F统计量和p值
    f_stat, p_value = results[f'{x_var} vs {y_var}']
    plt.figtext(0.15, 0.8, f'F-statistic: {f_stat:.4f}\np-value: {p_value:.4f}', fontsize=10, color='black')
    
    # 保存图片
    plt.savefig(f"{x_var}_vs_{y_var}.png")
    plt.close()

# 输出计算结果
for key, (f_stat, p_value) in results.items():
    print(f"{key}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")