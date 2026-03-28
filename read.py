import pandas as pd

# 文件路径
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"

# 尝试读取数据
btc_data = pd.read_csv(data_path)

# 显示前五行数据，检查列名和数据类型
print("First 5 rows of the data:")
print(btc_data.head())

# 显示数据信息，包括列名、非空值数量、数据类型等
print("\nData Info:")
print(btc_data.info())

# 显示数据的基本统计信息
print("\nData Description:")
print(btc_data.describe())
