import numpy as np
import matplotlib.pyplot as plt

# 示例数据，假设 Estimated Covariances 已经保存到一个变量 covariances
# 每个 covariances[i] 表示不同隐状态的协方差矩阵
covariances = [np.array([[ 5.68857086e-02, -8.48149719e-04,  6.57994090e-06],
       [-8.48149719e-04,  4.65613635e-03,  2.47078751e-05],
       [ 6.57994090e-06,  2.47078751e-05,  8.83866098e-06]]), np.array([[ 0.15380732, -0.01137487,  0.00215566],
       [-0.01137487,  0.01848491, -0.00069414],
       [ 0.00215566, -0.00069414,  0.00064507]]), np.array([[ 4.56254721e-02, -2.10968835e-04, -8.89574122e-06],
       [-2.10968835e-04,  5.42842684e-04,  2.72847982e-06],
       [-8.89574122e-06,  2.72847982e-06,  1.47540336e-06]])]


# 定义交易量（Volume）和波动（Volatility）的索引位置
volume_index = 0  # 假设 Volume 在第一个位置
volatility_index = 2  # 假设 Volatility 在第三个位置

# 提取每个状态下交易量与波动之间的协方差值
cov_values = [cov[volume_index, volatility_index] for cov in covariances]

# 打印每个状态下的协方差值
for i, cov_value in enumerate(cov_values, start=1):
    relation = "正相关" if cov_value > 0 else "负相关" if cov_value < 0 else "无明显相关"
    print(f"状态 {i} 下的交易量与波动的协方差：{cov_value:.4f}，关系：{relation}")

# 可视化不同状态下交易量和波动的协方差
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cov_values) + 1), cov_values, color='skyblue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("status number")
plt.ylabel("covariance between trading volume and volatility")
plt.title("the relationship between trading volume and volatility covariance across different states")
plt.show()

# Print each covariance matrix to manually verify the positions of volume and volatility
for i, cov in enumerate(covariances, start=1):
    print(f"状态 {i} 的协方差矩阵：\n{cov}\n")