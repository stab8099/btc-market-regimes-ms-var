import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
from scipy.stats import multivariate_normal

# 读取数据并预处理
data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/btc.csv"
btc_data = pd.read_csv(data_path)
btc_data['Start'] = pd.to_datetime(btc_data['Start'])
btc_data = btc_data[btc_data['Start'] >= '2017-01-01'].copy()

# 计算 log_volume, return 和 volatility
btc_data['log_volume'] = np.log(btc_data['Volume'])
btc_data['return'] = np.log(btc_data['Close']).diff()
btc_data['volatility'] = btc_data['return'] ** 2
btc_data.dropna(inplace=True)

# 准备数据，选择 log_volume, return 和 volatility 作为分析变量
data = btc_data[['log_volume', 'return', 'volatility']].values

# 定义ms-var模型的参数
n_states = 3
p = 8
n_vars = data.shape[1]

# 初始参数
def initialize_params(n_states, n_vars, p):
    transition_matrix = np.random.dirichlet(np.ones(n_states), size=n_states)
    coefficients = [np.random.randn(n_vars, n_vars * p + 1) for _ in range(n_states)]
    covariances = [np.eye(n_vars) for _ in range(n_states)]
    return transition_matrix, coefficients, covariances

# 创建滞后矩阵
def create_lagged_matrix(Y, p):
    T, n_vars = Y.shape
    lagged_Y = np.hstack([Y[p-i-1:T-i-1] for i in range(p)])
    lagged_Y = np.hstack([np.ones((T-p, 1)), lagged_Y])
    return lagged_Y, Y[p:]

# VAR模型输出
def var_model(Y, coefficients, p):
    lagged_Y, _ = create_lagged_matrix(Y, p)
    return [np.dot(lagged_Y, coef.T) for coef in coefficients]

# EM算法的E步和M步
def em_step_with_noise(Y, n_states, p, transition_matrix, coefficients, covariances):
    T, n_vars = Y.shape
    log_likelihoods = np.zeros((T-p, n_states))
    epsilon = 1e-6

    var_outputs = var_model(Y, coefficients, p)
    for k in range(n_states):
        likelihoods = np.array([
            max(multivariate_normal.pdf(Y[p:][t], mean=var_outputs[k][t], cov=covariances[k]), 1e-10)
            for t in range(T - p)
        ])
        log_likelihoods[:, k] = np.log(likelihoods)
    
    responsibilities = np.exp(log_likelihoods - log_likelihoods.max(axis=1, keepdims=True))
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    noise_level = 1e-3
    for k in range(n_states):
        weighted_Y = responsibilities[:, k][:, np.newaxis] * Y[p:]
        lagged_Y, _ = create_lagged_matrix(Y, p)
        X = lagged_Y
        coefficients[k] = np.linalg.lstsq(X.T @ (responsibilities[:, k][:, np.newaxis] * X), X.T @ weighted_Y, rcond=1e-5)[0].T
        coefficients[k] += noise_level * np.random.randn(*coefficients[k].shape)
        residuals = weighted_Y - X @ coefficients[k].T
        covariances[k] = (residuals.T @ (responsibilities[:, k][:, np.newaxis] * residuals)) / responsibilities[:, k].sum()
        covariances[k] += epsilon * np.eye(n_vars)
    
    transition_matrix = np.zeros_like(transition_matrix)
    for i in range(n_states):
        for j in range(n_states):
            transition_matrix[i, j] = responsibilities[:-1, i].dot(responsibilities[1:, j])
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix, coefficients, covariances

# 运行EM算法
def fit_ms_var_with_random_initialization(Y, n_states, p, max_iter=200, tol=1e-8, num_initializations=10):
    best_log_likelihood = -np.inf
    best_transition_matrix = None
    best_coefficients = None
    best_covariances = None

    for i in range(num_initializations):
        print(f"Initialization {i+1}/{num_initializations}")
        
        transition_matrix, coefficients, covariances = initialize_params(n_states, Y.shape[1], p)
        prev_log_likelihood = None
        T = Y.shape[0]

        for iteration in range(max_iter):
            transition_matrix, coefficients, covariances = em_step_with_noise(Y, n_states, p, transition_matrix, coefficients, covariances)
            log_likelihood = 0
            var_outputs = var_model(Y, coefficients, p)
            for t in range(T - p):
                likelihood_sum = 0
                for k in range(n_states):
                    likelihood_sum += multivariate_normal.pdf(Y[p:][t], mean=var_outputs[k][t], cov=covariances[k]) * transition_matrix[:, k]
                log_likelihood += np.log(np.maximum(np.sum(likelihood_sum), 1e-10))
            
            print(f"Iteration {iteration+1}, Log Likelihood: {log_likelihood}")
            
            if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tol:
                print(f"Converged in {iteration+1} iterations")
                break
            prev_log_likelihood = log_likelihood

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_transition_matrix = transition_matrix
            best_coefficients = coefficients
            best_covariances = covariances

    print("\nBest Model After Multiple Initializations:")
    print("Best Log Likelihood:", best_log_likelihood)
    print("Estimated Transition Matrix:", best_transition_matrix)
    print("Estimated Coefficients:", best_coefficients)
    print("Estimated Covariances:", best_covariances)
    
    return best_transition_matrix, best_coefficients, best_covariances

# 运行模型并选取最佳结果
transition_matrix, coefficients, covariances = fit_ms_var_with_random_initialization(data, n_states, p)

# 状态分类（根据最大后验概率）
log_likelihoods = np.zeros((len(data) - p, n_states))
var_outputs = var_model(data, coefficients, p)
for k in range(n_states):
    log_likelihoods[:, k] = np.array([
        multivariate_normal.logpdf(data[p:][t], mean=var_outputs[k][t], cov=covariances[k])
        for t in range(len(data) - p)
    ])

state_sequence = log_likelihoods.argmax(axis=1) + 1
btc_data = btc_data.iloc[p:].copy()
btc_data['state'] = state_sequence

# 分析非对称性
print("\nAsymmetry Analysis by State")

for state in range(1, n_states + 1):
    state_data = btc_data[btc_data['state'] == state]
    skew_volume = stats.skew(state_data['log_volume'])
    skew_volatility = stats.skew(state_data['volatility'])
    print(f"State {state}: Skewness of Volume: {skew_volume}, Skewness of Volatility: {skew_volatility}")