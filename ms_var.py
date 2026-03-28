import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

data_path = "/Users/liuqinyu/Desktop/datasci_postgrad/data thinking/gpwk/eth.csv"
btc_data = pd.read_csv(data_path)

btc_data['Start'] = pd.to_datetime(btc_data['Start'])
btc_data = btc_data[btc_data['Start'] >= '2017-01-01'].copy()

btc_data['log_volume'] = np.log(btc_data['Volume'])

# 使用更合理的收益率定义
btc_data['return'] = np.log(btc_data['Close']) - np.log(btc_data['Open'])
btc_data['volatility'] = btc_data['return'] ** 2

btc_data.dropna(inplace=True)

data = btc_data[['log_volume', 'return', 'volatility']].values 

n_states = 3 
p = 7
n_vars = data.shape[1]

#1.初始参数设置

def initialize_params(n_states, n_vars, p, data):
    data_mean = np.mean(data, axis=0)
    data_cov = np.cov(data, rowvar=False)

    # 1.1.初始化转移矩阵：随机生成一个非负矩阵后归一化，使行和为1
    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix += 1e-3
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    # 初始化系数矩阵：n_states组VAR系数，每组为 n_vars x (n_vars*p + 1)
    # 其中+1是截距项。此处初始化为0.01和0
    coefficients = []
    for _ in range(n_states):
        coef = np.zeros((n_vars, n_vars * p + 1))
        coef[:, 0] = 0.01
        coefficients.append(coef)

    # 初始化协方差矩阵：以数据协方差为基础，加上微小正定项
    covariances = [data_cov + 1e-6 * np.eye(n_vars) for _ in range(n_states)]

    return transition_matrix, coefficients, covariances

def create_lagged_matrix(Y, p):
    T, n_vars = Y.shape
    # 构建滞后矩阵：X = [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    lagged_Y = np.hstack([Y[p - i - 1:T - i - 1] for i in range(p)])
    lagged_Y = np.hstack([np.ones((T-p, 1)), lagged_Y])
    return lagged_Y, Y[p:]

def var_model(Y, coefficients, p):
    lagged_Y, _ = create_lagged_matrix(Y, p)
    var_outputs = [np.dot(lagged_Y, coef.T) for coef in coefficients]
    return var_outputs

def forward_backward(obs_lik, transition_matrix, init_state_dist):
    T_eff, n_states = obs_lik.shape
    alpha = np.zeros((T_eff, n_states))
    c = np.zeros(T_eff)
    alpha[0] = init_state_dist * obs_lik[0]
    c[0] = alpha[0].sum()
    if c[0] < 1e-15:
        c[0] = 1e-15
    alpha[0] /= c[0]

    for t in range(1, T_eff):
        alpha[t] = (alpha[t-1] @ transition_matrix) * obs_lik[t]
        c[t] = alpha[t].sum()
        if c[t] < 1e-15:
            c[t] = 1e-15
        alpha[t] /= c[t]

    log_likelihood = np.sum(np.log(c))

    beta = np.zeros((T_eff, n_states))
    beta[-1] = 1.0
    for t in range(T_eff-2, -1, -1):
        beta[t] = (transition_matrix @ (obs_lik[t+1]*beta[t+1])) / c[t+1]

    gamma = np.zeros((T_eff, n_states))
    xi = np.zeros((T_eff-1, n_states, n_states))
    for t in range(T_eff):
        gamma[t] = alpha[t] * beta[t]
        denom = gamma[t].sum()
        if denom < 1e-15:
            denom = 1e-15
        gamma[t] /= denom

    for t in range(T_eff-1):
        denominator = (alpha[t] @ transition_matrix) @ (obs_lik[t+1]*beta[t+1])
        if denominator < 1e-15:
            denominator = 1e-15
        for i in range(n_states):
            numerator = alpha[t, i]*transition_matrix[i]*obs_lik[t+1]*beta[t+1]
            xi[t, i] = numerator / denominator

    return gamma, xi, log_likelihood

# 3.误差项分布假设和协方差结构
def em_step(Y, n_states, p, transition_matrix, coefficients, covariances, init_state_dist):
    T, n_vars = Y.shape
    T_eff = T - p

    var_outputs = var_model(Y, coefficients, p)

    # 确保obs_lik不为0
    obs_lik = np.zeros((T_eff, n_states))
    for k in range(n_states):
        for t in range(T_eff):
            val = multivariate_normal.pdf(Y[p:][t], mean=var_outputs[k][t], cov=covariances[k])
            obs_lik[t, k] = max(val, 1e-15)

    gamma, xi, log_likelihood = forward_backward(obs_lik, transition_matrix, init_state_dist)

    lagged_Y, Y_actual = create_lagged_matrix(Y, p)

    # 带正则化的加权最小二乘
    for k in range(n_states):
        weights = gamma[:, k]
        W = np.diag(weights)
        X = lagged_Y
        Yk = Y_actual
        XTWX = X.T @ W @ X
        XTWY = X.T @ W @ Yk

        # 对XTWX加上一个ridge项，减少SVD失败的可能性
        XTWX_reg = XTWX + 1e-6 * np.eye(XTWX.shape[0])
        coef_k = np.linalg.lstsq(XTWX_reg, XTWY, rcond=1e-10)[0].T
        coefficients[k] = coef_k

        residuals = Yk - X @ coef_k.T
        cov_k = (residuals.T @ (W @ residuals)) / np.sum(weights)
        covariances[k] = cov_k + 1e-6 * np.eye(n_vars)

    sum_xi = np.sum(xi, axis=0)
    row_sum = sum_xi.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-15] = 1e-15
    transition_matrix = sum_xi / row_sum

    init_state_dist = gamma[0]

    return transition_matrix, coefficients, covariances, init_state_dist, log_likelihood

# 4. 收敛标准与终止条件
def fit_ms_var_with_random_initialization(Y, n_states, p, max_iter=100, tol=1e-4, num_initializations=10):
    best_log_likelihood = -np.inf
    best_transition_matrix = None
    best_coefficients = None
    best_covariances = None
    best_init_state_dist = None

    for i in range(num_initializations):
        print(f"Initialization {i+1}/{num_initializations}")
        
        transition_matrix, coefficients, covariances = initialize_params(n_states, Y.shape[1], p, Y)
        init_state_dist = np.ones(n_states) / n_states
        prev_log_likelihood = None

        for iteration in range(max_iter):
            transition_matrix, coefficients, covariances, init_state_dist, log_likelihood = em_step(
                Y, n_states, p, transition_matrix, coefficients, covariances, init_state_dist)

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
            best_init_state_dist = init_state_dist

    print("\nBest Model After Multiple Initializations:")
    print("Best Log Likelihood:", best_log_likelihood)
    print("Estimated Transition Matrix:", best_transition_matrix)
    print("Estimated Coefficients:", best_coefficients)
    print("Estimated Covariances:", best_covariances)
    print("Estimated Initial State Distribution:", best_init_state_dist)

    return best_transition_matrix, best_coefficients, best_covariances

transition_matrix, coefficients, covariances = fit_ms_var_with_random_initialization(data, n_states, p)
print("Final Estimated Transition Matrix:", transition_matrix)
print("Final Estimated Coefficients:", coefficients)
print("Final Estimated Covariances:", covariances)

