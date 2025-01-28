import numpy as np
def american_option_pricing(S0, K, T, M, r, sigma, option_type):
    dt = T / M
    discount = np.exp(-r * dt)
    u = np.exp(sigma * np.sqrt(dt) + (r - 0.5 * sigma**2) * dt)
    d = np.exp(-sigma * np.sqrt(dt) + (r - 0.5 * sigma**2) * dt)
    p_tilde = (np.exp(r * dt) - d) / (u - d)
    stock_tree = np.zeros((M + 1, M + 1))
    for i in range(M + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
    option_tree = np.zeros((M + 1, M + 1))
    if option_type == "call":
        option_tree[:, M] = np.maximum(stock_tree[:, M] - K, 0)
    elif option_type == "put":
        option_tree[:, M] = np.maximum(K - stock_tree[:, M], 0)
    optimal_strategy = np.zeros((M + 1, M + 1))
    for i in range(M - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p_tilde * option_tree[j, i + 1] + (1 - p_tilde) * option_tree[j + 1, i + 1])
            if option_type == "call":
                exercise_value = stock_tree[j, i] - K
            elif option_type == "put":
                exercise_value = K - stock_tree[j, i]
            option_tree[j, i] = max(continuation_value, exercise_value)
            optimal_strategy[j, i] = 1 if exercise_value > continuation_value else 0
    return option_tree, optimal_strategy
S0 = 100
K = 100
T = 1
M = 5
r = 0.08
sigma = 0.2
put_prices, put_strategy = american_option_pricing(S0, K, T, M, r, sigma, "put")

print("Put Option Prices at All Time Points:")
print(put_prices)
print("\nOptimal Exercise Strategy for Put Option (1 for Exercise, 0 for Hold):")
print(put_strategy)
