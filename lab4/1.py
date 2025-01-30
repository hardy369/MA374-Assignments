import numpy as np
import math
import matplotlib.pyplot as plt

m = [0.1, 0.25, 0.15]
C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]


def minimum_variance_portfolio(m, C):
    u = np.ones(len(m))
    uT = np.transpose(u)
    mT = np.transpose(m)

    weight_min_var = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ uT)
    mu_min_var = weight_min_var @ mT
    risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))

    return risk_min_var, mu_min_var


def minimum_variance_line(m, C, mu):
    C_inv = np.linalg.inv(C)
    u = np.ones(len(m))
    uT = np.transpose(u)
    mT = np.transpose(m)

    p = [[1, u @ C_inv @ mT], [mu, m @ C_inv @ mT]]
    q = [[u @ C_inv @ uT, 1], [m @ C_inv @ uT, mu]]
    r = [[u @ C_inv @ uT, u @ C_inv @ mT], [m @ C_inv @ uT, m @ C_inv @ mT]]

    det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)

    w = (det_p * (u @ C_inv) + det_q * (m @ C_inv)) / det_r
    var = math.sqrt(np.dot(np.dot(w, C), np.transpose(w)))

    return w, var


def q1a():
    weights10, return10, risk10 = [], [], []
    weights15, return15 = [], []

    returns = np.linspace(0, 0.5, 10000)
    risk = []

    count = 0
    for mu_v in returns:
        weights, var = minimum_variance_line(m, C, mu_v)
        risk.append(var)
        count += 1

        if count % 1000 == 0:
            weights10.append(weights)
            return10.append(mu_v)
            risk10.append(var * var)

        if abs(var - 0.15) < math.pow(10, -4.5):
            weights15.append(weights)
            return15.append(mu_v)

    min_var, mu_min_var = minimum_variance_portfolio(m, C)
    high_returns, high_return_risk, low_returns, low_return_risk = [], [], [], []

    for i in range(len(returns)):
        if returns[i] >= mu_min_var:
            high_returns.append(returns[i])
            high_return_risk.append(risk[i])
        else:
            low_returns.append(returns[i])
            low_return_risk.append(risk[i])

    plt.plot(high_return_risk, high_returns, color='mediumseagreen', label='Efficient frontier')  
    plt.plot(low_return_risk, low_returns, color='lightcoral')  
    plt.xlabel("Sigma")
    plt.ylabel("Mu")
    plt.title("Efficient Frontier plot | Minimum Variance Line")
    plt.plot(min_var, mu_min_var, color='mediumseagreen', marker='+')
    plt.annotate(f'Min. Variance Portfolio\n(Risk: {round(min_var, 2)}, Return: {round(mu_min_var, 2)})',
             xy=(min_var, mu_min_var), xytext=(min_var+0.05, mu_min_var),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.legend()
    plt.grid(True)
    plt.show()

    return weights10, return10, risk10, weights15, return15


def q1b_1c():
    weights10, return10, risk10, weights15, return15 = q1a()
    print("Index\tWeights\t\t\t\t\t\tReturn\t\t\tRisk\n")
    for i in range(10):
        print(f"{i+1}.\t{weights10[i]}\t\t{return10[i]:.4f}\t\t\t{risk10[i]:.6f}")

    min_return, max_return = return15[0], return15[1]
    min_return_weights, max_return_weights = weights15[0], weights15[1]

    if min_return > max_return:
        min_return, max_return = max_return, min_return
        min_return_weights, max_return_weights = max_return_weights, min_return_weights

    print(f"Minimum return = {min_return:.2f}")
    print(f"Weights = {min_return_weights}")

    print()

    print(f"Maximum return = {max_return:.2f}")
    print(f"Weights = {max_return_weights}")


def q1d():
    print()
    weights, var = minimum_variance_line(m, C, 0.18)
    print(f"Minimum risk = {var*100:.6f}", " %")
    print("Weights = ", weights)


def q1e_1f():
    risk_free_rate = 0.08
    u = np.array([1, 1, 1])
    market_weights = (m - risk_free_rate * u) @ np.linalg.inv(C) / (
            (m - risk_free_rate * u) @ np.linalg.inv(C) @ np.transpose(u))
    mu_market = np.dot(market_weights, np.transpose(m))
    risk_market = math.sqrt(np.dot(np.dot(market_weights, C), np.transpose(market_weights)))

    print(f"Market Portfolio Weights = {market_weights}")
    print(f"Market Return = {mu_market:.2f}")
    print(f"Market Risk = {risk_market*100:.6f}", "%")

    returns_cml = []
    risk_cml = np.linspace(0, 1, 10000)
    for i in risk_cml:
        returns_cml.append(risk_free_rate + (mu_market - risk_free_rate) * i / risk_market)

    slope, intercept = (mu_market - risk_free_rate) / risk_market, risk_free_rate
    print()
    print("Equation of Capital Market Line is:",f"y = {slope:.2f} x + {intercept:.2f}\n")

    returns = np.linspace(0, 0.9, 10000)
    risk = []
    for mu_v in returns:
        weights, var = minimum_variance_line(m, C, mu_v)
        risk.append(var)

    plt.scatter(risk_market, mu_market, color='orange',marker='+', label='Market portfolio')
    plt.annotate(f'Market Portfolio\n(Risk: {round(risk_market, 2)}, Return: {round(mu_market, 2)})',
             xy=(risk_market, mu_market), xytext=(risk_market+0.05, mu_market),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.plot(risk, returns, color='mediumseagreen', label='Minimum variance curve') 
    plt.plot(risk_cml, returns_cml, color='royalblue', label='CML')  
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Capital Market Line | Minimum variance curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    sigma = 0.1
    mu_curr = (mu_market - risk_free_rate) * sigma / risk_market + risk_free_rate
    weight_rf = (mu_curr - mu_market) / (risk_free_rate - mu_market)
    weights_risk = (1 - weight_rf) * market_weights

    print("Risk =", sigma * 100, "%")
    print("Risk-free weights =", weight_rf)
    print("Risky Weights =", weights_risk)
    print(f"Returns ={mu_curr:.2f}")

    print("\n\n")

    sigma = 0.25
    mu_curr = (mu_market - risk_free_rate) * sigma / risk_market + risk_free_rate
    weight_rf = (mu_curr - mu_market) / (risk_free_rate - mu_market)
    weights_risk = (1 - weight_rf) * market_weights

    print("Risk =", sigma * 100, "%")
    print("Risk-free weights =", weight_rf)
    print("Risky Weights =", weights_risk)
    print(f"Returns ={mu_curr:.2f}")



q1b_1c()
q1d()
q1e_1f()
