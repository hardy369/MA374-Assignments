import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

path = "C:/Users/hp/Downloads/52h.ramineniMA374lab04/merged_monthly_opening_prices.csv"
def get_online_data():
    companies = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "YESBANK.NS", "BAJFINANCE.NS", "INDUSINDBK.NS", "AXISBANK.NS", "PNB.NS", "RBLBANK.NS"]

    def get_monthly_opening_prices(ticker):
        stock_data = yf.download(ticker, start="2019-02-01", end="2024-02-01", interval='1mo')
        return stock_data['Open']

    for company in companies:
        monthly_opening_prices = get_monthly_opening_prices(company)
        monthly_opening_prices.to_csv(f"{company}_monthly_opening_prices.csv", header=True)
        
    dfs = [pd.read_csv(f"{company}_monthly_opening_prices.csv", index_col=0, parse_dates=True, names=[company]) for company in companies]
    merged_opening_prices = pd.concat(dfs, axis=1)
    merged_opening_prices.to_csv("merged_monthly_opening_prices.csv")


def minimum_variance_portfolio(m, C):
    C_inv = np.linalg.inv(C)
    u = np.ones(len(m))
    uT = np.transpose(u)
    mT = np.transpose(m)

    weight_min_var = u @ C_inv / (u @ C_inv @ uT)
    mu_min_var = weight_min_var @ mT
    risk_min_var = math.sqrt(np.dot(np.dot(weight_min_var,C),np.transpose(weight_min_var)))

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

def plot_fixed(x, y, x_axis, y_axis, title):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis) 
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_fixed_both(x1, y1, x2, y2, x_axis, y_axis, title):
    plt.plot(x1, y1, color = 'Blue', label = 'Minimum Variance Curve')
    plt.plot(x2, y2, color = 'Green', label = 'CML')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis) 
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def q2(generate_data):
    if generate_data == 1:
        get_online_data()

    df = pd.read_csv(path)
    df.set_index('Date', inplace=True)
    df = df.pct_change()
    m = np.mean(df, axis = 0) * 12
    C = df.cov()

    returns = np.linspace(-3, 5, num = 5000)
    u = np.ones(len(m))
    risk = []

    for mu in returns:
        w,var = minimum_variance_line(m, C, mu)
        risk.append(var)
  
    risk_min_var, mu_min_var = minimum_variance_portfolio(m,C)

    returns1, risk1, returns2, risk2 = [], [], [], []
    for i in range(len(returns)):
        if returns[i] >= mu_min_var: 
            returns1.append(returns[i])
            risk1.append(risk[i])
        else:
            returns2.append(returns[i])
            risk2.append(risk[i])


    risk_free_rate = 0.05

    market_weights = (m - risk_free_rate * u) @ np.linalg.inv(C) / ((m - risk_free_rate * u) @ np.linalg.inv(C) @ np.transpose(u) )
    mu_market = market_weights @ np.transpose(m)
    risk_market = math.sqrt(market_weights @ C @ np.transpose(market_weights))

    plt.plot(risk1, returns1, color='mediumseagreen', label='Efficient frontier')  
    plt.plot(risk2, returns2, color='royalblue') 
    plt.xlabel("Risk (var)")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve & Efficient Frontier")
    plt.plot(risk_market, mu_market, color='orange', marker='o')  
    plt.annotate(f'Market Portfolio\n(Risk: {round(risk_market, 2)}, Return: {round(mu_market, 2)})',
             xy=(risk_market, mu_market), xytext=(0.2, 0.6),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.plot(risk_min_var, mu_min_var, color='orange', marker='o') 
    plt.annotate(f'Min. Variance Portfolio\n(Risk: {round(risk_min_var, 2)}, Return: {round(mu_min_var, 2)})',
             xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.legend()
    plt.grid(True)
    plt.show()


    print()
    print("Market Portfolio Weights = ", market_weights)
    print("Return = ", mu_market)
    print("Risk = ", risk_market * 100, " %")


    returns_cml = []
    risk_cml = np.linspace(0, 2, num = 5000)
    for i in risk_cml:
        returns_cml.append(risk_free_rate + (mu_market - risk_free_rate) * i / risk_market)


    slope, intercept = (mu_market - risk_free_rate) / risk_market, risk_free_rate
    print()
    print("Equation of Capital Market Line is:")
    print(f"y = {slope:.2f} x + {intercept:.2f}\n")

    plot_fixed_both(risk, returns, risk_cml, returns_cml, "Risk", "Returns", "Capital Market Line with Markowitz Efficient Frontier")
    plot_fixed(risk_cml, returns_cml, "Risk", "Returns", "Capital Market Line")


    beta_k = np.linspace(-1, 1, 5000)
    mu_k = risk_free_rate + (mu_market - risk_free_rate) * beta_k
    plt.plot(beta_k, mu_k)

    print("Equation of Security Market Line is:",f"mu = {mu_market - risk_free_rate:.2f} beta + {risk_free_rate:.2f}")
    print()

    plt.title('Security Market Line for the 10 assets')
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()

q2(0)