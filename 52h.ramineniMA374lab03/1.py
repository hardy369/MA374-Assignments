import numpy as np
import math
import matplotlib.pyplot as plt

def plot_fixed(x, y, x_axis, y_axis, title):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis) 
    plt.title(title)
    plt.show()

def check_arbitrage(u, d, r, t):
    if d < math.exp(r*t) and math.exp(r*t) < u:
        return False
    else:
        return True

def american_binomial_model(S0, K, T, M, r, sigma, display=True):
    t = T/M
    
    # Calculate u and d using the specified formula (set 2)
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    
    R = math.exp(r*t)
    p = (R - d)/(u - d)
    
    if check_arbitrage(u, d, r, t):
        if display:
            print("Arbitrage Opportunity exists for M =", M)
        return 0, 0

    # Initialize option value arrays
    C = [[0 for i in range(M + 1)] for j in range(M + 1)]
    P = [[0 for i in range(M + 1)] for j in range(M + 1)]
    
    # Terminal values at expiration
    for i in range(0, M + 1):
        stock_price = S0*math.pow(u, M - i)*math.pow(d, i)
        C[M][i] = max(0, stock_price - K)
        P[M][i] = max(0, K - stock_price)

    # Backward recursion for American options
    for j in range(M - 1, -1, -1):
        for i in range(0, j + 1):
            stock_price = S0*math.pow(u, j - i)*math.pow(d, i)
            
            # Expected continuation value
            hold_value_call = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R
            hold_value_put = (p*P[j + 1][i] + (1 - p)*P[j + 1][i + 1]) / R
            
            # Early exercise value
            exercise_value_call = max(0, stock_price - K)
            exercise_value_put = max(0, K - stock_price)
            
            # American option takes maximum of holding or exercising
            C[j][i] = max(hold_value_call, exercise_value_call)
            P[j][i] = max(hold_value_put, exercise_value_put)

    if display:
        print("Price of American Call Option =", C[0][0])
        print("Price of American Put Option =", P[0][0])

    return C[0][0], P[0][0]

def plot_S0():
    S = np.linspace(20, 200, 100)
    call_option_prices = []
    put_option_prices = []
    
    for s in S:
        c, p = american_binomial_model(S0=s, K=100, T=1, M=100, r=0.08, sigma=0.20, display=False)
        call_option_prices.append(c)
        put_option_prices.append(p)
    
    plot_fixed(S, call_option_prices, "S0", "Price of American Call option", 
              "Initial American Call Option Price vs S0")
    plot_fixed(S, put_option_prices, "S0", "Price of American Put option", 
              "Initial American Put Option Price vs S0")

def plot_K():
    K = np.linspace(20, 200, 100)
    call_option_prices = []
    put_option_prices = []
    
    for k in K:
        c, p = american_binomial_model(S0=100, K=k, T=1, M=100, r=0.08, sigma=0.20, display=False)
        call_option_prices.append(c)
        put_option_prices.append(p)
    
    plot_fixed(K, call_option_prices, "K", "Price of American Call option", 
              "Initial American Call Option Price vs K")
    plot_fixed(K, put_option_prices, "K", "Price of American Put option", 
              "Initial American Put Option Price vs K")

def plot_r():
    r_list = np.linspace(0, 1, 100)
    call_option_prices = []
    put_option_prices = []
    
    for rate in r_list:
        c, p = american_binomial_model(S0=100, K=100, T=1, M=100, r=rate, sigma=0.20, display=False)
        call_option_prices.append(c)
        put_option_prices.append(p)
    
    plot_fixed(r_list, call_option_prices, "r", "Price of American Call option", 
              "Initial American Call Option Price vs r")
    plot_fixed(r_list, put_option_prices, "r", "Price of American Put option", 
              "Initial American Put Option Price vs r")

def plot_sigma():
    sigma_list = np.linspace(0.01, 1, 100)
    call_option_prices = []
    put_option_prices = []
    
    for sg in sigma_list:
        c, p = american_binomial_model(S0=100, K=100, T=1, M=100, r=0.08, sigma=sg, display=False)
        call_option_prices.append(c)
        put_option_prices.append(p)
    
    plot_fixed(sigma_list, call_option_prices, "sigma", "Price of American Call option", 
              "Initial American Call Option Price vs sigma")
    plot_fixed(sigma_list, put_option_prices, "sigma", "Price of American Put option", 
              "Initial American Put Option Price vs sigma")

def plot_M():
    M_list = [i for i in range(50, 200)]
    K_list = [95, 100, 105]
    
    for k in K_list:
        call_option_prices = []
        put_option_prices = []
        for m in M_list:
            c, p = american_binomial_model(S0=100, K=k, T=1, M=m, r=0.08, sigma=0.20, display=False)
            call_option_prices.append(c)
            put_option_prices.append(p)
        
        plot_fixed(M_list, call_option_prices, "M", "Price of American Call option", 
                  f"Initial American Call Option Price vs M for K = {k}")
        plot_fixed(M_list, put_option_prices, "M", "Price of American Put option", 
                  f"Initial American Put Option Price vs M for K = {k}")

def main():
    # Calculate base case prices
    american_binomial_model(S0=100, K=100, T=1, M=100, r=0.08, sigma=0.20)
    
    # Generate all sensitivity plots
    plot_S0()
    plot_K()
    plot_r()
    plot_sigma()
    plot_M()

if __name__ == "__main__":
    main()