import math
import numpy as np
import matplotlib.pyplot as plt

S0 = 9         
K = 10          
T = 3          
r = 0.06        
sigma = 0.3    
M_values = [1, 5, 10, 20, 50, 100, 200] 

def binomial_price(S0, K, T, r, sigma, M):
    dt = T / M  
    u = math.exp(sigma * math.sqrt(dt) + (r - 0.5 * sigma ** 2) * dt)
    d = math.exp(-sigma * math.sqrt(dt) + (r - 0.5 * sigma ** 2) * dt)
    
    # Check for no-arbitrage condition
    p = (math.exp(r * dt) - d) / (u - d)
    if p<0 or p>1:
     print("Arbitrage exists")
    
    # Initialize asset prices at final time step (T)
    ST = np.zeros(M + 1)
    for i in range(M + 1):
        ST[i] = S0 * u ** (M - i) * d ** i
 
    C = np.maximum(ST - K, 0)  
    P = np.maximum(K - ST, 0)  
    
    for t in range(M - 1, -1, -1):
        for i in range(t + 1):
            C[i] = math.exp(-r * dt) * (p * C[i] + (1 - p) * C[i + 1])
            P[i] = math.exp(-r * dt) * (p * P[i] + (1 - p) * P[i + 1])
    #current price of the option
    return C[0], P[0]

def bpa():
    call_prices = []
    put_prices = []
    
    for M in M_values:
        call_price, put_price = binomial_price(S0, K, T, r, sigma, M)
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    return call_prices, put_prices

def plot_prices(call_prices, put_prices, M_values):
    plt.plot(M_values, call_prices, label="Call Option")
    plt.plot(M_values, put_prices, label="Put Option")
    plt.xlabel("M")
    plt.ylabel("Option Price")
    plt.legend()
    plt.show()

def tabulate_prices(call_prices, put_prices, M_values):
    print("M",            "Call Price",        "Put Price")
    for M, call, put in zip(M_values, call_prices, put_prices):
        print(f"{M} \t {call:.4f} \t {put:.4f}")

call_prices, put_prices = bpa()
tabulate_prices(call_prices, put_prices, M_values)
plot_prices(call_prices, put_prices, M_values)



