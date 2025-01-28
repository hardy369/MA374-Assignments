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

def binomial_model(S0, K, T, M, r, sigma, set, display):
  u, d = 0, 0
  t = T/M

  if set == 1:
    u = math.exp(sigma*math.sqrt(t))
    d = math.exp(-sigma*math.sqrt(t)) 
  else:
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

  R = math.exp(r*t)
  p = (R - d)/(u - d);
  result = check_arbitrage(u, d, r, t)

  if result:
    if display == 1:
      print("Arbitrage Opportunity exists for M = {}".format(M))
    return 0, 0


  C = [[0 for i in range(M + 1)] for j in range(M + 1)]
  P = [[0 for i in range(M + 1)] for j in range(M + 1)]

  for i in range(0, M + 1):
    C[M][i] = max(0, S0*math.pow(u, M - i)*math.pow(d, i) - K)
    P[M][i] = max(0, K - S0*math.pow(u, M - i)*math.pow(d, i))

  for j in range(M - 1, -1, -1):
    for i in range(0, j + 1):
      C[j][i] = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R;
      P[j][i] = (p*P[j + 1][i] + (1 - p)*P[j + 1][i + 1]) / R;

  
  if display == 1: 
    print("For Set = {}".format(set))
    print("Price of Call Option = {}".format(C[0][0]))
    print("Price of Put Option = {}\n\n".format(P[0][0]))

  return C[0][0], P[0][0]


def plot_S0():
  S =  np.linspace(20, 200, 100)

  for set_num in range(1, 3):
    call_option_prices = []
    put_option_prices = []
    for s in S:
      c, p = binomial_model(S0 = s, K = 100, T = 1, M = 100, r = 0.08, sigma = 0.20, set = set_num, display = 0)
      call_option_prices.append(c)
      put_option_prices.append(p)

    plot_fixed(S, call_option_prices, "S0", "Prices of Call option", "Initial Call Option Price vs S0 for the set = " + str(set_num))
    plot_fixed(S, put_option_prices, "S0", "Prices of Put option", "Initial Put Option Price vs S0 for the set = " + str(set_num))


def plot_K():
  K =  np.linspace(20, 200, 100)

  for set_num in range(1, 3):
    call_option_prices = []
    put_option_prices = []
    for k in K:
      c, p = binomial_model(S0 = 100, K = k, T = 1, M = 100, r = 0.08, sigma = 0.20, set = set_num, display = 0)
      call_option_prices.append(c)
      put_option_prices.append(p)

    plot_fixed(K, call_option_prices, "K", "Prices of Call option ", "Initial Call Option Price vs K for the set = " + str(set_num))
    plot_fixed(K, put_option_prices, "K", "Prices of Put option ", "Initial Put Option Price vs K for the set = " + str(set_num))


def plot_r():
  r_list =  np.linspace(0, 1, 100)

  for set_num in range(1, 3):
    call_option_prices = []
    put_option_prices = []
    for rate in r_list:
      c, p = binomial_model(S0 = 100, K = 100, T = 1, M = 100, r = rate, sigma = 0.20, set = set_num, display = 0)
      call_option_prices.append(c)
      put_option_prices.append(p)

    plot_fixed(r_list, call_option_prices, "r", "Prices of Call option ", "Initial Call Option Price vs r for the set = " + str(set_num))
    plot_fixed(r_list, put_option_prices, "r", "Prices of Put option ", "Initial Put Option Price vs r for the set = " + str(set_num))


def plot_sigma():
  sigma_list =  np.linspace(0.01, 1, 100)

  for set_num in range(1, 3):
    call_option_prices = []
    put_option_prices = []
    for sg in sigma_list:
      c, p = binomial_model(S0 = 100, K = 100, T = 1, M = 100, r = 0.08, sigma = sg, set = set_num, display = 0)
      call_option_prices.append(c)
      put_option_prices.append(p)

    plot_fixed(sigma_list, call_option_prices, "sigma", "Prices of Call option", "Initial Call Option Price vs sigma for the set = " + str(set_num))
    plot_fixed(sigma_list, put_option_prices, "sigma", "Prices of Put option ", "Initial Put Option Price vs sigma for the set = " + str(set_num))


def plot_M():
  M_list =  [i for i in range(50, 200)]
  K_list = [95, 100, 105]

  for k in K_list:
    for set_num in range(1, 3):
      call_option_prices = []
      put_option_prices = []
      for m in M_list:
        c, p = binomial_model(S0 = 100, K = k, T = 1, M = m, r = 0.08, sigma = 0.20, set = set_num, display = 0)
        call_option_prices.append(c)
        put_option_prices.append(p)

      plot_fixed(M_list, call_option_prices, "M", "Price of Call option ", "Initial Call Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))
      plot_fixed(M_list, put_option_prices, "M", "Prices of Put option ", "Initial Put Option Price vs M for the set = " + str(set_num) + " and K = " + str(k))


def main():
  binomial_model(S0 = 100, K = 100, T = 1, M = 100, r = 0.08, sigma = 0.20, set = 1, display = 1)
  binomial_model(S0 = 100, K = 100, T = 1, M = 100, r = 0.08, sigma = 0.20, set = 2, display = 1)
  plot_S0()
  plot_K()
  plot_r()
  plot_sigma()
  plot_M()


if __name__=="__main__":
  main()

