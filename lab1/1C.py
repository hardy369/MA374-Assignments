import numpy as np
import math
import matplotlib.pyplot as plt

time_points = [0, 0.30, 0.75, 1.50, 2.70]
S = 9
K = 10
T = 3
r = 0.06
sigma = 0.3


def arbitrage_condition(u, d, r, t):
    if d < math.exp(r * t) and math.exp(r * t) < u:
        return False
    else:
        return True


def check_time_stamp(t):
    epsilon = 1e-6
    for i in time_points:
        if abs(t - i) < epsilon: 
            return True
    return False

def binomial_model():
    call_option_prices = [[] for _ in time_points]
    put_option_prices = [[] for _ in time_points]
    M = 20
    t = T / M
    u = math.exp(sigma * math.sqrt(t) + (r - 0.5 * sigma * sigma) * t)
    d = math.exp(-sigma * math.sqrt(t) + (r - 0.5 * sigma * sigma) * t)

    R = math.exp(r * t)
    p = (R - d) / (u - d)

    result = arbitrage_condition(u, d, r, t)
    if result:
        print("Arbitrage Opportunity exists for M = {}".format(M))
        return call_option_prices, put_option_prices

    C = [[0 for i in range(M + 1)] for j in range(M + 1)]
    P = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
        C[M][i] = max(0, S * math.pow(u, M - i) * math.pow(d, i) - K)
        P[M][i] = max(0, K - S * math.pow(u, M - i) * math.pow(d, i))

    for j in range(M - 1, -1, -1):
        for i in range(0, j + 1):
            C[j][i] = (p * C[j + 1][i] + (1 - p) * C[j + 1][i + 1]) / R
            P[j][i] = (p * P[j + 1][i] + (1 - p) * P[j + 1][i + 1]) / R

    for i in range(0, M + 1):
        t = i * T / M
        if check_time_stamp(t):
            intermediate_call = []
            intermediate_put = []

            for j in range(0, i + 1):
                intermediate_call.append(C[i][j])
                intermediate_put.append(P[i][j])

            idx = time_points.index(round(t, 2))
            call_option_prices[idx] = intermediate_call
            put_option_prices[idx] = intermediate_put

    return call_option_prices, put_option_prices

def main():
    call_option_prices, put_option_prices = binomial_model()
    for idx in range(len(time_points)):
        print("t = {}".format(time_points[idx]))
        if call_option_prices[idx]:
            print("Call Option\tPut Option")
            for j in range(len(call_option_prices[idx])):
                print("{:.2f}\t\t{:.2f}".format(call_option_prices[idx][j], put_option_prices[idx][j]))
        else:
            print("No data available for this time point.")
        print()

if __name__ == "__main__":
    main()
