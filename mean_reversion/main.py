# Mean reversion strategy main.py
# (Implement a pure mean reversion trading strategy here)

import numpy as np

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    if t < 10:
        return np.zeros(n)
    prices = prcSoFar
    signal = np.zeros(n)
    for i in range(n):
        mean = np.mean(prices[i, -10:])
        std = np.std(prices[i, -10:]) + 1e-8
        z = (prices[i, -1] - mean) / std
        if z > 1:
            signal[i] = -1  # Short if price is high
        elif z < -1:
            signal[i] = 1   # Long if price is low
    return signal
