# Momentum strategy main.py
# (Implement a pure momentum-based trading strategy here)

# Example skeleton:
import numpy as np

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    if t < 5:
        return np.zeros(n)
    prices = prcSoFar
    signal = np.zeros(n)
    for i in range(n):
        momentum = (prices[i, -1] - prices[i, -5]) / prices[i, -5]
        if momentum > 0.01:
            signal[i] = 1
        elif momentum < -0.01:
            signal[i] = -1
    return signal
