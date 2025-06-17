import numpy as np
import itertools

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    if t < 21:
        return np.zeros(n)

    lookback = 20
    returns = np.log(prcSoFar[:, -lookback:] / prcSoFar[:, -lookback-1:-1])

    # Compute correlation matrix
    corr = np.corrcoef(returns)

    # Find best correlated pair
    best_pair = None
    best_corr = 0
    for i, j in itertools.combinations(range(n), 2):
        if corr[i, j] > best_corr and i != j:
            best_corr = corr[i, j]
            best_pair = (i, j)

    if not best_pair:
        return np.zeros(n)

    i, j = best_pair
    spread = prcSoFar[i, -1] - prcSoFar[j, -1]
    spread_hist = prcSoFar[i, -lookback:] - prcSoFar[j, -lookback:]
    spread_mean = spread_hist.mean()
    spread_std = spread_hist.std()
    z = (spread - spread_mean) / (spread_std + 1e-6)

    # Mean reversion: if spread is wide, short the wider leg, long the narrower
    pos = np.zeros(n)
    if z > 1:
        pos[i] = -1000
        pos[j] = 1000
    elif z < -1:
        pos[i] = 1000
        pos[j] = -1000

    return pos
