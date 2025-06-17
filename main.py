import numpy as np

nInst = 50
max_notional = 10_000

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    if t < 21:
        return np.zeros(n)

    lookback = 20
    short_term = 5

    # Compute short-term return
    short_returns = np.log(prcSoFar[:, -1] / prcSoFar[:, -short_term-1])
    mean_ret = np.mean(short_returns)
    std_ret = np.std(short_returns) + 1e-6
    z_scores = (short_returns - mean_ret) / std_ret

    # Mean reversion: short high z, long low z
    signal = -z_scores

    # Scale using inverse variance (from historical returns)
    ret_hist = np.log(prcSoFar[:, -lookback:] / prcSoFar[:, -lookback-1:-1])
    vol = np.std(ret_hist, axis=1) + 1e-6
    inv_vol = 1 / vol
    weights = signal * inv_vol

    # Normalise weights
    weights -= np.mean(weights)  # make it dollar-neutral
    weights /= np.sum(np.abs(weights)) + 1e-6

    # Allocate $1M capital
    dollar_alloc = weights * 1_000_000
    prices_now = prcSoFar[:, -1]
    position = dollar_alloc / prices_now

    # Enforce position cap
    max_shares = max_notional / prices_now
    position = np.clip(position, -max_shares, max_shares)

    return np.round(position)
