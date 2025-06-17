import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional  # $500k

# Memory to smooth transitions
previous_position = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global previous_position
    n, t = prcSoFar.shape
    if t < 30:
        return np.zeros(n)

    lookback = 15
    smooth = 5

    # Log returns (recent and average)
    returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    recent_ret = np.mean(returns[:, -smooth:], axis=1)
    long_term_avg = np.mean(returns[:, -lookback:], axis=1)

    # Signal = mean reversion
    signal = long_term_avg - recent_ret

    # Rank signals (robust to outliers)
    ranks = signal.argsort().argsort()
    norm_signal = ranks - np.mean(ranks)  # mean zero
    norm_signal /= np.sum(np.abs(norm_signal)) + 1e-6  # scale to 1

    # Light volatility adjustment
    vol = np.std(returns[:, -lookback:], axis=1) + 1e-6
    vol_scale = 1 / vol
    vol_scale /= np.max(vol_scale)
    weights = norm_signal * vol_scale

    # Dollar allocation
    prices_now = prcSoFar[:, -1]
    dollar_alloc = weights * total_capital
    position = dollar_alloc / prices_now

    # Clip to $10k limit
    max_shares = max_notional / prices_now
    position = np.clip(position, -max_shares, max_shares)

    # Smooth transition (reduce turnover)
    alpha = 0.4  # smoothing factor (0.3–0.5 is good)
    position = alpha * position + (1 - alpha) * previous_position
    previous_position = position  # update memory

    return np.round(position)
