import numpy as np

nInst = 50
max_notional = 10_000  # Max $10K position per stock

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    if t < 21:
        return np.zeros(n)

    lookback = 20
    short_term = 5

    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    recent_returns = np.log(prices[:, -1] / prices[:, -short_term-1])
    vol = np.std(returns[:, -lookback:], axis=1) + 1e-6  # avoid div by 0

    # Cross-sectional momentum
    ranked = recent_returns.argsort()
    longs = ranked[-int(n * 0.2):]
    shorts = ranked[:int(n * 0.2)]
    cs_momentum = np.zeros(n)
    cs_momentum[longs] = 1
    cs_momentum[shorts] = -1

    # Correlation matrix
    corr_matrix = np.corrcoef(returns[:, -lookback:])
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] > 0.9:
                pairs.append((i, j))

    # Pair spread trading
    spread_signal = np.zeros(n)
    for i, j in pairs:
        spread = prices[i, -1] - prices[j, -1]
        spread_hist = prices[i, -lookback:] - prices[j, -lookback:]
        z = (spread - spread_hist.mean()) / (spread_hist.std() + 1e-6)
        if z > 1:
            spread_signal[i] -= 1
            spread_signal[j] += 1
        elif z < -1:
            spread_signal[i] += 1
            spread_signal[j] -= 1

    # Combine signals
    combined_signal = (0.5 * cs_momentum + 0.5 * spread_signal)

    # Position sizing: inverse-volatility weighted
    weights = combined_signal / vol
    weights[np.isnan(weights)] = 0
    weights /= np.sum(np.abs(weights)) + 1e-6  # Normalize to unit leverage

    # Convert weights to $ value and then to shares
    prices_now = prices[:, -1]
    dollar_positions = weights * 1_000_000
    share_positions = dollar_positions / prices_now

    # Enforce $10K position limit
    max_shares = max_notional / prices_now
    share_positions = np.clip(share_positions, -max_shares, max_shares)

    return np.round(share_positions)
