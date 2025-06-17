import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional

# Memory for position smoothing
previous_position = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global previous_position
    n, t = prcSoFar.shape
    if t < 21:
        return np.zeros(n)

    lookback = 20
    recent_window = 5

    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])

    # 1. Signal: short-term return vs long-term mean return
    short_ret = np.log(prices[:, -1] / prices[:, -recent_window-1])
    long_ret = np.mean(returns[:, -lookback:], axis=1)
    signal = short_ret - long_ret  # Mean-reversion signal

    # 2. Rank to get stable, relative scores (percentile-based)
    rank = signal.argsort().argsort()
    scores = 2 * (rank / (n - 1)) - 1  # Range [-1, 1]

    # 3. Volatility targeting
    vol = np.std(returns[:, -lookback:], axis=1) + 1e-6
    adj_scores = scores / vol
    adj_scores /= np.sum(np.abs(adj_scores)) + 1e-6  # unit leverage

    # 4. Scale to total capital
    prices_now = prices[:, -1]
    dollar_alloc = adj_scores * total_capital
    position = dollar_alloc / prices_now

    # 5. Enforce $10k position cap
    max_shares = max_notional / prices_now
    position = np.clip(position, -max_shares, max_shares)

    # 6. Smooth positions to reduce turnover
    alpha = 0.5  # smoothing weight
    smoothed_pos = alpha * position + (1 - alpha) * previous_position
    previous_position = smoothed_pos.copy()

    return np.round(smoothed_pos)
