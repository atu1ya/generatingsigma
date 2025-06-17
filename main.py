import numpy as np

nInst = 50
max_notional = 10_000
prev_pos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global prev_pos
    n, t = prcSoFar.shape
    if t < 60:
        return np.zeros(nInst)

    lookback = 50
    recent_window = 5
    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])

    # --- PCA: Extract 1st principal component as market trend ---
    R = returns[:, -lookback:]
    R -= R.mean(axis=1, keepdims=True)
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    pc1 = eigvecs[:, idx[0]]
    market_factor = pc1 @ R  # shape: [T]

    # If market is trending (PC1 strong), skip mean-reversion
    trend_strength = np.std(market_factor)
    if trend_strength > 0.04:
        return np.zeros(nInst)  # stay out when trend dominates

    # --- Compute residual alpha ---
    k = 3
    factors = eigvecs[:, idx[:k]].T @ R
    betas = np.zeros((nInst, k))
    for i in range(nInst):
        X = factors.T
        y = R[i]
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        betas[i] = beta

    # Predicted returns
    latest_factors = factors[:, -1]
    expected_ret = betas @ latest_factors
    actual_ret = np.log(prices[:, -1] / prices[:, -1 - recent_window])
    alpha = actual_ret - expected_ret

    # --- Residual filtering ---
    vol = np.std(returns[:, -lookback:], axis=1) + 1e-6
    confidence = 1 / vol
    z = (alpha - np.mean(alpha)) / (np.std(alpha) + 1e-6)
    signal = z * confidence

    # --- Trade only strongest alphas ---
    threshold = np.percentile(np.abs(signal), 80)
    filtered_signal = np.where(np.abs(signal) >= threshold, signal, 0)

    # Normalize to use capital based on number of trades
    num_active = np.count_nonzero(filtered_signal)
    if num_active == 0:
        return np.zeros(nInst)

    weights = filtered_signal
    weights /= np.sum(np.abs(weights)) + 1e-6

    # Dollar allocation
    total_cap = num_active * max_notional  # adaptive capital use
    dollar_alloc = weights * total_cap
    prices_now = prices[:, -1]
    pos = dollar_alloc / prices_now

    # Cap at $10k
    max_shares = max_notional / prices_now
    pos = np.clip(pos, -max_shares, max_shares)

    # Smooth transitions
    alpha = 0.3
    pos = alpha * pos + (1 - alpha) * prev_pos
    prev_pos = pos

    return np.round(pos)
