import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional

prev_pos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global prev_pos
    n, t = prcSoFar.shape
    if t < 60:
        return np.zeros(n)

    lookback = 50
    ret_window = 1  # use daily returns

    prices = prcSoFar
    log_returns = np.log(prices[:, 1:] / prices[:, :-1])

    # STEP 1: Extract principal components (factors) using PCA
    R = log_returns[:, -lookback:]  # shape: [nInst, T]
    R -= R.mean(axis=1, keepdims=True)  # demean
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    k = 5  # number of PCA factors
    factors = eigvecs[:, :k].T @ R  # shape: [k, T]

    # STEP 2: Rolling regression: get each asset’s beta to each factor
    betas = np.zeros((nInst, k))
    for i in range(nInst):
        y = R[i]
        X = factors.T
        beta_i, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        betas[i] = beta_i

    # STEP 3: Predict expected returns from factor exposure
    latest_factors = factors[:, -1]  # shape: [k]
    expected_ret = betas @ latest_factors  # shape: [nInst]

    # STEP 4: Get actual latest return
    recent_ret = np.log(prices[:, -1] / prices[:, -1 - ret_window])

    # STEP 5: Compute alpha (residuals)
    alpha = recent_ret - expected_ret

    # STEP 6: Cross-sectional z-score of alpha
    alpha_z = (alpha - alpha.mean()) / (alpha.std() + 1e-6)

    # STEP 7: Inverse-vol weighting
    vol = np.std(log_returns[:, -lookback:], axis=1) + 1e-6
    weights = alpha_z / vol
    weights -= weights.mean()  # dollar-neutral
    weights /= np.sum(np.abs(weights)) + 1e-6  # use all capital

    # STEP 8: Capital allocation
    dollar_alloc = weights * total_capital
    prices_now = prices[:, -1]
    position = dollar_alloc / prices_now

    # Enforce per-stock $10k cap
    max_shares = max_notional / prices_now
    position = np.clip(position, -max_shares, max_shares)

    # STEP 9: Smooth position changes
    alpha = 0.4
    position = alpha * position + (1 - alpha) * prev_pos
    prev_pos = position

    return np.round(position)
