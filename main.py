import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional
prev_pos = np.zeros(nInst)
rolling_sharpe = np.zeros(nInst)

# Black-Scholes inspired volatility estimation
def estimate_volatility(prices, window=20):
    """Estimate annualized volatility using log returns"""
    if len(prices) < 2:
        return 0.0
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns[-window:]) * np.sqrt(252)

# Time series momentum calculation
def calculate_momentum(prices, short_window=5, long_window=20):
    """Calculate price momentum"""
    if len(prices) < long_window:
        return 0.0
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])
    return (short_ma - long_ma) / long_ma

# Mean reversion signal
def mean_reversion_signal(spread, window=30):
    """Calculate mean reversion z-score with adaptive threshold"""
    if len(spread) < window:
        return 0.0, 0.0
    mean_spread = np.mean(spread[-window:])
    std_spread = np.std(spread[-window:]) + 1e-8
    z_score = (spread[-1] - mean_spread) / std_spread
    # Adaptive threshold based on recent volatility
    threshold = min(2.0, max(1.0, std_spread / np.mean(np.abs(spread[-window:])) * 10))
    return z_score, threshold

def getMyPosition(prcSoFar):
    global prev_pos, rolling_sharpe
    n, t = prcSoFar.shape
    if t < 80:  # Need more data for advanced features
        return np.zeros(nInst)

    lookback = 60
    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    R = returns[:, -lookback:]
    R -= np.mean(R, axis=1, keepdims=True)

    # PCA Decomposition on de-meaned returns
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, :7]  # Use more PCs for better coverage

    # Compute similarity matrix with enhanced cosine similarity
    norm_loadings = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-8)
    similarity = norm_loadings @ norm_loadings.T
    np.fill_diagonal(similarity, 0)

    # Dynamic threshold based on distribution
    sim_threshold = max(0.85, np.percentile(similarity, 95))
    
    # Enhanced pair selection with correlation filtering
    pairs = []
    for i in range(nInst):
        for j in range(i+1, nInst):
            if similarity[i, j] > sim_threshold:
                # Additional correlation check
                corr = np.corrcoef(prices[i, -lookback:], prices[j, -lookback:])[0, 1]
                if 0.7 < corr < 0.98:  # Avoid perfect correlation
                    pairs.append((i, j))
    
    if not pairs:
        return np.zeros(nInst)

    # Black-Scholes inspired volatility filter
    vols = np.array([estimate_volatility(prices[i, -30:]) for i in range(nInst)])
    avg_vol = np.nanmean(vols)
    vol_regime = "high" if avg_vol > 0.4 else "medium" if avg_vol > 0.25 else "low"
    
    # Skip in extreme volatility
    if np.isnan(avg_vol) or avg_vol > 0.6:
        return np.zeros(nInst)

    # Advanced signal generation with time series features
    signal = np.zeros(nInst)
    signal_strength = np.zeros(nInst)
    
    for i, j in pairs[:20]:  # Consider top 20 pairs
        # Calculate spread and its properties
        spread_hist = prices[i, -lookback:] - prices[j, -lookback:]
        
        # Mean reversion signal
        z_score, threshold = mean_reversion_signal(spread_hist, window=40)
        
        # Momentum signals for both assets
        mom_i = calculate_momentum(prices[i, -30:])
        mom_j = calculate_momentum(prices[j, -30:])
        
        # Volatility-adjusted signal strength
        vol_i = estimate_volatility(prices[i, -20:])
        vol_j = estimate_volatility(prices[j, -20:])
        vol_ratio = min(vol_i, vol_j) / (max(vol_i, vol_j) + 1e-8)
        
        # Enhanced signal logic
        if abs(z_score) > threshold:
            base_signal = 1.0
            
            # Momentum confirmation
            momentum_confirm = (mom_i * mom_j < 0)  # Diverging momentum
            if momentum_confirm:
                base_signal *= 1.5
            
            # Volatility adjustment
            base_signal *= vol_ratio  # Prefer pairs with similar volatility
            
            # Regime adjustment
            if vol_regime == "high":
                base_signal *= 0.5
            elif vol_regime == "low":
                base_signal *= 1.2
            
            if z_score > threshold:
                signal[i] -= base_signal
                signal[j] += base_signal
                signal_strength[i] += base_signal
                signal_strength[j] += base_signal
            elif z_score < -threshold:
                signal[i] += base_signal
                signal[j] -= base_signal
                signal_strength[i] += base_signal
                signal_strength[j] += base_signal

    # Skip if no significant signals
    if np.sum(np.abs(signal)) < 0.1:
        return np.zeros(nInst)

    # Advanced capital allocation with risk management
    total_signal = np.sum(np.abs(signal))
    if total_signal == 0:
        return np.zeros(nInst)
    
    # Normalize signals
    weights = signal / (total_signal + 1e-6)
    
    # Dynamic capital scaling based on multiple factors
    # 1. Volatility scaling
    vol_scale = min(1.0, 0.3 / (avg_vol + 1e-6))
    
    # 2. Signal confidence scaling
    confidence = np.mean(signal_strength) / (np.max(signal_strength) + 1e-6)
    confidence_scale = 0.5 + 0.5 * confidence
    
    # 3. Market regime scaling
    regime_scale = {"low": 1.2, "medium": 1.0, "high": 0.6}[vol_regime]
    
    # Combined scaling
    total_scale = vol_scale * confidence_scale * regime_scale
    dollar_alloc = weights * total_capital * total_scale
    
    prices_now = prices[:, -1]
    position = dollar_alloc / prices_now

    # Enforce position limits with volatility adjustment
    max_shares = max_notional / prices_now
    vol_adjusted_max = max_shares * (0.3 / (vols + 1e-6))  # Reduce size for high vol assets
    position = np.clip(position, -vol_adjusted_max, vol_adjusted_max)
    
    # Risk management: Position sizing based on recent performance
    for i in range(nInst):
        if abs(position[i]) > 0:
            recent_returns = returns[i, -10:]
            sharpe_proxy = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
            rolling_sharpe[i] = 0.9 * rolling_sharpe[i] + 0.1 * sharpe_proxy
            
            # Reduce position if poor recent performance
            if rolling_sharpe[i] < -0.5:
                position[i] *= 0.5

    # Enhanced position smoothing with volatility consideration
    alpha = 0.6 if vol_regime == "high" else 0.4  # More smoothing in high vol
    position = alpha * position + (1 - alpha) * prev_pos
    prev_pos = position

    return np.round(position)
