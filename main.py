import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional
prev_pos = np.zeros(nInst)
rolling_sharpe = np.zeros(nInst)
rolling_returns = np.zeros((nInst, 20))  # Store last 20 returns for each asset
trade_count = 0

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

# Ornstein-Uhlenbeck mean reversion speed
def ou_mean_reversion_speed(spread, dt=1/252):
    """Estimate mean reversion speed using Ornstein-Uhlenbeck process"""
    if len(spread) < 10:
        return 0.0
    
    spread_diff = np.diff(spread)
    spread_lag = spread[:-1]
    mean_spread = np.mean(spread_lag)
    
    # Linear regression: ds = alpha * (mu - s) * dt + noise
    y = spread_diff
    x = mean_spread - spread_lag
    
    if np.std(x) < 1e-8:
        return 0.0
    
    alpha = np.cov(x, y)[0, 1] / (np.var(x) * dt)
    return max(0, alpha)  # Ensure positive mean reversion

# Kelly criterion for position sizing
def kelly_position_size(expected_return, variance, max_leverage=2.0):
    """Calculate optimal position size using Kelly criterion"""
    if variance <= 0:
        return 0.0
    
    kelly_fraction = expected_return / variance
    # Cap at max leverage and ensure reasonable bounds
    return np.clip(kelly_fraction, -max_leverage, max_leverage)

# Multi-timeframe momentum
def multi_timeframe_momentum(prices):
    """Calculate momentum across multiple timeframes"""
    if len(prices) < 20:
        return 0.0
    
    # Short, medium, long term momentum
    mom_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
    mom_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
    mom_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
    
    # Weighted combination favoring recent momentum
    return 0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_20

# Enhanced volatility with GARCH-like features
def enhanced_volatility(returns, window=20):
    """Calculate volatility with persistence modeling"""
    if len(returns) < window:
        return 0.0
    
    recent_returns = returns[-window:]
    
    # Standard volatility
    vol = np.std(recent_returns)
    
    # Volatility persistence (GARCH-like)
    vol_squared = recent_returns ** 2
    persistence = np.corrcoef(vol_squared[:-1], vol_squared[1:])[0, 1]
    persistence = max(0, min(0.99, persistence))  # Keep in reasonable bounds
    
    # Adjust for persistence
    adjusted_vol = vol * (1 + 0.5 * persistence)
    
    return adjusted_vol * np.sqrt(252)

# Regime detection using rolling statistics
def detect_regime(returns, window=30):
    """Detect market regime based on return characteristics"""
    if len(returns) < window:
        return "normal"
    
    recent_returns = returns[-window:]
    vol = np.std(recent_returns)
    skewness = np.mean(((recent_returns - np.mean(recent_returns)) / vol) ** 3)
    kurtosis = np.mean(((recent_returns - np.mean(recent_returns)) / vol) ** 4)
    
    # Simple regime classification
    if vol > np.percentile(returns, 90):
        return "high_vol"
    elif kurtosis > 4:  # Fat tails
        return "crisis"
    elif abs(skewness) > 1:  # High skewness
        return "trending"
    else:
        return "normal"

# Enhanced mean reversion with multiple signals
def enhanced_mean_reversion_signal(spread, window=25):
    """Calculate multiple mean reversion signals"""
    if len(spread) < window:
        return 0.0, 0.8  # Default threshold
    
    # Standard z-score
    mean_spread = np.mean(spread[-window:])
    std_spread = np.std(spread[-window:]) + 1e-8
    z_score = (spread[-1] - mean_spread) / std_spread
    
    # OU mean reversion speed
    ou_speed = ou_mean_reversion_speed(spread[-window:])
    
    # Adaptive threshold based on mean reversion strength
    base_threshold = 0.8  # Lower threshold for more trading
    threshold = base_threshold * (1 + ou_speed * 2)  # Higher threshold if strong mean reversion
    threshold = min(1.5, max(0.5, threshold))  # Keep reasonable bounds
    
    return z_score, threshold

def getMyPosition(prcSoFar):
    global prev_pos, rolling_sharpe, rolling_returns, trade_count
    n, t = prcSoFar.shape
    if t < 60:  # Reduced from 80 for more frequent trading
        return np.zeros(nInst)

    lookback = 50  # Reduced for more responsive trading
    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    R = returns[:, -lookback:]
    R -= np.mean(R, axis=1, keepdims=True)

    # Update rolling returns for regime detection
    if t >= 20:
        rolling_returns[:, :-1] = rolling_returns[:, 1:]
        rolling_returns[:, -1] = returns[:, -1]

    # PCA Decomposition with more components for richer representation
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, :10]  # Use even more PCs

    # Enhanced similarity matrix
    norm_loadings = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-8)
    similarity = norm_loadings @ norm_loadings.T
    np.fill_diagonal(similarity, 0)

    # More aggressive threshold for frequent trading
    sim_threshold = max(0.75, np.percentile(similarity, 90))  # Lower threshold
    
    # Enhanced pair selection with multiple criteria
    pairs = []
    for i in range(nInst):
        for j in range(i+1, nInst):
            if similarity[i, j] > sim_threshold:
                # Correlation check
                corr = np.corrcoef(prices[i, -lookback:], prices[j, -lookback:])[0, 1]
                if 0.6 < corr < 0.95:  # Wider range for more pairs
                    # Volatility similarity check
                    vol_i = enhanced_volatility(returns[i, -30:])
                    vol_j = enhanced_volatility(returns[j, -30:])
                    vol_ratio = min(vol_i, vol_j) / (max(vol_i, vol_j) + 1e-8)
                    if vol_ratio > 0.5:  # Similar volatility assets
                        pairs.append((i, j))
    
    if not pairs:
        return np.zeros(nInst)

    # Enhanced volatility analysis
    vols = np.array([enhanced_volatility(returns[i, -30:]) for i in range(nInst)])
    avg_vol = np.nanmean(vols)
    
    # Detect market regime
    avg_returns = np.mean(returns[:, -30:], axis=0)
    regime = detect_regime(avg_returns)
    
    # More permissive volatility filter for frequent trading
    if np.isnan(avg_vol) or avg_vol > 1.0:  # Much higher threshold
        return np.zeros(nInst)

    # Advanced signal generation with multiple quantitative features
    signal = np.zeros(nInst)
    signal_strength = np.zeros(nInst)
    expected_returns = np.zeros(nInst)
    variances = np.zeros(nInst)
    
    for i, j in pairs[:25]:  # Consider more pairs for active trading
        # Calculate spread and its properties
        spread_hist = prices[i, -lookback:] - prices[j, -lookback:]
        
        # Enhanced mean reversion signal
        z_score, threshold = enhanced_mean_reversion_signal(spread_hist, window=30)
        
        # Multi-timeframe momentum for both assets
        mom_i = multi_timeframe_momentum(prices[i, -40:])
        mom_j = multi_timeframe_momentum(prices[j, -40:])
        
        # Enhanced volatility estimates
        vol_i = enhanced_volatility(returns[i, -25:])
        vol_j = enhanced_volatility(returns[j, -25:])
        vol_ratio = min(vol_i, vol_j) / (max(vol_i, vol_j) + 1e-8)
        
        # Mean reversion speed
        ou_speed = ou_mean_reversion_speed(spread_hist)
        
        # Expected return estimation for Kelly sizing
        spread_returns = np.diff(spread_hist) / (np.abs(spread_hist[:-1]) + 1e-8)
        expected_return_spread = np.mean(spread_returns[-10:]) if len(spread_returns) >= 10 else 0
        
        # Enhanced signal logic with multiple confirmations
        if abs(z_score) > threshold:
            # Base signal strength
            base_signal = min(3.0, abs(z_score) / threshold)  # Scale with z-score magnitude
            
            # Momentum confirmation (diverging momentum is good for mean reversion)
            momentum_divergence = abs(mom_i + mom_j) / (abs(mom_i) + abs(mom_j) + 1e-8)
            momentum_boost = 1 + momentum_divergence
            
            # OU speed boost (faster mean reversion = stronger signal)
            ou_boost = 1 + min(1.0, ou_speed * 5)
            
            # Volatility preference (similar vol assets)
            vol_boost = 0.5 + 0.5 * vol_ratio
            
            # Regime-based adjustment
            regime_multiplier = {
                "normal": 1.2,
                "trending": 1.5,  # Good for mean reversion
                "high_vol": 0.8,
                "crisis": 0.6
            }.get(regime, 1.0)
            
            # Combined signal
            final_signal = base_signal * momentum_boost * ou_boost * vol_boost * regime_multiplier
            
            # Expected return and variance for Kelly criterion
            expected_ret = expected_return_spread * final_signal
            variance = max(1e-8, np.var(spread_returns[-10:]) if len(spread_returns) >= 10 else 0.01)
            
            if z_score > threshold:
                signal[i] -= final_signal
                signal[j] += final_signal
                expected_returns[i] = -expected_ret
                expected_returns[j] = expected_ret
                variances[i] = variance
                variances[j] = variance
                signal_strength[i] += final_signal
                signal_strength[j] += final_signal
            elif z_score < -threshold:
                signal[i] += final_signal
                signal[j] -= final_signal
                expected_returns[i] = expected_ret
                expected_returns[j] = -expected_ret
                variances[i] = variance
                variances[j] = variance
                signal_strength[i] += final_signal
                signal_strength[j] += final_signal

    # More permissive signal threshold for frequent trading
    if np.sum(np.abs(signal)) < 0.05:  # Lower threshold
        return np.zeros(nInst)

    trade_count += 1

    # Kelly criterion position sizing with risk management
    total_signal = np.sum(np.abs(signal))
    if total_signal == 0:
        return np.zeros(nInst)
    
    # Base weights from signals
    weights = signal / (total_signal + 1e-6)
    
    # Kelly criterion scaling
    kelly_weights = np.zeros(nInst)
    for i in range(nInst):
        if abs(weights[i]) > 1e-6:
            kelly_size = kelly_position_size(expected_returns[i], variances[i], max_leverage=1.5)
            kelly_weights[i] = weights[i] * abs(kelly_size)
    
    # Normalize Kelly weights
    kelly_total = np.sum(np.abs(kelly_weights))
    if kelly_total > 0:
        kelly_weights = kelly_weights / kelly_total
    else:
        kelly_weights = weights
    
    # Multi-factor capital scaling
    # 1. Volatility scaling (more aggressive)
    vol_scale = min(2.0, 0.5 / (avg_vol + 1e-6))  # Higher scaling
    
    # 2. Signal confidence scaling
    confidence = np.mean(signal_strength[signal_strength > 0]) if np.any(signal_strength > 0) else 0
    confidence_scale = 0.7 + 0.6 * min(1.0, confidence / 2.0)  # More aggressive
    
    # 3. Regime-based scaling
    regime_scale = {
        "normal": 1.3,
        "trending": 1.5,
        "high_vol": 0.9,
        "crisis": 0.7
    }.get(regime, 1.0)
    
    # 4. Trading frequency boost
    frequency_boost = min(1.5, 1.0 + trade_count * 0.001)  # Slight boost for active trading
    
    # Combined scaling (more aggressive overall)
    total_scale = vol_scale * confidence_scale * regime_scale * frequency_boost
    total_scale = min(3.0, total_scale)  # Cap at 3x leverage
    
    dollar_alloc = kelly_weights * total_capital * total_scale
    
    prices_now = prices[:, -1]
    position = dollar_alloc / prices_now

    # Dynamic position limits based on volatility and performance
    max_shares = max_notional / prices_now
    
    # Volatility-adjusted limits (more permissive)
    vol_adjusted_max = max_shares * (0.8 / (vols + 1e-6))
    vol_adjusted_max = np.clip(vol_adjusted_max, max_shares * 0.2, max_shares * 2.0)
    
    position = np.clip(position, -vol_adjusted_max, vol_adjusted_max)
    
    # Performance-based position adjustments
    for i in range(nInst):
        if abs(position[i]) > 0:
            recent_returns = returns[i, -15:]  # Shorter window for responsiveness
            if len(recent_returns) >= 10:
                sharpe_proxy = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
                rolling_sharpe[i] = 0.8 * rolling_sharpe[i] + 0.2 * sharpe_proxy
                
                # Less aggressive reduction for poor performance
                if rolling_sharpe[i] < -0.8:
                    position[i] *= 0.7
                elif rolling_sharpe[i] > 0.3:  # Boost good performers
                    position[i] *= 1.2

    # Less aggressive smoothing for more responsive trading
    alpha = 0.7 if regime == "high_vol" else 0.5  # Less smoothing overall
    position = alpha * position + (1 - alpha) * prev_pos
    prev_pos = position

    return np.round(position)
