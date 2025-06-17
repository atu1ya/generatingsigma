import numpy as np

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional
prev_pos = np.zeros(nInst)
rolling_sharpe = np.zeros(nInst)
rolling_returns = np.zeros((nInst, 20))
trade_count = 0
recent_pnl = np.zeros(10)  # Track recent P&L for drawdown detection
volatility_regime = "normal"  # Track volatility regime

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

# Volatility circuit breaker
def check_volatility_regime(returns, window=20):
    """Detect high volatility periods and circuit breaker conditions"""
    if len(returns) < window:
        return "normal", 1.0
    
    recent_vol = np.std(returns[-window:]) * np.sqrt(252)
    very_recent_vol = np.std(returns[-5:]) * np.sqrt(252) if len(returns) >= 5 else recent_vol
    
    # Define volatility regimes
    if very_recent_vol > 0.8 or recent_vol > 0.6:
        return "crisis", 0.1  # Pull out 90% of money
    elif very_recent_vol > 0.5 or recent_vol > 0.4:
        return "high_vol", 0.3  # Reduce to 30% allocation
    elif very_recent_vol > 0.3 or recent_vol > 0.25:
        return "elevated", 0.6  # Reduce to 60% allocation
    else:
        return "normal", 1.0  # Full allocation

# Bad momentum detection and shorting
def detect_bad_momentum(prices, returns, window=15):
    """Detect when momentum turns bad and suggest shorting"""
    if len(prices) < window or len(returns) < window:
        return 0.0
    
    # Multiple momentum indicators
    price_mom = (prices[-1] - prices[-window]) / prices[-window]
    recent_returns = np.mean(returns[-5:]) if len(returns) >= 5 else 0
    volatility_adjusted_mom = price_mom / (np.std(returns[-window:]) + 1e-8)
    
    # Check for momentum deterioration
    if price_mom < -0.02 and recent_returns < -0.005:  # Both negative
        bad_momentum_signal = min(3.0, abs(price_mom) * 50 + abs(recent_returns) * 100)
        return -bad_momentum_signal  # Negative for shorting
    elif price_mom > 0.02 and recent_returns > 0.005:  # Both positive
        good_momentum_signal = min(2.0, price_mom * 30 + recent_returns * 60)
        return good_momentum_signal  # Positive for longing
    
    return 0.0

# Drawdown protection
def calculate_drawdown_factor(recent_pnl):
    """Calculate position reduction factor based on recent drawdowns"""
    if len(recent_pnl) < 5:
        return 1.0
    
    cumulative_pnl = np.cumsum(recent_pnl)
    max_pnl = np.maximum.accumulate(cumulative_pnl)
    drawdown = (cumulative_pnl - max_pnl) / (np.abs(max_pnl) + 1)
    current_drawdown = drawdown[-1]
    
    # Reduce positions during significant drawdowns
    if current_drawdown < -0.15:  # 15% drawdown
        return 0.2  # Reduce to 20%
    elif current_drawdown < -0.10:  # 10% drawdown
        return 0.4  # Reduce to 40%
    elif current_drawdown < -0.05:  # 5% drawdown
        return 0.7  # Reduce to 70%
    else:
        return 1.0  # Full positions

def getMyPosition(prcSoFar):
    global prev_pos, rolling_sharpe, rolling_returns, trade_count, recent_pnl, volatility_regime
    n, t = prcSoFar.shape
    if t < 40:
        return np.zeros(nInst)

    lookback = 40
    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    R = returns[:, -lookback:]
    R -= np.mean(R, axis=1, keepdims=True)

    # Update rolling returns and recent PnL tracking
    if t >= 20:
        rolling_returns[:, :-1] = rolling_returns[:, 1:]
        rolling_returns[:, -1] = returns[:, -1]
        
        # Estimate today's PnL for drawdown tracking
        if t > 1:
            price_change = (prices[:, -1] - prices[:, -2]) / prices[:, -2]
            estimated_pnl = np.sum(prev_pos * price_change * prices[:, -2])
            recent_pnl[:-1] = recent_pnl[1:]
            recent_pnl[-1] = estimated_pnl

    # Check volatility regime and get allocation factor
    avg_returns = np.mean(returns[:, -25:], axis=0) if returns.shape[1] >= 25 else np.mean(returns, axis=0)
    volatility_regime, vol_allocation_factor = check_volatility_regime(avg_returns)
    
    # Calculate drawdown protection factor
    drawdown_factor = calculate_drawdown_factor(recent_pnl)
    
    # Combined defensive factor
    defensive_factor = min(vol_allocation_factor, drawdown_factor)
    
    # If defensive factor is very low, exit most positions
    if defensive_factor < 0.2:
        return prev_pos * 0.1  # Keep only 10% of positions
    
    # PCA Decomposition
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, :12]

    # Enhanced similarity matrix
    norm_loadings = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-8)
    similarity = norm_loadings @ norm_loadings.T
    np.fill_diagonal(similarity, 0)

    # Adjust similarity threshold based on volatility regime
    if volatility_regime == "crisis":
        sim_threshold = max(0.8, np.percentile(similarity, 95))  # Higher threshold in crisis
    elif volatility_regime == "high_vol":
        sim_threshold = max(0.75, np.percentile(similarity, 92))
    else:
        sim_threshold = max(0.65, np.percentile(similarity, 85))
    
    # Enhanced pair selection
    pairs = []
    for i in range(nInst):
        for j in range(i+1, nInst):
            if similarity[i, j] > sim_threshold:
                corr = np.corrcoef(prices[i, -lookback:], prices[j, -lookback:])[0, 1]
                if 0.4 < corr < 0.99:
                    pairs.append((i, j))
    
    # Always ensure some pairs in normal conditions
    if not pairs and volatility_regime in ["normal", "elevated"]:
        for i in range(min(15, nInst)):
            for j in range(i+1, min(i+4, nInst)):
                pairs.append((i, j))

    # Enhanced volatility analysis
    vols = np.array([enhanced_volatility(returns[i, -25:]) for i in range(nInst)])
    avg_vol = np.nanmean(vols)
    
    # Detect market regime
    regime = detect_regime(avg_returns)
    
    # Very high volatility exit
    if np.isnan(avg_vol) or avg_vol > 1.5:
        return prev_pos * 0.05  # Keep only 5% of positions

    # Enhanced signal generation with bad momentum detection
    signal = np.zeros(nInst)
    signal_strength = np.zeros(nInst)
    expected_returns = np.zeros(nInst)
    variances = np.zeros(nInst)
    
    # Primary momentum signals with bad momentum shorting
    for i in range(nInst):
        # Detect bad momentum for aggressive shorting
        bad_momentum_signal = detect_bad_momentum(prices[i, -20:], returns[i, -20:])
        
        if abs(bad_momentum_signal) > 0.1:
            # Strong momentum signal (positive or negative)
            signal[i] = bad_momentum_signal
            signal_strength[i] = abs(bad_momentum_signal)
            expected_returns[i] = bad_momentum_signal * 0.1
            variances[i] = max(1e-6, np.var(returns[i, -10:]) if len(returns[i]) >= 10 else 0.01)
        else:
            # Fallback to regular momentum if no strong bad momentum
            mom_short = multi_timeframe_momentum(prices[i, -15:])
            
            if abs(mom_short) > 0.001:
                momentum_signal = min(3.0, abs(mom_short) * 80)  # Reduced amplification
                
                if mom_short > 0:
                    signal[i] = momentum_signal
                else:
                    signal[i] = -momentum_signal
                    
                signal_strength[i] = momentum_signal
                expected_returns[i] = mom_short * 8
                variances[i] = max(1e-6, np.var(returns[i, -10:]) if len(returns[i]) >= 10 else 0.01)
    
    # Secondary mean reversion pairs (reduced weight in volatile conditions)
    mean_reversion_weight = 0.3 if volatility_regime in ["crisis", "high_vol"] else 0.5
    
    for i, j in pairs[:20]:  # Fewer pairs in volatile conditions
        spread_hist = prices[i, -lookback:] - prices[j, -lookback:]
        
        # Adjust threshold based on volatility regime
        if volatility_regime == "crisis":
            z_score, threshold = enhanced_mean_reversion_signal(spread_hist, window=15)
            threshold *= 1.5  # Higher threshold in crisis
        else:
            z_score, threshold = enhanced_mean_reversion_signal(spread_hist, window=20)
        
        ou_speed = ou_mean_reversion_speed(spread_hist)
        spread_returns = np.diff(spread_hist) / (np.abs(spread_hist[:-1]) + 1e-8)
        expected_return_spread = np.mean(spread_returns[-6:]) if len(spread_returns) >= 6 else 0
        
        if abs(z_score) > threshold:
            base_signal = min(1.5, abs(z_score) / threshold)  # Reduced signal strength
            ou_boost = 1 + min(0.5, ou_speed * 2)
            
            regime_multiplier = {
                "normal": 1.0,
                "trending": 0.5,  # Less mean reversion in trending
                "high_vol": 0.3,  # Much less in high vol
                "crisis": 0.1     # Minimal in crisis
            }.get(volatility_regime, 0.5)
            
            final_signal = base_signal * ou_boost * regime_multiplier * mean_reversion_weight
            
            if z_score > threshold:
                signal[i] -= final_signal
                signal[j] += final_signal
            elif z_score < -threshold:
                signal[i] += final_signal
                signal[j] -= final_signal
                
            signal_strength[i] += final_signal
            signal_strength[j] += final_signal

    # Apply stop-loss and risk management
    for i in range(nInst):
        # Stop-loss based on recent performance
        if rolling_sharpe[i] < -1.5:  # Very poor recent performance
            signal[i] *= 0.1  # Drastically reduce signal
        elif rolling_sharpe[i] < -1.0:
            signal[i] *= 0.3
        elif rolling_sharpe[i] < -0.5:
            signal[i] *= 0.6

    # Ensure some trading in normal conditions, but not in crisis
    total_signal = np.sum(np.abs(signal))
    if total_signal < 0.01 and volatility_regime in ["normal", "elevated"]:
        # Create minimal momentum signals
        for i in range(nInst):
            recent_return = returns[i, -1] if len(returns[i]) > 0 else 0
            signal[i] = recent_return * 20  # Much reduced amplification
            expected_returns[i] = recent_return * 2
            variances[i] = 0.01
            signal_strength[i] = abs(signal[i])

    trade_count += 1

    # Defensive capital allocation with volatility and drawdown protection
    total_signal = np.sum(np.abs(signal))
    
    # Base weights from signals
    weights = signal / (total_signal + 1e-6)
    
    # Kelly criterion scaling with reduced leverage
    kelly_weights = np.zeros(nInst)
    for i in range(nInst):
        if abs(weights[i]) > 1e-8:
            max_leverage = 1.5 if volatility_regime == "normal" else 0.8  # Much lower leverage
            kelly_size = kelly_position_size(expected_returns[i], variances[i], max_leverage)
            kelly_weights[i] = weights[i] * abs(kelly_size)
    
    # Normalize Kelly weights
    kelly_total = np.sum(np.abs(kelly_weights))
    if kelly_total > 0:
        kelly_weights = kelly_weights / kelly_total
    else:
        kelly_weights = weights
    
    # Defensive capital scaling based on regime and performance
    # 1. Base scaling (much more conservative)
    base_scale = {
        "crisis": 0.2,      # 20% allocation in crisis
        "high_vol": 0.5,    # 50% in high vol
        "elevated": 0.8,    # 80% in elevated vol
        "normal": 1.5       # 150% in normal (reduced from 4x)
    }.get(volatility_regime, 1.0)
    
    # 2. Volatility scaling (defensive)
    vol_scale = min(1.5, 0.3 / (avg_vol + 1e-6))  # More conservative
    
    # 3. Signal confidence scaling
    confidence = np.mean(signal_strength[signal_strength > 0]) if np.any(signal_strength > 0) else 0
    confidence_scale = 0.5 + 0.5 * min(1.0, confidence / 3.0)  # More conservative
    
    # 4. Regime-based scaling (very defensive)
    regime_scale = {
        "normal": 1.2,
        "trending": 1.0,    # Reduced from 2.5
        "high_vol": 0.4,    # Much more defensive
        "crisis": 0.2       # Very defensive
    }.get(regime, 0.8)
    
    # 5. Performance-based scaling
    recent_sharpe = np.mean(rolling_sharpe[rolling_sharpe != 0]) if np.any(rolling_sharpe != 0) else 0
    performance_scale = max(0.2, min(1.5, 1.0 + recent_sharpe))
    
    # 6. Drawdown protection
    drawdown_scale = drawdown_factor
    
    # Combined scaling with all defensive factors
    total_scale = (base_scale * vol_scale * confidence_scale * regime_scale * 
                  performance_scale * drawdown_scale * defensive_factor)
    total_scale = min(2.0, total_scale)  # Cap at 2x leverage (much lower)
    
    dollar_alloc = kelly_weights * total_capital * total_scale
    
    prices_now = prices[:, -1]
    position = dollar_alloc / prices_now

    # Much more conservative position limits
    max_shares = max_notional / prices_now
    
    # Volatility-adjusted limits (very conservative)
    vol_adjusted_max = max_shares * (0.5 / (vols + 1e-6))  # More conservative
    vol_adjusted_max = np.clip(vol_adjusted_max, max_shares * 0.1, max_shares * 1.5)
    
    position = np.clip(position, -vol_adjusted_max, vol_adjusted_max)
    
    # Performance-based adjustments (more aggressive reduction for poor performance)
    for i in range(nInst):
        if abs(position[i]) > 0:
            recent_returns_asset = returns[i, -8:]
            if len(recent_returns_asset) >= 5:
                sharpe_proxy = np.mean(recent_returns_asset) / (np.std(recent_returns_asset) + 1e-6)
                rolling_sharpe[i] = 0.6 * rolling_sharpe[i] + 0.4 * sharpe_proxy
                
                # More aggressive reduction for poor performance
                if rolling_sharpe[i] < -1.2:
                    position[i] *= 0.1  # Reduce to 10%
                elif rolling_sharpe[i] < -0.8:
                    position[i] *= 0.3  # Reduce to 30%
                elif rolling_sharpe[i] < -0.4:
                    position[i] *= 0.6  # Reduce to 60%
                elif rolling_sharpe[i] > 0.5:  # Boost good performers (less aggressive)
                    position[i] *= 1.1  # Small boost

    # Adaptive smoothing based on volatility regime
    if volatility_regime == "crisis":
        alpha = 0.9   # High smoothing in crisis
    elif volatility_regime == "high_vol":
        alpha = 0.8   # High smoothing in high vol
    elif volatility_regime == "elevated":
        alpha = 0.7   # Medium smoothing
    else:
        alpha = 0.6   # Normal smoothing
        
    position = alpha * position + (1 - alpha) * prev_pos
    prev_pos = position

    return np.round(position)
