import numpy as np
from scipy import stats
from scipy.optimize import minimize

nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional
prev_pos = np.zeros(nInst)
rolling_sharpe = np.zeros(nInst)
rolling_returns = np.zeros((nInst, 100))  # Store much more history
trade_count = 0
recent_pnl = np.zeros(50)  # Track more P&L history
volatility_regime = "normal"
return_history = []  # Store all returns for complex analysis
price_history = []   # Store all prices for complex analysis
position_history = [] # Store position history

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

# Advanced statistical analysis functions
def calculate_var_cvar(returns, confidence=0.05):
    """Calculate Value at Risk and Conditional Value at Risk"""
    if len(returns) < 10:
        return 0.0, 0.0
    
    sorted_returns = np.sort(returns)
    var_index = int(confidence * len(sorted_returns))
    var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[-1]
    cvar = np.mean(sorted_returns[:var_index]) if var_index > 0 else var
    return var, cvar

def advanced_regime_detection(all_returns, window=50):
    """Advanced regime detection using multiple statistical measures"""
    if len(all_returns) < window:
        return "insufficient_data", 1.0
    
    recent_returns = all_returns[-window:]
    
    # Multiple statistical measures
    volatility = np.std(recent_returns) * np.sqrt(252)
    mean_return = np.mean(recent_returns)
    
    # Simple skewness and kurtosis calculations
    centered_returns = recent_returns - mean_return
    skewness = np.mean(centered_returns**3) / (np.std(recent_returns)**3 + 1e-8)
    kurtosis = np.mean(centered_returns**4) / (np.std(recent_returns)**4 + 1e-8) - 3
    
    # Autocorrelation
    if len(recent_returns) > 1:
        autocorr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0,1]
        if np.isnan(autocorr):
            autocorr = 0
    else:
        autocorr = 0
    
    # Regime classification with confidence scores
    if volatility > 0.5 and abs(skewness) > 1.5:
        return "crisis", 0.1
    elif volatility > 0.35 and kurtosis > 5:
        return "high_stress", 0.2
    elif abs(autocorr) > 0.3 and volatility > 0.25:
        return "trending_volatile", 0.4
    elif autocorr > 0.2 and volatility < 0.2:
        return "trending_stable", 1.2
    elif volatility > 0.3:
        return "high_volatility", 0.5
    elif abs(skewness) > 1.0 or kurtosis > 3:  # Non-normal returns
        return "anomalous", 0.3
    else:
        return "normal", 1.0

def portfolio_optimization(expected_returns, cov_matrix, risk_aversion=1.0):
    """Mean-variance portfolio optimization"""
    n = len(expected_returns)
    if n == 0 or np.any(np.isnan(expected_returns)) or np.any(np.isnan(cov_matrix)):
        return np.zeros(n)
    
    # Add regularization to covariance matrix
    cov_matrix += np.eye(n) * 1e-6
    
    try:
        # Solve for optimal weights: w = (1/λ) * Σ^(-1) * μ
        inv_cov = np.linalg.pinv(cov_matrix)
        optimal_weights = np.dot(inv_cov, expected_returns) / risk_aversion
        
        # Normalize weights
        total_weight = np.sum(np.abs(optimal_weights))
        if total_weight > 0:
            optimal_weights = optimal_weights / total_weight
        
        return optimal_weights
    except:
        return np.zeros(n)

def advanced_momentum_analysis(prices, returns_history):
    """Comprehensive momentum analysis using multiple timeframes and methods"""
    if len(prices) < 20 or len(returns_history) < 20:
        return 0.0, 0.0  # momentum, confidence
    
    # Multiple momentum calculations
    momentum_signals = []
    
    # Price momentum (multiple timeframes)
    for window in [5, 10, 20, 30]:
        if len(prices) >= window + 1:
            mom = (prices[-1] - prices[-window-1]) / prices[-window-1]
            momentum_signals.append(mom)
    
    # Return-based momentum
    if len(returns_history) >= 10:
        recent_returns = returns_history[-10:]
        avg_return = np.mean(recent_returns)
        momentum_signals.append(avg_return * 10)  # Scale up
    
    # Trend strength using simple linear regression
    if len(prices) >= 20:
        x = np.arange(len(prices[-20:]))
        y = prices[-20:]
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        
        # Calculate R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean)**2)
        y_pred = slope * x + (sum_y - slope * sum_x) / n
        ss_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        trend_strength = slope * r_squared  # Slope weighted by R-squared
        momentum_signals.append(trend_strength / np.mean(prices[-20:]))
    
    if not momentum_signals:
        return 0.0, 0.0
    
    # Combine momentum signals
    momentum_array = np.array(momentum_signals)
    avg_momentum = np.mean(momentum_array)
    momentum_consistency = 1.0 - np.std(momentum_array) / (np.abs(avg_momentum) + 1e-8)
    confidence = max(0.0, min(1.0, momentum_consistency))
    
    return avg_momentum, confidence

def sophisticated_mean_reversion(spread_history, all_returns):
    """Advanced mean reversion analysis with statistical tests"""
    if len(spread_history) < 30:
        return 0.0, 1.0, 0.0
    
    # Simple mean reversion test
    spread_diff = np.diff(spread_history)
    spread_lag = spread_history[:-1]
    
    if np.std(spread_lag) < 1e-8:
        return 0.0, 1.0, 0.0
    
    # Simple regression for mean reversion
    mean_spread = np.mean(spread_lag)
    regression_x = spread_lag - mean_spread
    regression_y = spread_diff
    
    # Simple linear regression
    n = len(regression_x)
    sum_x = np.sum(regression_x)
    sum_y = np.sum(regression_y)
    sum_xy = np.sum(regression_x * regression_y)
    sum_x2 = np.sum(regression_x * regression_x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
    
    # Calculate R-squared
    y_mean = np.mean(regression_y)
    ss_tot = np.sum((regression_y - y_mean)**2)
    y_pred = slope * regression_x + (sum_y - slope * sum_x) / n
    ss_res = np.sum((regression_y - y_pred)**2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-8))
    
    mean_reversion_speed = -slope  # Negative slope indicates mean reversion
    
    # Calculate z-score with adaptive window
    adaptive_window = min(len(spread_history), max(20, len(all_returns) // 10))
    recent_spread = spread_history[-adaptive_window:]
    z_score = (spread_history[-1] - np.mean(recent_spread)) / (np.std(recent_spread) + 1e-8)
    
    # Adaptive threshold based on mean reversion strength
    base_threshold = 1.0
    threshold = base_threshold * (1.0 + abs(r_squared))  # Higher threshold if stronger relationship
    
    return z_score, threshold, mean_reversion_speed

def dynamic_position_sizing(signal_strength, expected_return, variance, regime_confidence, 
                          recent_performance, max_leverage=2.0):
    """Dynamic position sizing using multiple factors"""
    if variance <= 0 or signal_strength == 0:
        return 0.0
    
    # Kelly criterion base
    kelly_fraction = expected_return / variance
    kelly_fraction = np.clip(kelly_fraction, -max_leverage, max_leverage)
    
    # Adjust for signal strength
    strength_multiplier = min(2.0, abs(signal_strength))
    
    # Adjust for regime confidence
    regime_multiplier = regime_confidence
    
    # Adjust for recent performance
    performance_multiplier = max(0.1, min(2.0, 1.0 + recent_performance))
    
    # Combine all factors
    final_size = kelly_fraction * strength_multiplier * regime_multiplier * performance_multiplier
    
    return np.clip(final_size, -max_leverage, max_leverage)
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
    global return_history, price_history, position_history
    
    n, t = prcSoFar.shape
    if t < 10:  # Much lower requirement for starting trades
        return np.zeros(nInst)

    # Store all historical data for complex analysis
    prices = prcSoFar
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    
    # Update comprehensive history storage
    if t > 1:
        current_returns = returns[:, -1]
        return_history.append(current_returns.copy())
        price_history.append(prices[:, -1].copy())
        position_history.append(prev_pos.copy())
        
        # Keep only last 1000 periods to manage memory (can be increased)
        if len(return_history) > 1000:
            return_history = return_history[-1000:]
            price_history = price_history[-1000:]
            position_history = position_history[-1000:]

    # Create comprehensive return history matrix
    if len(return_history) < 5:  # Much lower requirement
        # Generate simple momentum-based positions for early trading
        if t >= 5:
            recent_returns = np.log(prices[:, -1] / prices[:, -5])
            simple_signal = recent_returns * 20  # Amplify for position sizing
            prices_now = prices[:, -1]
            max_shares = max_notional / prices_now
            simple_position = simple_signal * max_shares * 0.5
            return np.clip(simple_position, -max_shares, max_shares)
        return np.zeros(nInst)
    
    all_returns_matrix = np.array(return_history).T  # Shape: (nInst, time)
    all_price_matrix = np.array(price_history).T     # Shape: (nInst, time)
    
    # Calculate portfolio-level returns for regime detection
    portfolio_returns = []
    for i in range(len(return_history)-1):
        if i < len(position_history):
            port_return = np.sum(position_history[i] * return_history[i+1])
            portfolio_returns.append(port_return)
    
    # Advanced regime detection using all available data
    if len(portfolio_returns) > 10:
        regime, regime_confidence = advanced_regime_detection(portfolio_returns)
    else:
        regime, regime_confidence = "insufficient_data", 0.5
    
    # Calculate VaR and CVaR for risk management
    if len(portfolio_returns) > 20:
        var_5, cvar_5 = calculate_var_cvar(portfolio_returns, 0.05)
        var_1, cvar_1 = calculate_var_cvar(portfolio_returns, 0.01)
    else:
        var_5 = var_1 = cvar_5 = cvar_1 = 0.0
    
    # Stop trading if extreme risk detected (much more permissive)
    if cvar_1 < -0.25:  # Only stop in truly extreme cases
        return prev_pos * 0.8  # Less aggressive reduction
    
    # PCA analysis using longer history
    lookback = min(100, all_returns_matrix.shape[1])
    R = all_returns_matrix[:, -lookback:]
    R -= np.mean(R, axis=1, keepdims=True)
    
    # Enhanced PCA
    cov = np.cov(R)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, :15]  # Use more principal components
    
    # Compute similarity matrix
    norm_loadings = loadings / (np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-8)
    similarity = norm_loadings @ norm_loadings.T
    np.fill_diagonal(similarity, 0)
    
    # Regime-adjusted similarity threshold (more permissive)
    threshold_adj = {
        "crisis": 0.8, "high_stress": 0.75, "high_volatility": 0.7,
        "trending_volatile": 0.65, "trending_stable": 0.6, "anomalous": 0.75,
        "normal": 0.6, "insufficient_data": 0.65  # Lower thresholds for more trading
    }
    sim_threshold = threshold_adj.get(regime, 0.7)
    
    # Find pairs with sophisticated filtering
    pairs = []
    for i in range(nInst):
        for j in range(i+1, nInst):
            if similarity[i, j] > sim_threshold:
                # Correlation check using more data
                if all_returns_matrix.shape[1] >= 50:
                    corr = np.corrcoef(all_returns_matrix[i, -50:], all_returns_matrix[j, -50:])[0, 1]
                    if 0.2 < corr < 0.97:  # Much wider range for more pairs
                        pairs.append((i, j))
    
    # Ensure minimum number of pairs for regular trading
    if len(pairs) < 5:  # Lower minimum for more consistent trading
        # Add backup pairs based on highest similarity
        similarity_flat = similarity.flatten()
        top_indices = np.argsort(similarity_flat)[::-1]
        for idx in top_indices[:100]:  # Check more indices
            i, j = divmod(idx, nInst)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
                if len(pairs) >= 15:  # Ensure we have enough pairs
                    break

    # Advanced signal generation using all historical data
    signal = np.zeros(nInst)
    signal_strength = np.zeros(nInst)
    expected_returns = np.zeros(nInst)
    variances = np.zeros(nInst)
    
    # Individual asset momentum analysis using full history
    for i in range(nInst):
        if all_price_matrix.shape[1] >= 10:  # Much lower requirement
            asset_prices = all_price_matrix[i, :]
            asset_returns = all_returns_matrix[i, :]
            
            # Simple momentum for faster trading decisions
            if len(asset_prices) >= 5:
                simple_momentum = (asset_prices[-1] - asset_prices[-5]) / asset_prices[-5]
                momentum_signal = simple_momentum * 30.0  # Strong amplification
                
                # Always trade if there's any momentum (no confidence threshold)
                signal[i] = momentum_signal
                signal_strength[i] = abs(momentum_signal)
                expected_returns[i] = simple_momentum * 2
                variances[i] = np.var(asset_returns[-10:]) if len(asset_returns) >= 10 else 0.01
            else:
                # Advanced momentum analysis (if enough data)
                momentum, momentum_confidence = advanced_momentum_analysis(asset_prices, asset_returns)
                
                # Much lower confidence threshold
                if momentum_confidence > 0.1:  # Reduced from 0.2
                    base_signal = momentum * momentum_confidence * 12.0  # Increased amplification
                    
                    # Regime adjustment for momentum (more permissive)
                    regime_momentum_multiplier = {
                        "trending_stable": 3.0, "trending_volatile": 2.5, "normal": 2.0,
                        "high_volatility": 1.5, "anomalous": 1.2, "crisis": 0.8, "high_stress": 1.0
                    }.get(regime, 1.5)  # Higher multipliers for more trading
                    
                    final_momentum_signal = base_signal * regime_momentum_multiplier
                    
                    # Apply position sizing
                    asset_variance = np.var(asset_returns[-15:]) if len(asset_returns) >= 15 else 0.01
                    recent_sharpe = rolling_sharpe[i] if rolling_sharpe[i] != 0 else 0
                    
                    position_size = dynamic_position_sizing(
                        abs(final_momentum_signal), momentum * momentum_confidence,
                        asset_variance, regime_confidence, recent_sharpe, max_leverage=3.0  # Higher leverage
                    )
                    
                    signal[i] = np.sign(final_momentum_signal) * abs(position_size)
                    signal_strength[i] = abs(final_momentum_signal)
                    expected_returns[i] = momentum * momentum_confidence
                    variances[i] = asset_variance
    
    # Pairs trading with sophisticated mean reversion
    for i, j in pairs[:50]:  # Increased from 40 for more trading
        if all_price_matrix.shape[1] >= 10:  # Much lower requirement
            # Create spread history
            spread_history = all_price_matrix[i, :] - all_price_matrix[j, :]
            portfolio_returns_for_spread = portfolio_returns if len(portfolio_returns) > 0 else [0]
            
            # Simple mean reversion for faster decisions
            if len(spread_history) >= 10:
                spread_mean = np.mean(spread_history[-10:])
                spread_std = np.std(spread_history[-10:]) + 1e-8
                z_score = (spread_history[-1] - spread_mean) / spread_std
                
                # Very permissive trading threshold
                if abs(z_score) > 0.5:  # Much lower threshold
                    pair_signal_strength = min(3.0, abs(z_score)) * 0.5
                    
                    if z_score > 0.5:  # Spread too high, short i, long j
                        signal[i] -= pair_signal_strength
                        signal[j] += pair_signal_strength
                    elif z_score < -0.5:  # Spread too low, long i, short j
                        signal[i] += pair_signal_strength
                        signal[j] -= pair_signal_strength
                    
                    # Update tracking
                    signal_strength[i] = np.maximum(signal_strength[i], pair_signal_strength)
                    signal_strength[j] = np.maximum(signal_strength[j], pair_signal_strength)
                    expected_returns[i] += np.sign(signal[i]) * 0.01
                    expected_returns[j] += np.sign(signal[j]) * 0.01
                    variances[i] = max(float(variances[i]), spread_std**2)
                    variances[j] = max(float(variances[j]), spread_std**2)
            else:
                # Sophisticated mean reversion analysis (if enough data)
                z_score, threshold, mean_rev_speed = sophisticated_mean_reversion(
                    spread_history, portfolio_returns_for_spread
                )
                
                # More permissive trading conditions
                if abs(z_score) > threshold * 0.5 and mean_rev_speed > 0.01:  # Much lower thresholds
                    # Calculate signal strength based on z-score and mean reversion speed
                    base_signal = min(3.0, abs(z_score) / threshold) * mean_rev_speed
                    
                    # Regime adjustment for mean reversion (more permissive)
                    regime_mr_multiplier = {
                        "high_volatility": 2.5, "crisis": 3.0, "high_stress": 2.8,
                        "anomalous": 2.2, "normal": 2.0, "trending_stable": 1.2, "trending_volatile": 1.8
                    }.get(regime, 1.5)  # Higher multipliers
                    
                    final_mr_signal = base_signal * regime_mr_multiplier * 1.2  # Increased weight
                    
                    # Calculate expected returns and variances for the pair
                    spread_returns = np.diff(spread_history[-20:]) if len(spread_history) >= 21 else [0]
                    spread_variance = np.var(spread_returns) if len(spread_returns) > 3 else 0.01
                    expected_spread_return = np.mean(spread_returns) if len(spread_returns) > 0 else 0
                    
                    if z_score > threshold:  # Spread too high, short i, long j
                        signal[i] -= final_mr_signal
                        signal[j] += final_mr_signal
                        expected_returns[i] -= expected_spread_return
                        expected_returns[j] += expected_spread_return
                    elif z_score < -threshold:  # Spread too low, long i, short j
                        signal[i] += final_mr_signal
                        signal[j] -= final_mr_signal
                        expected_returns[i] += expected_spread_return
                        expected_returns[j] -= expected_spread_return
                    
                    signal_strength[i] = np.maximum(signal_strength[i], final_mr_signal)
                    signal_strength[j] = np.maximum(signal_strength[j], final_mr_signal)
                    variances[i] = max(float(variances[i]), spread_variance)
                    variances[j] = max(float(variances[j]), spread_variance)

    # Portfolio optimization
    if np.any(signal != 0):
        # Create expected returns and covariance matrix for optimization
        non_zero_indices = np.where(signal != 0)[0]
        if len(non_zero_indices) > 1:
            sub_expected_returns = expected_returns[non_zero_indices]
            sub_returns_matrix = all_returns_matrix[non_zero_indices, -50:] if all_returns_matrix.shape[1] >= 50 else all_returns_matrix[non_zero_indices, :]
            sub_cov_matrix = np.cov(sub_returns_matrix)
            
            # Portfolio optimization
            optimized_weights = portfolio_optimization(sub_expected_returns, sub_cov_matrix, risk_aversion=2.0)
            
            # Apply optimized weights back to signals
            for idx, orig_idx in enumerate(non_zero_indices):
                if idx < len(optimized_weights):
                    signal[orig_idx] = optimized_weights[idx] * np.sign(signal[orig_idx]) * signal_strength[orig_idx]

    # Risk management and position sizing
    total_signal = np.sum(np.abs(signal))
    if total_signal < 0.001:  # Much lower threshold for ensuring trading
        # Force some trading based on recent returns if no signals
        for i in range(nInst):
            if len(return_history) >= 5:
                recent_momentum = np.mean(all_returns_matrix[i, -5:])
                if abs(recent_momentum) > 0.001:  # Very low threshold
                    signal[i] = recent_momentum * 50  # Amplify for trading
                    expected_returns[i] = recent_momentum * 5
                    variances[i] = 0.02
                    signal_strength[i] = abs(signal[i])
        
        total_signal = np.sum(np.abs(signal))
        if total_signal < 0.001:
            # Force minimal trading with random walk component
            np.random.seed(trade_count)  # Deterministic randomness
            random_signals = np.random.randn(nInst) * 0.1
            signal = random_signals
            expected_returns = random_signals * 0.5
            variances[:] = 0.02
            signal_strength = np.abs(signal)
            return prev_pos * 0.98  # Small decay instead of no trading
    
    # Calculate portfolio-level risk metrics
    if len(portfolio_returns) > 10:
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
    else:
        portfolio_vol = 0.3
        portfolio_sharpe = 0.0
    
    # Regime-based capital allocation (more aggressive for trading)
    base_allocation = {
        "crisis": 0.4, "high_stress": 0.6, "high_volatility": 0.8, "anomalous": 0.7,
        "trending_volatile": 1.2, "trending_stable": 1.5, "normal": 1.3, "insufficient_data": 1.0
    }.get(regime, 0.8)  # Higher allocations across the board
    
    # Performance-based scaling
    performance_scale = max(0.2, min(2.0, 1.0 + portfolio_sharpe))
    
    # Volatility-based scaling
    vol_scale = min(2.0, 0.3 / (portfolio_vol + 1e-6))
    
    # CVaR-based scaling
    cvar_scale = max(0.1, min(1.5, 1.0 + cvar_5 * 10)) if cvar_5 != 0 else 1.0
    
    # Combined scaling (more aggressive)
    total_scale = base_allocation * performance_scale * vol_scale * cvar_scale * regime_confidence
    total_scale = min(4.0, max(0.5, total_scale))  # Higher cap, higher floor
    
    # Normalize signals and apply scaling
    weights = signal / (total_signal + 1e-6)
    dollar_alloc = weights * total_capital * total_scale
    
    prices_now = prices[:, -1]
    position = dollar_alloc / prices_now
    
    # Apply position limits based on individual asset risk
    max_shares = max_notional / prices_now
    for i in range(nInst):
        if len(return_history) >= 20:
            asset_vol = np.std(all_returns_matrix[i, -20:]) * np.sqrt(252)
            vol_adjusted_limit = max_shares[i] * (0.4 / (asset_vol + 1e-6))
            position[i] = np.clip(position[i], -vol_adjusted_limit, vol_adjusted_limit)
    
    # Final risk checks and ensure some trading always happens
    total_final_signal = np.sum(np.abs(position))
    if total_final_signal < max_notional * 0.1:  # If total position is too small
        # Boost positions to ensure meaningful trading
        scaling_factor = max_notional * 0.2 / (total_final_signal + 1e-6)
        position *= min(scaling_factor, 3.0)  # Cap the boost
    
    for i in range(nInst):
        if rolling_sharpe[i] < -2.0:  # Very poor recent performance
            position[i] *= 0.1
        elif rolling_sharpe[i] < -1.0:
            position[i] *= 0.3
        elif rolling_sharpe[i] > 1.0:  # Very good performance
            position[i] *= 1.3
    
    # Adaptive smoothing based on regime and volatility (less smoothing for more trading)
    smoothing_factor = {
        "crisis": 0.6, "high_stress": 0.55, "high_volatility": 0.5,
        "anomalous": 0.45, "trending_volatile": 0.3, "trending_stable": 0.2, "normal": 0.3
    }.get(regime, 0.4)
    
    position = smoothing_factor * position + (1 - smoothing_factor) * prev_pos
    
    # Update performance tracking
    if len(return_history) >= 2:
        for i in range(nInst):
            if len(return_history) >= 15:
                recent_returns = all_returns_matrix[i, -15:]
                sharpe_proxy = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
                rolling_sharpe[i] = 0.7 * rolling_sharpe[i] + 0.3 * sharpe_proxy
    
    prev_pos = position.copy()
    trade_count += 1
    
    return np.round(position)
