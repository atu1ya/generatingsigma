import numpy as np
import warnings
warnings.filterwarnings('ignore')

nInst = 50
currentPos = np.zeros(nInst)
model_weights = None
market_regime = 'neutral'
performance_tracker = []
signal_history = []

def compute_multi_horizon_features(prices):
    """Compute features across multiple time horizons for better prediction"""
    n_inst, n_days = prices.shape
    
    if n_days < 25:
        return None
    
    # Log returns
    log_prices = np.log(np.maximum(prices, 1e-8))
    returns = np.diff(log_prices, axis=1)
    
    features = []
    
    for i in range(n_inst):
        inst_returns = returns[i, :]
        inst_prices = prices[i, :]
        
        # Multi-horizon momentum (1d, 5d, 10d, 20d)
        mom_1d = inst_returns[-1] if n_days > 1 else 0
        mom_5d = np.sum(inst_returns[-5:]) if n_days > 5 else 0
        mom_10d = np.sum(inst_returns[-10:]) if n_days > 10 else 0
        mom_20d = np.sum(inst_returns[-20:]) if n_days > 20 else 0
        
        # Mean reversion signals
        mean_20 = np.mean(inst_returns[-20:]) if n_days > 20 else 0
        std_20 = np.std(inst_returns[-20:]) if n_days > 20 else 0.01
        z_score = (inst_returns[-1] - mean_20) / std_20 if std_20 > 0 else 0
        
        # Volatility (10-day rolling)
        volatility = np.std(inst_returns[-10:]) if n_days > 10 else 0.01
        
        # Price trend vs moving averages
        sma_5 = np.mean(inst_prices[-5:]) if n_days > 5 else inst_prices[-1]
        sma_20 = np.mean(inst_prices[-20:]) if n_days > 20 else inst_prices[-1]
        price_vs_sma5 = (inst_prices[-1] - sma_5) / sma_5 if sma_5 > 0 else 0
        price_vs_sma20 = (inst_prices[-1] - sma_20) / sma_20 if sma_20 > 0 else 0
        
        # Bollinger band position
        bb_position = (inst_prices[-1] - sma_20) / (2 * std_20 * sma_20) if std_20 > 0 and sma_20 > 0 else 0
        
        features.append([
            mom_1d, mom_5d, mom_10d, mom_20d,     # Momentum features
            z_score, -z_score,                    # Mean reversion (positive and negative)
            volatility, 1.0/max(volatility, 0.001), # Volatility and inverse vol
            price_vs_sma5, price_vs_sma20,        # Trend features
            bb_position                           # Technical indicator
        ])
    
    features_array = np.array(features)
    
    # Cross-sectional z-scores (relative performance)
    cross_sectional_features = []
    for j in range(features_array.shape[1]):
        col = features_array[:, j]
        if np.std(col) > 0:
            cs_z = (col - np.mean(col)) / np.std(col)
        else:
            cs_z = np.zeros_like(col)
        cross_sectional_features.append(cs_z)
    
    cross_sectional_features = np.array(cross_sectional_features).T
    
    # Combine individual and cross-sectional features
    all_features = np.concatenate([features_array, cross_sectional_features], axis=1)
    
    return all_features

def detect_market_regime(prices):
    """Improved market regime detection with better stability"""
    n_inst, n_days = prices.shape
    
    if n_days < 20:
        return 'neutral'
    
    # Calculate market index (equal weight of all instruments)
    log_prices = np.log(np.maximum(prices, 1e-8))
    market_returns = np.mean(np.diff(log_prices, axis=1), axis=0)
    
    if len(market_returns) < 15:
        return 'neutral'
    
    # Use longer lookback for stability
    lookback = min(15, len(market_returns))
    recent_returns = market_returns[-lookback:]
    
    # Trend metrics
    cumulative_return = np.sum(recent_returns)
    volatility = np.std(recent_returns)
    trend_strength = abs(cumulative_return) / (volatility * np.sqrt(lookback))
    
    # Correlation with time (trend persistence)
    time_indices = np.arange(lookback)
    if volatility > 0:
        trend_correlation = np.corrcoef(recent_returns, time_indices)[0, 1]
    else:
        trend_correlation = 0
    
    # Regime classification with more conservative thresholds
    if trend_strength > 1.5 and abs(trend_correlation) > 0.3:
        if cumulative_return > 0:
            return 'trending_up'
        else:
            return 'trending_down'
    elif volatility > np.std(market_returns[-30:] if len(market_returns) >= 30 else market_returns) * 1.3:
        return 'volatile'
    else:
        return 'mean_reverting'

def train_simple_predictor(features, returns, regime):
    """Train a simple linear model with relaxed requirements"""
    if features is None or len(features) < 5:
        return None
    
    n_inst, n_features = features.shape
    
    # Use shorter training window for more data
    lookback = min(15, len(returns[0]) - 1)
    if lookback < 5:
        return None
    
    X_train = []
    y_train = []
    
    # Build training data with relaxed target definition
    for t in range(max(0, len(returns[0]) - lookback), len(returns[0]) - 1):
        for i in range(n_inst):
            if t < len(returns[i]) - 1:
                # Features at time t
                inst_features = features[i, :8].copy()  # Use first 8 features only
                
                # Target: Next day return (regression approach)
                next_return = returns[i][t + 1]
                
                # Use all data, not just large moves
                X_train.append(inst_features)
                y_train.append(next_return)
    
    if len(X_train) < 10:
        return None
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Remove any NaN/inf values
    valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) < 5:
        return None
    
    try:
        # Simple linear regression with regularization
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0) + 1e-8
        X_scaled = (X_train - X_mean) / X_std
        
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        # Ridge regression solution
        lambda_reg = 0.1
        I = np.eye(X_with_intercept.shape[1])
        I[0, 0] = 0  # Don't regularize intercept
        
        weights = np.linalg.solve(
            X_with_intercept.T @ X_with_intercept + lambda_reg * I,
            X_with_intercept.T @ y_train
        )
        
        model_params = {
            'weights': weights,
            'X_mean': X_mean,
            'X_std': X_std,
            'regime': regime
        }
        
        return model_params
        
    except:
        # Fallback to simple momentum weights
        momentum_weights = np.array([0.1, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1])  # Intercept + 8 features
        return {
            'weights': momentum_weights,
            'X_mean': np.zeros(8),
            'X_std': np.ones(8),
            'regime': regime
        }

def generate_predictions(features, model_params, regime):
    """Generate return predictions using trained linear model"""
    if features is None or model_params is None:
        # Fallback to simple momentum signals
        if features is not None:
            momentum_5d = features[:, 1]  # 5-day momentum
            z_scores = features[:, 4]     # Z-scores
            predictions = 0.5 * np.tanh(momentum_5d * 5) + 0.5 * np.tanh(-z_scores)
            confidences = np.minimum(np.abs(predictions) + 0.3, 1.0)
            return predictions, confidences
        else:
            return np.zeros(nInst), np.zeros(nInst)
    
    n_inst = features.shape[0]
    predictions = np.zeros(n_inst)
    confidences = np.zeros(n_inst)
    
    weights = model_params['weights']
    X_mean = model_params['X_mean']
    X_std = model_params['X_std']
    
    for i in range(n_inst):
        # Use first 8 features for prediction
        inst_features = features[i, :8]
        
        # Standardize using training statistics
        inst_features_scaled = (inst_features - X_mean) / X_std
        
        # Add intercept and predict
        features_with_intercept = np.concatenate([[1.0], inst_features_scaled])
        prediction = np.dot(features_with_intercept, weights)
        
        # Scale prediction to reasonable range
        prediction = np.tanh(prediction * 3)
        
        # Confidence based on absolute prediction strength
        confidence = min(abs(prediction) + 0.2, 1.0)
        
        predictions[i] = prediction
        confidences[i] = confidence
    
    # Apply regime-specific adjustments
    if regime == 'volatile':
        predictions *= 0.5
        confidences *= 0.7
    elif regime in ['trending_up', 'trending_down']:
        # Boost momentum-aligned signals
        momentum_5d = features[:, 1]
        momentum_direction = np.sign(momentum_5d)
        prediction_direction = np.sign(predictions)
        
        # Boost aligned signals
        alignment = momentum_direction * prediction_direction
        boost_mask = alignment > 0
        predictions[boost_mask] *= 1.2
        confidences[boost_mask] *= 1.1
    
    return predictions, confidences

def filter_signals(predictions, confidences, features, regime):
    """Apply more relaxed signal filtering to ensure trading"""
    if features is None:
        return predictions.copy()
    
    n_inst = len(predictions)
    filtered_signals = predictions.copy()
    
    # 1. BASIC CONFIDENCE THRESHOLD - Much more relaxed
    confidence_threshold = 0.3  # Reduced from 0.6 to 0.3
    low_confidence_mask = confidences < confidence_threshold
    filtered_signals[low_confidence_mask] *= 0.5  # Don't zero out, just reduce
    
    # 2. SIGNAL STRENGTH THRESHOLD - More relaxed
    signal_strength_threshold = 0.1  # Reduced from 0.3 to 0.1
    weak_signal_mask = np.abs(predictions) < signal_strength_threshold
    filtered_signals[weak_signal_mask] = 0
    
    # 3. VOLATILITY FILTER - Less aggressive penalty
    volatilities = features[:, 6]  # Volatility feature
    high_vol_threshold = np.percentile(volatilities, 90)  # Only avoid top 10% most volatile
    high_vol_mask = volatilities > high_vol_threshold
    filtered_signals[high_vol_mask] *= 0.5  # Reduce instead of near-zero
    
    # 4. REGIME-SPECIFIC FILTERS - More permissive
    if regime == 'mean_reverting':
        # For mean reversion, allow moderate z-scores
        z_scores = features[:, 4]  # Z-score feature
        moderate_z_mask = np.abs(z_scores) < 1.0  # Reduced from 1.5
        filtered_signals[moderate_z_mask] *= 0.7  # Reduce instead of zero
        
    elif regime in ['trending_up', 'trending_down']:
        # For trending markets, be more permissive with momentum
        momentum_5d = features[:, 1]
        momentum_direction = np.sign(momentum_5d)
        signal_direction = np.sign(filtered_signals)
        
        # Reduce (don't eliminate) counter-momentum signals
        opposite_direction_mask = (momentum_direction * signal_direction) < 0
        filtered_signals[opposite_direction_mask] *= 0.6
        
        # Less strict momentum requirement
        weak_momentum_mask = np.abs(momentum_5d) < np.percentile(np.abs(momentum_5d), 50)  # Reduced from 70%
        filtered_signals[weak_momentum_mask] *= 0.8
    
    # 5. ENSURE WE HAVE SOME SIGNALS - Keep top 20% of signals
    signal_quality_scores = np.abs(filtered_signals) * confidences
    if np.sum(signal_quality_scores > 0) > 0:
        quality_threshold = np.percentile(signal_quality_scores[signal_quality_scores > 0], 50)  # Reduced from 75%
        low_quality_mask = signal_quality_scores < quality_threshold
        filtered_signals[low_quality_mask] *= 0.3  # Reduce instead of eliminate
    
    # 6. CROSS-SECTIONAL FILTER - Only apply if we have enough signals
    if np.sum(np.abs(filtered_signals) > 0) > 10:  # Only if we have >10 non-zero signals
        if np.std(predictions) > 0:
            cs_z_scores = (predictions - np.mean(predictions)) / np.std(predictions)
            moderate_signal_mask = np.abs(cs_z_scores) < 0.7  # Reduced from 1.0
            filtered_signals[moderate_signal_mask] *= 0.7
    
    return filtered_signals

def construct_optimal_portfolio(signals, prices, regime):
    """Build portfolio with more aggressive position sizing to ensure trading"""
    n_inst = len(signals)
    current_prices = prices[:, -1]
    
    # Less conservative position limits
    max_dollar_per_instrument = 9000  # Slightly reduced but not too conservative
    max_positions = np.floor(max_dollar_per_instrument / current_prices).astype(int)
    
    # Less aggressive volatility penalty
    if prices.shape[1] > 10:
        recent_returns = np.diff(np.log(prices[:, -10:]), axis=1)
        volatilities = np.std(recent_returns, axis=1)
        vol_adjustment = 1.0 / (1.0 + 1.5 * volatilities)  # Reduced penalty
    else:
        vol_adjustment = np.ones(n_inst)
    
    # Trade more positions (top 10-15 long, 10-15 short)
    signal_ranks = np.argsort(signals)
    
    target_positions = np.zeros(n_inst)
    
    # Select more performers to ensure trading
    n_positions = min(15, max(8, n_inst // 4))  # 8-15 positions each side
    
    # Ensure we have enough non-zero signals
    non_zero_signals = np.sum(np.abs(signals) > 0)
    if non_zero_signals < 10:
        # If too few signals, lower the bar
        signal_threshold = np.percentile(np.abs(signals), 60) if non_zero_signals > 0 else 0
        boosted_signals = signals.copy()
        weak_mask = (np.abs(signals) > signal_threshold/2) & (np.abs(signals) <= signal_threshold)
        boosted_signals[weak_mask] *= 2  # Boost weaker signals
        signals = boosted_signals
        
        # Recompute ranks
        signal_ranks = np.argsort(signals)
    
    # Long positions (highest predicted returns)
    long_candidates = signal_ranks[-n_positions:]
    for i, idx in enumerate(long_candidates):
        if signals[idx] > 0:
            # More aggressive position sizing
            signal_strength = min(abs(signals[idx]), 2.0)
            rank_weight = (i + 1) / n_positions
            
            base_size = max_positions[idx] * 0.7 * vol_adjustment[idx]  # Less conservative
            position_size = base_size * max(signal_strength / 2.0, 0.3) * rank_weight  # Minimum 30% sizing
            target_positions[idx] = max(int(position_size), 1)  # Ensure at least 1 share
    
    # Short positions (lowest predicted returns)
    short_candidates = signal_ranks[:n_positions]
    for i, idx in enumerate(short_candidates):
        if signals[idx] < 0:
            signal_strength = min(abs(signals[idx]), 2.0)
            rank_weight = (n_positions - i) / n_positions
            
            base_size = max_positions[idx] * 0.7 * vol_adjustment[idx]
            position_size = base_size * max(signal_strength / 2.0, 0.3) * rank_weight
            target_positions[idx] = min(-max(int(position_size), 1), -1)  # Ensure at least -1 share
    
    # Apply position limits
    target_positions = np.clip(target_positions, -max_positions, max_positions)
    
    # Less strict dollar neutrality
    long_value = np.sum(np.maximum(target_positions, 0) * current_prices)
    short_value = abs(np.sum(np.minimum(target_positions, 0) * current_prices))
    
    if long_value > 0 and short_value > 0:
        imbalance = long_value / short_value
        if imbalance > 1.5:  # More permissive threshold
            target_positions[target_positions > 0] = (target_positions[target_positions > 0] * 0.85).astype(int)
        elif imbalance < 0.67:
            target_positions[target_positions < 0] = (target_positions[target_positions < 0] * 0.85).astype(int)
    
    return target_positions

def getMyPosition(prcSoFar):
    global currentPos, signal_history

    n_inst, n_days = prcSoFar.shape
    if n_days < 25:
        return np.zeros(n_inst)

    # === 1. Feature Engineering ===
    log_prices = np.log(np.maximum(prcSoFar, 1e-8))
    returns = np.diff(log_prices, axis=1)
    features = []
    for i in range(n_inst):
        inst_returns = returns[i, :]
        mom_1d = inst_returns[-1]
        mom_5d = np.sum(inst_returns[-5:])
        z_10 = (inst_returns[-1] - np.mean(inst_returns[-10:])) / (np.std(inst_returns[-10:]) + 1e-8)
        z_20 = (inst_returns[-1] - np.mean(inst_returns[-20:])) / (np.std(inst_returns[-20:]) + 1e-8)
        vol_10 = np.std(inst_returns[-10:])
        # PCA residual (first PC removed)
        if n_days > 20:
            X = returns[:, -20:]
            X = X - X.mean(axis=1, keepdims=True)
            u, s, vh = np.linalg.svd(X, full_matrices=False)
            pc1 = np.outer(u[:, 0], s[0] * vh[0])
            pca_resid = inst_returns[-1] - pc1[i, -1]
        else:
            pca_resid = 0
        features.append([mom_1d, mom_5d, z_10, vol_10, pca_resid])
    features = np.array(features)

    # === 2. Regime Detection ===
    market_ret_10 = np.mean(np.mean(returns[:, -10:], axis=0))
    trending = abs(market_ret_10) > 0.002

    # === 3. Signal Boosting (Momentum + Mean-Reversion + Volatility Penalty) ===
    # Short-term momentum: mom_1d, mom_5d
    # Medium-term reversion: z_10
    # Volatility: vol_10
    # PCA residual: pca_resid
    # Penalize top 20% volatility
    vol_10 = features[:, 3]
    vol_thresh = np.percentile(vol_10, 80)
    vol_penalty = np.where(vol_10 > vol_thresh, 0.5, 1.0) * (1 / (1 + vol_10))

    # === 4. Simple Linear Model/Threshold Classifier ===
    # We'll use a simple linear model with hand-tuned weights
    # [Return-1d, Return-5d, Z-score, Volatility, PCA Residual]
    weights = np.array([0.6, 0.4, -0.5, -0.2, 0.2])
    # Regime switching: use momentum in trending, mean-reversion otherwise
    if trending:
        # Momentum regime: upweight momentum, downweight reversion
        weights = np.array([0.8, 0.7, -0.2, -0.2, 0.1])
    else:
        # Mean-reversion regime: upweight reversion, downweight momentum
        weights = np.array([0.2, 0.1, -0.7, -0.2, 0.3])

    # Linear model prediction
    model_signal = features @ weights
    # Model confidence: sigmoid of abs(model_signal)
    model_confidence = 1 / (1 + np.exp(-2 * np.abs(model_signal)))

    # Only take positions if confidence > 0.55 or abs(pred) > 0.002
    valid_mask = (model_confidence > 0.55) | (np.abs(model_signal) > 0.002)
    filtered_signal = np.where(valid_mask, model_signal, 0.0)

    # Apply volatility penalty
    combined_signal = filtered_signal * vol_penalty

    # === 5. Entry/Exit Logic ===
    std_signal = np.std(combined_signal)
    entry_thresh = 1.5 * std_signal
    exit_thresh = 0.5 * std_signal

    positions = np.zeros(n_inst)
    # Only trade top 5 long and 5 short instruments
    ranked = np.argsort(combined_signal)
    longs = ranked[-5:]
    shorts = ranked[:5]

    max_pos = np.floor(10000 / prcSoFar[:, -1]).astype(int)
    for idx in longs:
        if combined_signal[idx] > entry_thresh:
            # Volatility-adjusted sizing
            size = max_pos[idx] * (1 / (1 + vol_10[idx]))
            positions[idx] = int(size * min(1, abs(combined_signal[idx]) / (2 * entry_thresh)))
        elif abs(combined_signal[idx]) < exit_thresh:
            positions[idx] = 0

    for idx in shorts:
        if combined_signal[idx] < -entry_thresh:
            size = max_pos[idx] * (1 / (1 + vol_10[idx]))
            positions[idx] = -int(size * min(1, abs(combined_signal[idx]) / (2 * entry_thresh)))
        elif abs(combined_signal[idx]) < exit_thresh:
            positions[idx] = 0

    # Smoothing to reduce turnover
    if len(signal_history) == n_inst:
        alpha = 0.5
        # Ensure signal_history is a numpy array for arithmetic
        positions = alpha * positions + (1 - alpha) * np.array(signal_history)
    signal_history[:] = positions

    currentPos = positions.astype(int)
    return currentPos