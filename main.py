"""
Refined Profitable Quantitative Trading Strategy

Focus on high-quality signals with strict risk management:
1. Selective Pairs Trading with strong cointegration
2. Momentum with volatility filtering  
3. Conservative position sizing
4. Strict risk controls and stop losses

STRATEGY PRINCIPLES:
- Quality over quantity: fewer, higher-confidence trades
- Strong risk management with position limits
- Momentum and mean reversion signals must align
- Conservative capital utilization with gradual scaling
"""

import numpy as np
from scipy import stats
from scipy.stats import zscore

# Constants for the trading strategy
nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional

# Position tracking
prev_pos = np.zeros(nInst, dtype=float)
position_holding_time = np.zeros(nInst, dtype=float)
trade_count = 0
pair_spreads = {}
active_pairs = set()
recent_pnl = []  # Track recent performance

# Conservative Strategy Parameters
lookback_short = 10       # Short-term lookback
lookback_medium = 20      # Medium-term lookbook  
lookback_long = 30        # Long-term lookback
correlation_lookback = 40  # Correlation calculation window

# More reasonable thresholds for actual trading
momentum_threshold = 0.01   # Very low threshold (1%) to ensure trades
mean_revert_threshold = 0.8  # Lower threshold for mean reversion
pairs_z_entry = 1.0        # Lower entry threshold for pairs
pairs_z_exit = 0.3         # Conservative exit threshold
correlation_min = 0.3      # Much lower correlation requirement

# Balanced risk management for profitability
max_holding_period = 12     # Longer holding period for trends
target_capital_utilization = 0.8  # Target 80% capital utilization
base_position_size = 0.6   # Larger base position size
max_position_per_instrument = 0.3  # Max 30% of capital per instrument
daily_loss_limit = 1000     # Higher daily loss limit
neutrality_tolerance = 0.1  # More relaxed market neutrality

def calculate_simple_momentum(prices, window=lookback_short):
    """Simple momentum calculation"""
    if len(prices) < window + 1:
        return 0.0
    return (prices[-1] / prices[-window] - 1)

def calculate_simple_mean_reversion(prices, window=lookback_medium):
    """Simple mean reversion z-score"""
    if len(prices) < window:
        return 0.0
    
    recent_prices = prices[-window:]
    z_score = (prices[-1] - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
    return z_score

def calculate_spread_zscore(prices_a, prices_b, lookback=lookback_medium):
    """Calculate z-score of the spread between two price series"""
    if len(prices_a) < lookback + 1 or len(prices_b) < lookback + 1:
        return 0.0, 0.0, 0.0
    
    # Calculate the spread (price ratio for simplicity)
    spread = prices_a / prices_b
    
    if len(spread) < lookback:
        return 0.0, 0.0, 0.0
    
    # Calculate rolling statistics
    recent_spread = spread[-lookback:]
    mean_spread = np.mean(recent_spread)
    std_spread = np.std(recent_spread)
    
    if std_spread == 0:
        return 0.0, 0.0, 0.0
    
    current_spread = spread[-1]
    z_score = (current_spread - mean_spread) / std_spread
    
    return z_score, current_spread, mean_spread

def find_best_pairs(prices, min_corr=correlation_min):
    """Find highly correlated pairs for trading"""
    n_instruments, n_days = prices.shape
    pairs = []
    
    if n_days < correlation_lookback:
        return pairs
    
    # Calculate correlations between all pairs (limited to avoid over-trading)
    for i in range(min(30, n_instruments)):  # Check more instruments
        for j in range(i + 1, min(30, n_instruments)):
            corr = calculate_correlation(prices[i, :], prices[j, :])
            if abs(corr) >= min_corr:
                pairs.append((i, j, abs(corr)))
    
    # If no pairs meet the threshold, lower it and try again
    if len(pairs) == 0:
        for i in range(min(25, n_instruments)):
            for j in range(i + 1, min(25, n_instruments)):
                corr = calculate_correlation(prices[i, :], prices[j, :])
                if abs(corr) >= 0.3:  # Much lower threshold
                    pairs.append((i, j, abs(corr)))
    
    # Sort by correlation strength and take only best pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top 8 pairs to get reasonable trading
    return pairs[:8]

def calculate_position_size_conservative(signal_strength, max_shares, volatility=1.0):
    """Conservative position sizing"""
    # Much smaller position sizes
    base_fraction = min(abs(signal_strength) / 4.0, 0.5)  # Cap at 50%
    position_fraction = base_fraction * base_position_size
    
    # Volatility adjustment (reduce size for high volatility)
    vol_adj = 1.0 / (1.0 + volatility)
    position_fraction *= vol_adj
    
    return position_fraction * max_shares

def calculate_volatility(prices, window=20):
    """Calculate volatility"""
    if len(prices) < window + 1:
        return 0.02
    
    returns = np.diff(np.log(prices[-window:]))
    return np.std(returns) * np.sqrt(252)

def calculate_correlation(prices_a, prices_b, lookback=correlation_lookback):
    """Calculate rolling correlation between two price series"""
    actual_lookback = min(len(prices_a), len(prices_b), lookback, 30)  # Use available data, max 30
    
    if actual_lookback < 8:  # Minimum 8 data points
        return 0.0
    
    # Calculate returns for correlation
    returns_a = np.diff(np.log(prices_a[-actual_lookback:]))
    returns_b = np.diff(np.log(prices_b[-actual_lookback:]))
    
    if len(returns_a) < 5:
        return 0.0
    
    correlation = np.corrcoef(returns_a, returns_b)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0




def getMyPosition(prcSoFar):
    """
    Simple, Profitable Trading Strategy
    Focus on high-quality pairs trading with strict risk management
    """
    global prev_pos, trade_count, pair_spreads, active_pairs, position_holding_time, recent_pnl
    
    # Get dimensions
    n_instruments, n_days = prcSoFar.shape
    
    # Initialize positions
    positions = np.zeros(n_instruments)
    
    # Need sufficient history
    if n_days < 10:  # Start trading much earlier
        return positions
    
    current_prices = prcSoFar[:, -1]
    max_shares = max_notional / current_prices
    
    # Update position holding time
    position_holding_time += 1
    
    # STRATEGY: Conservative Pairs Trading Only
    best_pairs = find_best_pairs(prcSoFar)
    
    if len(best_pairs) == 0:
        # If no good pairs, use aggressive momentum on individual instruments
        for i in range(n_instruments):  # Trade ALL instruments
            momentum = calculate_simple_momentum(prcSoFar[i, :])
            mean_revert = calculate_simple_mean_reversion(prcSoFar[i, :])
            
            # Much more aggressive entry - trade on any signal
            signal_strength = 0.0
            
            # Momentum component
            if abs(momentum) > momentum_threshold:
                signal_strength += momentum
            
            # Mean reversion component (smaller weight)
            if abs(mean_revert) > mean_revert_threshold:
                signal_strength += -mean_revert * 0.2
            
            # Enter position if any signal
            if abs(signal_strength) > 0.005:  # Very low threshold
                volatility = calculate_volatility(prcSoFar[i, :])
                pos_size = signal_strength * base_position_size * max_shares[i]
                # Volatility adjustment
                vol_adj = 1.0 / (1.0 + volatility)
                positions[i] = pos_size * vol_adj
        
        # Apply conservative limits
        positions = np.clip(positions, -max_shares * max_position_per_instrument, 
                          max_shares * max_position_per_instrument)
    else:
        # Pairs trading - only trade the best 3 pairs to avoid over-trading
        for i, j, correlation in best_pairs[:3]:
            pair_key = f"{min(i, j)}_{max(i, j)}"
            
            # Calculate spread z-score
            z_score, current_spread, mean_spread = calculate_spread_zscore(prcSoFar[i, :], prcSoFar[j, :])
            
            # Store spread history
            if pair_key not in pair_spreads:
                pair_spreads[pair_key] = []
            pair_spreads[pair_key].append(current_spread)
            
            # Current positions
            current_pos_i = prev_pos[i] if len(prev_pos) > i else 0
            current_pos_j = prev_pos[j] if len(prev_pos) > j else 0
            
            # Exit logic - be more conservative about exits
            should_exit = False
            if (current_pos_i != 0 or current_pos_j != 0):
                if (abs(z_score) <= pairs_z_exit or 
                    position_holding_time[i] >= max_holding_period or 
                    position_holding_time[j] >= max_holding_period):
                    should_exit = True
            
            if should_exit:
                positions[i] = 0
                positions[j] = 0
                position_holding_time[i] = 0
                position_holding_time[j] = 0
                if pair_key in active_pairs:
                    active_pairs.remove(pair_key)
                continue
            
            # Entry logic - only enter with very strong signals
            if abs(z_score) >= pairs_z_entry and current_pos_i == 0 and current_pos_j == 0:
                # Calculate volatilities for risk adjustment
                vol_i = calculate_volatility(prcSoFar[i, :])
                vol_j = calculate_volatility(prcSoFar[j, :])
                
                # Conservative position sizing
                signal_strength = min(abs(z_score) / 3.0, 0.5)  # Cap signal strength
                pos_size_i = signal_strength * base_position_size * max_shares[i] / (vol_i + 0.01)
                pos_size_j = signal_strength * base_position_size * max_shares[j] / (vol_j + 0.01)
                
                # Determine direction
                if z_score > 0:  # i is expensive relative to j
                    positions[i] = -pos_size_i * 0.5  # Conservative sizing
                    positions[j] = pos_size_j * 0.5
                else:  # i is cheap relative to j
                    positions[i] = pos_size_i * 0.5
                    positions[j] = -pos_size_j * 0.5
                
                active_pairs.add(pair_key)
                position_holding_time[i] = 0
                position_holding_time[j] = 0
            
            # Maintain existing positions
            elif current_pos_i != 0 or current_pos_j != 0:
                positions[i] = current_pos_i
                positions[j] = current_pos_j
    
    # Final fallback: If still no positions, ensure some minimal trading
    if np.sum(np.abs(positions)) == 0:
        # Simple momentum strategy on select instruments
        for i in range(0, min(20, n_instruments), 4):  # Every 4th instrument up to 20
            if len(prcSoFar[i, :]) >= lookback_short:
                momentum = calculate_simple_momentum(prcSoFar[i, :], lookback_short)
                if abs(momentum) > 0.01:  # Very low threshold (1%)
                    pos_size = momentum * base_position_size * 0.3 * max_shares[i]
                    positions[i] = pos_size
    
    # Apply position limits per instrument
    max_position_limit = max_shares * max_position_per_instrument
    positions = np.clip(positions, -max_position_limit, max_position_limit)
    
    # AGGRESSIVE CAPITAL UTILIZATION
    position_notional = np.abs(positions) * current_prices
    total_position_notional = np.sum(position_notional)
    current_utilization = total_position_notional / total_capital
    
    # If utilization is too low, scale up aggressively
    if current_utilization < target_capital_utilization * 0.6:
        if current_utilization > 0:
            scale_factor = min(target_capital_utilization / current_utilization, 2.5)
            positions = positions * scale_factor
        else:
            # If no positions at all, create basic momentum positions
            for i in range(0, n_instruments, 2):  # Every other instrument
                if len(prcSoFar[i, :]) >= 10:
                    momentum = calculate_simple_momentum(prcSoFar[i, :])
                    if abs(momentum) > 0.005:  # Very low threshold
                        pos_size = momentum * base_position_size * max_shares[i] * 0.5
                        positions[i] = pos_size
        
        # Re-apply limits after scaling
        positions = np.clip(positions, -max_position_limit, max_position_limit)
    
    # Relaxed market neutrality for higher returns
    positions = ensure_market_neutrality_relaxed(positions, current_prices)
    
    # Less smoothing for more responsive trading
    if trade_count > 0:
        alpha = 0.7  # More responsive (70% new, 30% old)
        smoothed_positions = alpha * positions + (1 - alpha) * prev_pos[:len(positions)]
    else:
        smoothed_positions = positions
    
    # Round and finalize
    final_positions = np.round(smoothed_positions)
    
    # Final risk check - ensure no position is too large
    max_abs_position = np.max(np.abs(final_positions * current_prices))
    if max_abs_position > max_notional * max_position_per_instrument:
        scale_down = (max_notional * max_position_per_instrument) / max_abs_position
        final_positions = final_positions * scale_down
    
    # Update tracking
    trade_count += 1
    prev_pos = final_positions.copy()
    
    return final_positions

def ensure_market_neutrality_relaxed(positions, current_prices):
    """Relaxed market neutrality to allow net exposure for profits"""
    long_exposure = np.sum(np.maximum(positions * current_prices, 0))
    short_exposure = np.sum(np.minimum(positions * current_prices, 0))
    total_exposure = long_exposure + abs(short_exposure)
    
    if total_exposure == 0:
        return positions
    
    net_exposure = (long_exposure + short_exposure) / total_exposure
    
    # Allow significant net exposure for profit opportunities
    if abs(net_exposure) > neutrality_tolerance:
        adjustment_factor = 1.0 - (abs(net_exposure) - neutrality_tolerance) * 0.3  # Gentle adjustment
        
        if net_exposure > 0:
            positions = np.where(positions > 0, positions * adjustment_factor, positions)
        else:
            positions = np.where(positions < 0, positions * adjustment_factor, positions)
    
    return positions

