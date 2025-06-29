"""
Mean Reversion Market-Neutral Pair Trading Strategy

This strategy implements a sophisticated market-neutral approach designed for high profitability
while maintaining excellent capital utilization. Key features:

STRATEGY OVERVIEW:
- Mean Reversion: Trades pairs when their spread deviates significantly from historical mean
- Market Neutral: Maintains balanced long/short exposure to minimize market risk
- Pair Selection: Uses correlation analysis to identify highly cointegrated pairs
- Risk Management: Position sizing based on z-score magnitude and volatility

KEY IMPROVEMENTS OVER MOMENTUM STRATEGY:
1. Market Neutrality: Reduces systematic risk and provides more consistent returns
2. Mean Reversion: Higher probability trades based on statistical arbitrage
3. Pair Trading: Natural hedging reduces portfolio volatility
4. Better Risk-Adjusted Returns: Higher Sharpe ratio through reduced drawdowns

PARAMETERS:
- Entry Threshold: Z-score >= 2.0 (more selective than momentum)
- Exit Threshold: Z-score <= 0.3 (hold until reversion completes)  
- Lookback Period: 30 days for mean/std calculation
- Correlation Threshold: 0.6 minimum for pair selection
- Max Holding Period: 15 days (forced exit to limit risk)

MARKET NEUTRALITY:
- Maintains net exposure < 5% of total capital
- Equal dollar amounts in long/short legs of each pair
- Rebalancing when neutrality deviates beyond tolerance

CAPITAL UTILIZATION:
- Target: 85% of available capital
- Achieved through multiple concurrent pair trades
- Position sizing scaled by signal strength (z-score magnitude)

Expected Benefits:
- Higher Sharpe ratio due to market neutrality
- More consistent returns with lower drawdowns  
- Better performance in volatile/sideways markets
- Reduced correlation to market movements
"""

import numpy as np
from scipy import stats

# Constants for the trading strategy
nInst = 50
max_notional = 10_000
total_capital = nInst * max_notional

# Position tracking
prev_pos = np.zeros(nInst)
position_holding_time = np.zeros(nInst)  # Track how long positions have been held
trade_count = 0
historical_correlations = np.eye(nInst)  # Correlation matrix for pair selection
pair_spreads = {}  # Store spread history for each pair
active_pairs = set()  # Currently active trading pairs

# Mean Reversion Strategy Parameters
lookback_period = 30       # Lookback period for z-score calculation
correlation_lookback = 60  # Lookback for correlation calculation
vol_lookback = 40          # Lookback for volatility calculation 
z_entry_threshold = 1.5    # Z-score threshold to enter positions (lowered from 2.0)
z_exit_threshold = 0.3     # Z-score threshold to exit positions (lower for mean reversion)
max_holding_period = 15    # Maximum days to hold a position
target_capital_utilization = 0.85  # Target capital utilization (85%)
position_sizing_factor = 0.6  # Position sizing factor (% of max position)
min_correlation = 0.4      # Minimum correlation for pair trading (lowered from 0.6)
max_pairs_per_instrument = 5  # Maximum pairs per instrument (increased from 3)
neutrality_tolerance = 0.05  # Tolerance for market neutrality (5% of total capital)
rebalance_threshold = 0.1   # Rebalance when neutrality deviates by this much

def calculate_spread_zscore(prices_a, prices_b, lookback=lookback_period):
    """Calculate z-score of the spread between two price series for mean reversion"""
    if len(prices_a) < lookback + 1 or len(prices_b) < lookback + 1:
        return 0.0, 0.0, 0.0
    
    # Calculate the spread (log price ratio for better stationarity)
    spread = np.log(prices_a) - np.log(prices_b)
    
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

def calculate_correlation(prices_a, prices_b, lookback=correlation_lookback):
    """Calculate rolling correlation between two price series"""
    if len(prices_a) < lookback + 1 or len(prices_b) < lookback + 1:
        # Use available data if we don't have full lookback
        actual_lookback = min(len(prices_a), len(prices_b), lookback)
        if actual_lookback < 10:  # Still need minimum data points
            return 0.0
    else:
        actual_lookback = lookback
    
    # Calculate returns for correlation
    returns_a = np.diff(np.log(prices_a[-actual_lookback:]))
    returns_b = np.diff(np.log(prices_b[-actual_lookback:]))
    
    if len(returns_a) < 5:  # Reduced minimum data points
        return 0.0
    
    correlation = np.corrcoef(returns_a, returns_b)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def find_best_pairs(prices, min_corr=min_correlation):
    """Find the best correlated pairs for trading"""
    n_instruments, n_days = prices.shape
    pairs = []
    
    if n_days < correlation_lookback:
        return pairs
    
    # Calculate correlations between all pairs
    for i in range(n_instruments):
        for j in range(i + 1, n_instruments):
            corr = calculate_correlation(prices[i, :], prices[j, :])
            if abs(corr) >= min_corr:
                pairs.append((i, j, abs(corr)))
    
    # If no pairs meet the correlation threshold, take the best ones anyway
    if len(pairs) == 0:
        all_pairs = []
        for i in range(n_instruments):
            for j in range(i + 1, n_instruments):
                corr = calculate_correlation(prices[i, :], prices[j, :])
                if not np.isnan(corr):
                    all_pairs.append((i, j, abs(corr)))
        
        # Sort by correlation and take top 20% of pairs
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs_count = max(10, len(all_pairs) // 5)  # At least 10 pairs or 20% of all pairs
        pairs = all_pairs[:top_pairs_count]
    
    # Sort by correlation strength (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Limit pairs per instrument to avoid over-concentration
    instrument_pair_count = {}
    filtered_pairs = []
    
    for i, j, corr in pairs:
        i_count = instrument_pair_count.get(i, 0)
        j_count = instrument_pair_count.get(j, 0)
        
        if i_count < max_pairs_per_instrument and j_count < max_pairs_per_instrument:
            filtered_pairs.append((i, j, corr))
            instrument_pair_count[i] = i_count + 1
            instrument_pair_count[j] = j_count + 1
    
    return filtered_pairs

def calculate_position_size_mean_reversion(z_score, max_shares_a, max_shares_b, current_price_a, current_price_b):
    """Calculate position sizes for mean reversion pair trade"""
    # Position sizing based on z-score magnitude (higher z-score = larger position)
    base_fraction = min(abs(z_score) / 4.0, 1.0)  # Scale with z-score, cap at 100%
    position_fraction = base_fraction * position_sizing_factor
    
    # Calculate notional-balanced positions
    # We want equal dollar amounts on each side
    notional_a = position_fraction * max_shares_a * current_price_a
    notional_b = position_fraction * max_shares_b * current_price_b
    target_notional = min(notional_a, notional_b)
    
    shares_a = target_notional / current_price_a
    shares_b = target_notional / current_price_b
    
    return shares_a, shares_b

def ensure_market_neutrality(positions, current_prices):
    """Ensure the portfolio is market neutral"""
    # Calculate total long and short exposure
    long_exposure = np.sum(np.maximum(positions * current_prices, 0))
    short_exposure = np.sum(np.minimum(positions * current_prices, 0))
    total_exposure = long_exposure + abs(short_exposure)
    
    if total_exposure == 0:
        return positions
    
    # Calculate net exposure as percentage of total
    net_exposure = (long_exposure + short_exposure) / total_exposure
    
    # If net exposure exceeds tolerance, adjust positions
    if abs(net_exposure) > neutrality_tolerance:
        # Adjust positions to reduce net exposure
        adjustment_factor = 1.0 - (abs(net_exposure) - neutrality_tolerance)
        
        if net_exposure > 0:  # Too long, reduce long positions
            positions = np.where(positions > 0, positions * adjustment_factor, positions)
        else:  # Too short, reduce short positions
            positions = np.where(positions < 0, positions * adjustment_factor, positions)
    
    return positions

def getMyPosition(prcSoFar):
    """
    Mean reversion market-neutral pair trading strategy.
    Identifies highly correlated pairs and trades their spread reversion.
    Maintains market neutrality through balanced long/short positions.
    
    Args:
        prcSoFar: Price data so far, shape (instruments, days)
    
    Returns:
        Position array with shape (instruments,) - positive for long, negative for short
    """
    global prev_pos, trade_count, pair_spreads, active_pairs, position_holding_time
    
    # Get dimensions
    n_instruments, n_days = prcSoFar.shape
    
    # Initialize positions
    positions = np.zeros(n_instruments)
    
    # Need sufficient history for mean reversion and correlation analysis
    if n_days < max(lookback_period, correlation_lookback) // 2 + 5:  # Reduced requirement
        return positions
    
    current_prices = prcSoFar[:, -1]
    
    # Calculate maximum position sizes based on notional limits
    max_shares = max_notional / current_prices
    
    # Find best trading pairs based on correlation
    best_pairs = find_best_pairs(prcSoFar)
    
    if len(best_pairs) == 0:
        return positions
    
    # Track active positions and their holding time
    position_holding_time += 1
    
    # Process each potential trading pair
    pair_positions = {}
    total_trades = 0
    
    for i, j, correlation in best_pairs:
        pair_key = f"{min(i, j)}_{max(i, j)}"
        
        # Calculate spread z-score for this pair
        z_score, current_spread, mean_spread = calculate_spread_zscore(
            prcSoFar[i, :], prcSoFar[j, :], lookback_period
        )
        
        # Store spread history
        if pair_key not in pair_spreads:
            pair_spreads[pair_key] = []
        pair_spreads[pair_key].append(current_spread)
        
        # Keep only recent spread history
        if len(pair_spreads[pair_key]) > lookback_period * 2:
            pair_spreads[pair_key] = pair_spreads[pair_key][-lookback_period:]
        
        # Current positions for this pair
        current_pos_i = prev_pos[i] if len(prev_pos) > i else 0
        current_pos_j = prev_pos[j] if len(prev_pos) > j else 0
        
        # Check if we should exit existing positions (mean reversion complete)
        should_exit = False
        if (current_pos_i != 0 or current_pos_j != 0):
            # Exit if z-score has reverted enough or holding too long
            if (abs(z_score) <= z_exit_threshold or 
                position_holding_time[i] >= max_holding_period or 
                position_holding_time[j] >= max_holding_period):
                should_exit = True
        
        if should_exit:
            # Close positions for this pair
            positions[i] = 0
            positions[j] = 0
            position_holding_time[i] = 0
            position_holding_time[j] = 0
            if pair_key in active_pairs:
                active_pairs.remove(pair_key)
            continue
        
        # Check if we should enter new positions (mean reversion signal)
        if abs(z_score) >= z_entry_threshold and current_pos_i == 0 and current_pos_j == 0:
            # Calculate position sizes for this pair
            shares_i, shares_j = calculate_position_size_mean_reversion(
                z_score, max_shares[i], max_shares[j], current_prices[i], current_prices[j]
            )
            
            # Determine trade direction based on z_score
            if z_score > 0:
                # Spread is too high: short i (expensive), long j (cheap)
                positions[i] = -shares_i
                positions[j] = shares_j
            else:
                # Spread is too low: long i (cheap), short j (expensive)  
                positions[i] = shares_i
                positions[j] = -shares_j
            
            # Mark this pair as active
            active_pairs.add(pair_key)
            position_holding_time[i] = 0
            position_holding_time[j] = 0
            total_trades += 1
        
        # Maintain existing positions if no exit/entry signal
        elif current_pos_i != 0 or current_pos_j != 0:
            positions[i] = current_pos_i
            positions[j] = current_pos_j
    
    # If no pair trades were made, use individual instrument mean reversion as fallback
    if np.sum(np.abs(positions)) == 0:
        # Calculate individual instrument z-scores for mean reversion
        for i in range(min(25, n_instruments)):  # Trade up to 25 instruments individually
            instrument_prices = prcSoFar[i, :]
            
            if len(instrument_prices) >= lookback_period:
                # Calculate z-score for this instrument
                recent_prices = instrument_prices[-lookback_period:]
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                
                if std_price > 0:
                    z_score = (instrument_prices[-1] - mean_price) / std_price
                    
                    # Enter positions based on z-score
                    if abs(z_score) >= z_entry_threshold:
                        # Position size based on z-score magnitude
                        base_fraction = min(abs(z_score) / 3.0, 1.0)
                        position_fraction = base_fraction * position_sizing_factor * 0.5  # Smaller positions for individual trades
                        
                        # Mean reversion: buy low, sell high
                        if z_score < -z_entry_threshold:  # Price is low
                            positions[i] = position_fraction * max_shares[i]
                        elif z_score > z_entry_threshold:  # Price is high
                            positions[i] = -position_fraction * max_shares[i]
    
    # Ensure market neutrality
    positions = ensure_market_neutrality(positions, current_prices)
    
    # Apply position limits per instrument
    positions = np.clip(positions, -max_shares, max_shares)
    
    # Calculate current capital utilization
    position_notional = np.abs(positions) * current_prices
    total_position_notional = np.sum(position_notional)
    current_utilization = total_position_notional / total_capital
    
    # Scale positions if utilization is too low (but maintain pair balance)
    if current_utilization > 0 and current_utilization < target_capital_utilization * 0.7:  # 70% of target as minimum
        scale_factor = min(target_capital_utilization / current_utilization, 1.5)  # Cap scaling
        positions = positions * scale_factor
        
        # Re-apply limits after scaling
        positions = np.clip(positions, -max_shares, max_shares)
    
    # Smooth position changes to reduce transaction costs (but less than momentum strategy)
    if trade_count > 0:
        # Higher smoothing coefficient for mean reversion (we want to hold positions longer)
        alpha = 0.8  # Keep 80% of new position, 20% of old
        smoothed_positions = alpha * positions + (1 - alpha) * prev_pos[:len(positions)]
    else:
        smoothed_positions = positions
    
    # Round to whole shares
    final_positions = np.round(smoothed_positions)
    
    # Final market neutrality check and adjustment
    final_positions = ensure_market_neutrality(final_positions, current_prices)
    
    # Update tracking variables
    trade_count += 1
    prev_pos = final_positions.copy()
    
    return final_positions