import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Global state
nInst = 50
currentPos = np.zeros(nInst)
signal_history = deque(maxlen=3)
prev_confirmed_signals = np.zeros(nInst)

# Parameters
FAST_PERIOD = 7
SLOW_PERIOD = 30
SIGNAL_CONFIRMATION_DAYS = 2
CAPITAL_PER_INSTRUMENT = 10000.0
MA_THRESHOLD = 0.002
VOLATILITY_WINDOW = 20

def calculate_true_range(prices):
    if prices.shape[1] < VOLATILITY_WINDOW + 1:
        return np.zeros(nInst)
    high = prices[:, -VOLATILITY_WINDOW:].max(axis=1)
    low = prices[:, -VOLATILITY_WINDOW:].min(axis=1)
    prev_close = prices[:, -VOLATILITY_WINDOW-1:-1][:, -1]
    range1 = high - low
    range2 = np.abs(high - prev_close)
    range3 = np.abs(low - prev_close)
    return np.maximum(np.maximum(range1, range2), range3)

def calculate_moving_average_signal(prices):
    if prices.shape[1] < SLOW_PERIOD:
        return np.zeros(nInst)
    fast_average = prices[:, -FAST_PERIOD:].mean(axis=1)
    slow_average = prices[:, -SLOW_PERIOD:].mean(axis=1)
    return (fast_average - slow_average) / slow_average

def generate_trading_signals(ma_signal, volatility_measure):
    signals = np.zeros(nInst, dtype=int)
    signals[ma_signal > MA_THRESHOLD] = 1
    signals[ma_signal < -MA_THRESHOLD] = -1
    if np.sum(volatility_measure > 0) > 0:
        volatility_median = np.median(volatility_measure[volatility_measure > 0])
        volatility_filter = (volatility_measure > volatility_median).astype(int)
        signals = signals * volatility_filter
    return signals

def validate_signal_persistence(raw_signals):
    global signal_history, prev_confirmed_signals
    signal_history.append(raw_signals)
    if len(signal_history) < SIGNAL_CONFIRMATION_DAYS:
        return prev_confirmed_signals.copy()
    signal_stack = np.stack(signal_history, axis=0)
    consistency_check = np.all(signal_stack == raw_signals, axis=0)
    validated_signals = np.where(consistency_check, raw_signals, prev_confirmed_signals)
    prev_confirmed_signals = validated_signals.copy()
    return validated_signals

def calculate_position_sizes(validated_signals, prices, ma_signal, volatility_measure):
    current_prices = prices[:, -1]
    maximum_shares = np.floor(CAPITAL_PER_INSTRUMENT / current_prices).astype(int)
    base_positions = validated_signals * maximum_shares
    signal_magnitude = np.abs(ma_signal)
    strength_factor = np.clip(signal_magnitude / MA_THRESHOLD, 0.5, 2.0)
    adjusted_positions = (base_positions * strength_factor).astype(int)
    if np.sum(volatility_measure > 0) > 0:
        high_volatility_threshold = np.percentile(volatility_measure[volatility_measure > 0], 75)
        high_volatility_mask = volatility_measure > high_volatility_threshold
        adjusted_positions[high_volatility_mask] = (adjusted_positions[high_volatility_mask] * 0.7).astype(int)
    final_positions = np.clip(adjusted_positions, -maximum_shares, maximum_shares)
    return final_positions

def getMyPosition(prcSoFar):
    global currentPos
    n_inst, n_days = prcSoFar.shape
    if n_days < SLOW_PERIOD + 1:
        return np.zeros(n_inst)
    ma_crossover_signal = calculate_moving_average_signal(prcSoFar)
    volatility_indicator = calculate_true_range(prcSoFar)
    raw_trading_signals = generate_trading_signals(ma_crossover_signal, volatility_indicator)
    confirmed_trading_signals = validate_signal_persistence(raw_trading_signals)
    target_positions = calculate_position_sizes(confirmed_trading_signals, prcSoFar, ma_crossover_signal, volatility_indicator)

    active_positions = np.sum(np.abs(target_positions) > 0)
    maximum_allowed_positions = min(25, nInst // 2)
    if active_positions > maximum_allowed_positions:
        signal_quality_score = np.abs(ma_crossover_signal) * (volatility_indicator > np.median(volatility_indicator[volatility_indicator > 0])).astype(float)
        quality_rankings = np.argsort(signal_quality_score)
        low_quality_positions = quality_rankings[:-maximum_allowed_positions]
        target_positions[low_quality_positions] = 0

    if n_days > 5:
        log_price_series = np.log(np.maximum(prcSoFar, 1e-8))
        short_term_returns = np.diff(log_price_series, axis=1)[:, -3:]
        momentum_indicator = np.sum(short_term_returns, axis=1)
        
        for instrument in range(n_inst):
            if target_positions[instrument] != 0:
                position_direction = np.sign(target_positions[instrument])
                momentum_direction = np.sign(momentum_indicator[instrument])
                momentum_magnitude = np.abs(momentum_indicator[instrument])
                
                if momentum_magnitude > 0.001:  # Only adjust if momentum is significant
                    # Scale adjustment proportionally to momentum magnitude
                    # Base scaling factor: momentum * 100 gives us a reasonable range
                    momentum_scale = momentum_magnitude * 100
                    
                    if position_direction == momentum_direction:
                        # Boost aligned positions: 1.0 + (0.02 to 0.15) based on momentum
                        boost_factor = 1.0 + np.clip(momentum_scale * 0.02, 0.02, 0.15)
                        target_positions[instrument] = int(target_positions[instrument] * boost_factor)
                    else:
                        # Reduce opposing positions: 1.0 - (0.02 to 0.12) based on momentum
                        reduction_factor = 1.0 - np.clip(momentum_scale * 0.015, 0.02, 0.12)
                        target_positions[instrument] = int(target_positions[instrument] * reduction_factor)

    current_prices = prcSoFar[:, -1]
    maximum_shares_allowed = np.floor(CAPITAL_PER_INSTRUMENT / current_prices).astype(int)
    target_positions = np.clip(target_positions, -maximum_shares_allowed, maximum_shares_allowed)
    currentPos = target_positions.copy()
    return currentPos
