"""
回调形态末期的窄幅交易区间识别模块

窄幅交易区间特征:
1. 价格在回调后进入窄幅震荡
2. 价格波动范围逐渐缩小
3. 成交量明显萎缩
4. 通常出现在回调末期，预示着行情即将恢复原趋势
5. 突破方向通常与主趋势一致

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_narrow_range_after_retracement(df: pd.DataFrame, lookback_period: int = 30,
                                         min_retracement_pct: float = 0.03,
                                         narrow_range_bars: int = 8,
                                         volume_threshold: float = 0.8) -> List[Dict]:
    """
    检测回调形态末期的窄幅交易区间
    
    参数:
        df: 包含OHLCV数据的DataFrame
        lookback_period: 用于确定主趋势的回溯期K线数
        min_retracement_pct: 最小回调幅度（相对于价格的百分比）
        narrow_range_bars: 窄幅区间的最小K线数
        volume_threshold: 成交量萎缩阈值（相对于回调前的平均成交量）
    
    返回:
        List[Dict]: 检测到的窄幅交易区间列表，每个字典包含形态详情
    """
    if len(df) < lookback_period + narrow_range_bars:
        return []
    
    results = []
    
    # 计算每个窗口的波动范围
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close']
    
    # 计算移动平均波动范围（20日）
    df['avg_range_20'] = df['range'].rolling(window=20).mean()
    
    # 遍历可能的窄幅区间起点
    for i in range(lookback_period, len(df) - narrow_range_bars):
        # 检查前期趋势
        trend_section = df.iloc[i-lookback_period:i]
        trend_start_price = trend_section['close'].iloc[0]
        trend_end_price = trend_section['close'].iloc[-1]
        
        # 确定趋势方向
        if trend_end_price > trend_start_price * 1.1:  # 上涨超过10%
            trend = 'up'
        elif trend_end_price < trend_start_price * 0.9:  # 下跌超过10%
            trend = 'down'
        else:
            continue  # 无明显趋势
        
        # 检查是否有足够的回调
        if trend == 'up':
            # 上升趋势中的回调（下跌）
            max_price = trend_section['high'].max()
            current_price = df['close'].iloc[i]
            retracement = (max_price - current_price) / max_price
            
            if retracement < min_retracement_pct:
                continue  # 回调不足
        else:
            # 下降趋势中的回调（上涨）
            min_price = trend_section['low'].min()
            current_price = df['close'].iloc[i]
            retracement = (current_price - min_price) / min_price
            
            if retracement < min_retracement_pct:
                continue  # 回调不足
        
        # 检查后续是否形成窄幅区间
        narrow_section = df.iloc[i:i+narrow_range_bars]
        
        # 计算窄幅区间的特征
        avg_range = narrow_section['range_pct'].mean()
        prev_avg_range = df.iloc[i-narrow_range_bars:i]['range_pct'].mean()
        
        # 窄幅条件：当前波动范围小于前期波动范围的70%
        if avg_range > prev_avg_range * 0.7:
            continue
        
        # 检查窄幅区间内的波动是否逐渐缩小
        range_trend = linregress(range(len(narrow_section)), narrow_section['range_pct'])[0]
        if range_trend >= 0:
            continue  # 波动范围没有缩小
        
        # 检查成交量是否萎缩
        avg_volume = narrow_section['volume'].mean()
        prev_avg_volume = trend_section['volume'].tail(narrow_range_bars).mean()
        
        if avg_volume > prev_avg_volume * volume_threshold:
            continue  # 成交量没有足够萎缩
        
        # 计算形态强度
        strength = _calculate_narrow_range_strength(
            retracement, avg_range / prev_avg_range, 
            avg_volume / prev_avg_volume, range_trend
        )
        
        # 计算关键价格水平
        resistance = narrow_section['high'].max()
        support = narrow_section['low'].min()
        
        # 保存检测到的形态
        results.append({
            'type': 'narrow_range_after_retracement',
            'start': df.index[i],
            'end': df.index[i + narrow_range_bars - 1],
            'trend': trend,
            'retracement': retracement,
            'avg_range': avg_range,
            'prev_avg_range': prev_avg_range,
            'range_reduction': 1 - (avg_range / prev_avg_range),
            'volume_reduction': 1 - (avg_volume / prev_avg_volume),
            'strength': strength,
            'resistance': resistance,
            'support': support,
            'expected_breakout': trend  # 预期突破方向与主趋势一致
        })
        
        # 跳过重叠的区间
        i += narrow_range_bars
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _calculate_narrow_range_strength(retracement: float, range_ratio: float,
                                    volume_ratio: float, range_trend: float) -> float:
    """
    计算窄幅交易区间的强度
    
    参数:
        retracement: 回调幅度
        range_ratio: 当前波动范围/前期波动范围
        volume_ratio: 当前成交量/前期成交量
        range_trend: 波动范围的线性趋势
        
    返回:
        float: 形态强度评分 (0-1)
    """
    # 回调幅度评分 (0.3权重)
    # 理想的回调幅度为8%-15%
    retracement_score = (1.0 - min(abs(retracement - 0.1) / 0.1, 1.0)) * 0.3
    
    # 波动范围缩小评分 (0.3权重)
    # 理想情况下，波动范围缩小到前期的30%-50%
    range_score = (1.0 - min(range_ratio, 1.0)) * 0.3
    
    # 成交量萎缩评分 (0.2权重)
    # 理想情况下，成交量萎缩到前期的50%或更低
    volume_score = (1.0 - min(volume_ratio, 1.0)) * 0.2
    
    # 波动趋势评分 (0.2权重)
    # 波动范围应该有明显的下降趋势
    trend_score = min(abs(range_trend) / 0.01, 1.0) * 0.2 if range_trend < 0 else 0
    
    # 总分
    return retracement_score + range_score + volume_score + trend_score


def is_narrow_range_breakout(df: pd.DataFrame, pattern: Dict) -> Dict:
    """
    检查窄幅交易区间是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 窄幅交易区间形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    pattern_end_idx = df.index.get_loc(pattern['end'])
    
    # 确保有后续数据
    if pattern_end_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 获取形态的阻力位和支撑位
    resistance = pattern['resistance']
    support = pattern['support']
    
    # 计算区间高度
    range_height = resistance - support
    
    # 检查后续3根K线是否突破
    post_pattern = df.iloc[pattern_end_idx+1:pattern_end_idx+4]
    
    # 向上突破条件
    up_breakout = post_pattern['close'].iloc[-1] > resistance and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean() * 1.5
    
    # 向下突破条件
    down_breakout = post_pattern['close'].iloc[-1] < support and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean() * 1.5
    
    expected_direction = pattern['expected_breakout']
    
    if up_breakout:
        # 计算突破强度（与预期方向一致时更强）
        strength = 0.7 + 0.3 * (1 if expected_direction == 'up' else 0)
        return {
            'breakout': True,
            'direction': 'up',
            'price': post_pattern['close'].iloc[-1],
            'target': resistance + range_height,  # 目标价格：阻力位 + 区间高度
            'stop_loss': support,
            'strength': strength,
            'matches_expectation': expected_direction == 'up'
        }
    elif down_breakout:
        # 计算突破强度（与预期方向一致时更强）
        strength = 0.7 + 0.3 * (1 if expected_direction == 'down' else 0)
        return {
            'breakout': True,
            'direction': 'down',
            'price': post_pattern['close'].iloc[-1],
            'target': support - range_height,  # 目标价格：支撑位 - 区间高度
            'stop_loss': resistance,
            'strength': strength,
            'matches_expectation': expected_direction == 'down'
        }
    else:
        return {'breakout': False}
