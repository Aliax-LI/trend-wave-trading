"""
三推楔形形态识别模块

三推楔形形态特征:
1. 价格形成三个连续的波动（三推）
2. 波动形成收敛的楔形通道
3. 上升楔形：上轨和下轨都向上倾斜，但上轨斜率较小
4. 下降楔形：上轨和下轨都向下倾斜，但下轨斜率较小
5. 成交量通常在形态发展过程中逐渐萎缩
6. 突破方向通常与楔形方向相反

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_wedge(df: pd.DataFrame, min_length: int = 15, max_length: int = 50,
                min_swings: int = 3, max_angle_diff: float = 8.0) -> List[Dict]:
    """
    检测三推楔形形态
    
    参数:
        df: 包含OHLCV数据的DataFrame
        min_length: 形态的最小长度（K线数量）
        max_length: 形态的最大长度（K线数量）
        min_swings: 最小波动次数（通常为3，即三推）
        max_angle_diff: 上下轨道角度最大差异（度）
    
    返回:
        List[Dict]: 检测到的楔形形态列表，每个字典包含形态详情
    """
    if len(df) < min_length:
        return []
    
    results = []
    
    # 寻找局部极值点（高点和低点）
    highs, lows = _find_local_extrema(df)
    
    # 如果高点或低点数量不足，无法形成楔形
    if len(highs) < min_swings or len(lows) < min_swings:
        return []
    
    # 遍历可能的楔形起点
    for start_idx in range(len(df) - min_length):
        # 尝试不同的形态长度
        for pattern_length in range(min_length, min(max_length, len(df) - start_idx)):
            pattern_end_idx = start_idx + pattern_length - 1
            pattern_section = df.iloc[start_idx:pattern_end_idx+1]
            
            # 获取形态区间内的高点和低点
            pattern_highs = [(idx, row['high']) for idx, row in pattern_section.iterrows() 
                            if idx in [h[0] for h in highs]]
            pattern_lows = [(idx, row['low']) for idx, row in pattern_section.iterrows() 
                           if idx in [l[0] for l in lows]]
                           
            # 确保有足够的高点和低点
            if len(pattern_highs) < min_swings or len(pattern_lows) < min_swings:
                continue
            
            # 计算高点和低点的趋势线
            high_indices = []
            high_values = []
            for idx, val in pattern_highs:
                if idx in df.index:
                    high_indices.append(df.index.get_loc(idx) - start_idx)
                    high_values.append(val)
            
            low_indices = []
            low_values = []
            for idx, val in pattern_lows:
                if idx in df.index:
                    low_indices.append(df.index.get_loc(idx) - start_idx)
                    low_values.append(val)
            
            # 如果点数不足以进行线性回归，则跳过
            if len(high_indices) < 2 or len(low_indices) < 2:
                continue
                
            # 计算高点趋势线
            high_slope, high_intercept, high_r_value, _, _ = linregress(high_indices, high_values)
            
            # 计算低点趋势线
            low_slope, low_intercept, low_r_value, _, _ = linregress(low_indices, low_values)
            
            # 计算趋势线角度（度）
            high_angle = np.degrees(np.arctan(high_slope))
            low_angle = np.degrees(np.arctan(low_slope))
            
            # 检查是否形成楔形（趋势线收敛）
            # 上升楔形：两条线都向上，但上轨斜率小于下轨斜率
            # 下降楔形：两条线都向下，但上轨斜率大于下轨斜率
            is_converging = (high_slope < 0 and low_slope < 0 and high_slope > low_slope) or \
                           (high_slope > 0 and low_slope > 0 and high_slope < low_slope)
            
            # 确保趋势线有明显的收敛性
            slope_diff = abs(high_slope - low_slope)
            if not is_converging or slope_diff < 0.0001:
                continue
                
            # 检查趋势线的拟合质量
            # 计算每个高点到高点趋势线的距离
            high_distances = []
            for i, (idx, price) in enumerate(pattern_highs):
                x_pos = high_indices[i]
                trend_line_value = high_slope * x_pos + high_intercept
                distance = abs(price - trend_line_value) / price
                high_distances.append(distance)
                
            # 计算每个低点到低点趋势线的距离
            low_distances = []
            for i, (idx, price) in enumerate(pattern_lows):
                x_pos = low_indices[i]
                trend_line_value = low_slope * x_pos + low_intercept
                distance = abs(price - trend_line_value) / price
                low_distances.append(distance)
                
            # 如果任何点距离趋势线太远（超过0.5%），则认为拟合质量不佳
            if max(high_distances + low_distances) > 0.005:
                continue
                
            # 检查价格是否突破了趋势线
            # 计算最后一个点的趋势线值
            last_idx = pattern_length - 1
            last_high_line_value = high_slope * last_idx + high_intercept
            last_low_line_value = low_slope * last_idx + low_intercept
            
            # 获取最后几个K线的高低点
            last_candles = pattern_section.iloc[-3:]
            
            # 如果价格已经明显突破趋势线，则跳过
            if last_candles['high'].max() > last_high_line_value * 1.005 or \
               last_candles['low'].min() < last_low_line_value * 0.995:
                continue
            
            # 检查角度差异是否在允许范围内
            angle_diff = abs(high_angle - low_angle)
            if angle_diff > max_angle_diff:
                continue
            
            # 确定楔形类型
            if high_slope > 0 and low_slope > 0:
                wedge_type = 'rising'  # 上升楔形（看跌）
            elif high_slope < 0 and low_slope < 0:
                wedge_type = 'falling'  # 下降楔形（看涨）
            else:
                continue
            
            # 计算收敛点（趋势线相交的点）
            if high_slope != low_slope:
                x_converge = (low_intercept - high_intercept) / (high_slope - low_slope)
                y_converge = high_slope * x_converge + high_intercept
                
                # 确保收敛点在合理范围内
                if x_converge < 0 or x_converge > pattern_length * 3:
                    continue
            else:
                # 平行线，不会收敛
                continue
            
            # 计算形态强度
            strength = _calculate_wedge_strength(
                high_r_value, low_r_value, angle_diff, 
                pattern_length, len(pattern_highs), len(pattern_lows)
            )
            
            # 计算成交量趋势
            volume_trend = linregress(range(len(pattern_section)), pattern_section['volume'])[0]
            volume_decreasing = volume_trend < 0
            
            # 保存检测到的形态
            results.append({
                'type': f'{wedge_type}_wedge',
                'start': df.index[start_idx],
                'end': df.index[pattern_end_idx],
                'highs': pattern_highs,
                'lows': pattern_lows,
                'high_slope': high_slope,
                'low_slope': low_slope,
                'high_angle': high_angle,
                'low_angle': low_angle,
                'converge_x': x_converge + start_idx,  # 相对于原始数据的索引
                'converge_y': y_converge,
                'strength': strength,
                'volume_decreasing': volume_decreasing,
                'expected_breakout': 'down' if wedge_type == 'rising' else 'up'
            })
            
            # 跳过重叠的区间
            start_idx += pattern_length // 2
            break
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _find_local_extrema(df: pd.DataFrame, window: int = 3) -> Tuple[List[Tuple], List[Tuple]]:
    """
    查找局部极值点（高点和低点）
    
    参数:
        df: 包含OHLCV数据的DataFrame
        window: 用于确定局部极值的窗口大小
        
    返回:
        Tuple[List[Tuple], List[Tuple]]: 高点和低点列表，每个元素为(index, value)
    """
    highs = []
    lows = []
    
    # 使用滚动窗口查找局部极值
    for i in range(window, len(df) - window):
        # 检查是否为局部高点
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            highs.append((df.index[i], df['high'].iloc[i]))
        
        # 检查是否为局部低点
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            lows.append((df.index[i], df['low'].iloc[i]))
    
    # 过滤掉太接近的极值点
    filtered_highs = []
    filtered_lows = []
    
    # 对高点进行过滤，保留区域内的最高点
    if highs:
        current_high = highs[0]
        for i in range(1, len(highs)):
            # 检查索引是否存在
            if highs[i][0] not in df.index or current_high[0] not in df.index:
                continue
            idx_diff = df.index.get_loc(highs[i][0]) - df.index.get_loc(current_high[0])
            if idx_diff > window:  # 如果与当前高点距离足够远
                filtered_highs.append(current_high)
                current_high = highs[i]
            elif highs[i][1] > current_high[1]:  # 如果新高点更高
                current_high = highs[i]
        filtered_highs.append(current_high)  # 添加最后一个高点
    
    # 对低点进行过滤，保留区域内的最低点
    if lows:
        current_low = lows[0]
        for i in range(1, len(lows)):
            # 检查索引是否存在
            if lows[i][0] not in df.index or current_low[0] not in df.index:
                continue
            idx_diff = df.index.get_loc(lows[i][0]) - df.index.get_loc(current_low[0])
            if idx_diff > window:  # 如果与当前低点距离足够远
                filtered_lows.append(current_low)
                current_low = lows[i]
            elif lows[i][1] < current_low[1]:  # 如果新低点更低
                current_low = lows[i]
        filtered_lows.append(current_low)  # 添加最后一个低点
    
    return filtered_highs, filtered_lows


def _calculate_wedge_strength(high_r_value: float, low_r_value: float, 
                             angle_diff: float, pattern_length: int,
                             num_highs: int, num_lows: int) -> float:
    """
    计算楔形形态的强度
    
    参数:
        high_r_value: 高点趋势线的R值（拟合度）
        low_r_value: 低点趋势线的R值（拟合度）
        angle_diff: 高低趋势线角度差（度）
        pattern_length: 形态长度
        num_highs: 高点数量
        num_lows: 低点数量
        
    返回:
        float: 形态强度评分 (0-1)
    """
    # 趋势线拟合度评分 (0.4权重)
    fit_score = (abs(high_r_value) + abs(low_r_value)) / 2 * 0.4
    
    # 角度差异评分 (0.3权重)
    # 理想的角度差异为3-8度
    angle_score = (1.0 - min(abs(angle_diff - 5.0) / 5.0, 1.0)) * 0.3
    
    # 波动次数评分 (0.2权重)
    # 理想的波动次数为3-5次
    swings_score = (min(num_highs, 5) + min(num_lows, 5)) / 10 * 0.2
    
    # 形态长度评分 (0.1权重)
    # 理想的形态长度为20-30天
    length_score = (1.0 - min(abs(pattern_length - 25) / 25, 1.0)) * 0.1
    
    # 总分
    return fit_score + angle_score + swings_score + length_score


def is_wedge_breakout(df: pd.DataFrame, pattern: Dict) -> Dict:
    """
    检查楔形形态是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 楔形形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    pattern_end_idx = df.index.get_loc(pattern['end'])
    
    # 确保有后续数据
    if pattern_end_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 获取形态的最后一个高点和低点
    last_high = pattern['highs'][-1][1]
    last_low = pattern['lows'][-1][1]
    
    # 计算楔形的高度
    wedge_height = last_high - last_low
    
    # 检查后续3根K线是否突破
    post_pattern = df.iloc[pattern_end_idx+1:pattern_end_idx+4]
    
    # 向上突破条件
    up_breakout = post_pattern['close'].iloc[-1] > last_high and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean()
    
    # 向下突破条件
    down_breakout = post_pattern['close'].iloc[-1] < last_low and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean()
    
    expected_direction = pattern['expected_breakout']
    
    if up_breakout:
        # 计算突破强度（与预期方向一致时更强）
        strength = 0.7 + 0.3 * (1 if expected_direction == 'up' else 0)
        return {
            'breakout': True,
            'direction': 'up',
            'price': post_pattern['close'].iloc[-1],
            'target': last_high + wedge_height,  # 目标价格：突破点 + 楔形高度
            'stop_loss': last_low,
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
            'target': last_low - wedge_height,  # 目标价格：突破点 - 楔形高度
            'stop_loss': last_high,
            'strength': strength,
            'matches_expectation': expected_direction == 'down'
        }
    else:
        return {'breakout': False}
