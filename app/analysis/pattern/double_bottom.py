"""
双底形态识别模块

双底形态特征:
1. 价格在下跌趋势中形成两个相近的低点
2. 第二个低点不应显著低于第一个低点
3. 两个低点之间有一个明显的反弹（颈线）
4. 突破颈线后，价格通常会继续上涨
5. 成交量在第二个底部通常大于第一个底部

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_double_bottom(df: pd.DataFrame, min_depth_pct: float = 0.05,
                        max_diff_pct: float = 0.03, min_height_pct: float = 0.03,
                        min_bars_between: int = 10, max_bars_between: int = 50) -> List[Dict]:
    """
    检测双底形态
    
    参数:
        df: 包含OHLCV数据的DataFrame
        min_depth_pct: 底部深度最小百分比（相对于价格）
        max_diff_pct: 两个底部之间的最大差异百分比
        min_height_pct: 颈线高度最小百分比（相对于价格）
        min_bars_between: 两个底部之间的最小K线数
        max_bars_between: 两个底部之间的最大K线数
    
    返回:
        List[Dict]: 检测到的双底形态列表，每个字典包含形态详情
    """
    if len(df) < max_bars_between + 10:  # 确保有足够的数据
        return []
    
    results = []
    
    # 寻找局部低点
    lows = _find_local_lows(df)
    if len(lows) < 2:
        return []
    
    # 遍历所有可能的底部对
    for i in range(len(lows) - 1):
        first_bottom_idx, first_bottom_price = lows[i]
        
        # 寻找符合条件的第二个底部
        for j in range(i + 1, len(lows)):
            second_bottom_idx, second_bottom_price = lows[j]
            
            # 检查两个底部之间的K线数量
            bars_between = second_bottom_idx - first_bottom_idx
            if bars_between < min_bars_between or bars_between > max_bars_between:
                continue
            
            # 检查两个底部的价格差异
            price_diff_pct = abs(second_bottom_price - first_bottom_price) / first_bottom_price
            if price_diff_pct > max_diff_pct:
                continue
            
            # 检查第二个底部不应显著低于第一个底部
            if second_bottom_price < first_bottom_price * 0.97:  # 允许3%的下探
                continue
            
            # 寻找两个底部之间的最高点（颈线）
            between_section = df.iloc[first_bottom_idx+1:second_bottom_idx]
            if between_section.empty:
                continue
                
            neckline_idx = between_section['high'].idxmax()
            neckline_price = between_section.loc[neckline_idx, 'high']
            
            # 检查颈线高度是否足够
            neckline_height_pct = (neckline_price - min(first_bottom_price, second_bottom_price)) / min(first_bottom_price, second_bottom_price)
            if neckline_height_pct < min_height_pct:
                continue
            
            # 检查底部深度是否足够
            depth_pct = neckline_height_pct
            if depth_pct < min_depth_pct:
                continue
            
            # 寻找颈线突破点
            post_section = df.loc[df.index > df.index[second_bottom_idx]]
            if post_section.empty:
                continue
                
            # 找到第一个收盘价高于颈线的K线
            breakout_bars = post_section[post_section['close'] > neckline_price]
            if breakout_bars.empty:
                continue
                
            neckline_breakout = breakout_bars.index[0]
            
            # 检查成交量确认
            first_bottom_volume = df.iloc[first_bottom_idx]['volume']
            second_bottom_volume = df.iloc[second_bottom_idx]['volume']
            volume_increase = second_bottom_volume > first_bottom_volume
            
            # 计算形态强度
            strength = _calculate_double_bottom_strength(
                depth_pct, price_diff_pct, neckline_height_pct,
                bars_between, volume_increase
            )
            
            # 保存检测到的形态
            results.append({
                'type': 'double_bottom',
                'first_bottom': df.index[first_bottom_idx],
                'second_bottom': df.index[second_bottom_idx],
                'neckline': df.index[neckline_idx],
                'neckline_breakout': neckline_breakout,
                'first_bottom_price': first_bottom_price,
                'second_bottom_price': second_bottom_price,
                'neckline_price': neckline_price,
                'depth_pct': depth_pct,
                'price_diff_pct': price_diff_pct,
                'neckline_height_pct': neckline_height_pct,
                'volume_increase': volume_increase,
                'strength': strength,
                'target_price': neckline_price + (neckline_price - min(first_bottom_price, second_bottom_price)),  # 目标价格：颈线 + 形态高度
                'stop_loss': min(first_bottom_price, second_bottom_price) * 0.98  # 止损位：最低点下方2%
            })
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _find_local_lows(df: pd.DataFrame, window: int = 3) -> List[Tuple[int, float]]:
    """
    查找局部低点
    
    参数:
        df: 包含OHLCV数据的DataFrame
        window: 用于确定局部低点的窗口大小
        
    返回:
        List[Tuple[int, float]]: 局部低点列表，每个元素为(索引, 价格)
    """
    lows = []
    
    for i in range(window, len(df) - window):
        # 检查是否为局部低点
        if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
            lows.append((i, df['low'].iloc[i]))
    
    # 过滤掉太接近的低点
    filtered_lows = []
    if lows:
        current_low = lows[0]
        filtered_lows.append(current_low)
        
        for i in range(1, len(lows)):
            if lows[i][0] - filtered_lows[-1][0] >= window * 2:
                filtered_lows.append(lows[i])
    
    return filtered_lows


def _calculate_double_bottom_strength(depth_pct: float, price_diff_pct: float,
                                     neckline_height_pct: float, bars_between: int,
                                     volume_increase: bool) -> float:
    """
    计算双底形态的强度
    
    参数:
        depth_pct: 底部深度百分比
        price_diff_pct: 两个底部之间的价格差异百分比
        neckline_height_pct: 颈线高度百分比
        bars_between: 两个底部之间的K线数量
        volume_increase: 第二个底部的成交量是否大于第一个底部
        
    返回:
        float: 形态强度评分 (0-1)
    """
    # 底部深度评分 (0.3权重)
    depth_score = min(depth_pct / 0.1, 1.0) * 0.3
    
    # 底部相似度评分 (0.2权重)
    similarity_score = (1.0 - min(price_diff_pct / 0.03, 1.0)) * 0.2
    
    # 颈线高度评分 (0.2权重)
    height_score = min(neckline_height_pct / 0.08, 1.0) * 0.2
    
    # 底部间距评分 (0.2权重)
    # 理想的间距为15-30根K线
    spacing_score = (1.0 - abs(bars_between - 20) / 20) * 0.2
    spacing_score = max(0, min(spacing_score, 0.2))
    
    # 成交量确认评分 (0.1权重)
    volume_score = 0.1 if volume_increase else 0.0
    
    # 总分
    return depth_score + similarity_score + height_score + spacing_score + volume_score


def is_double_bottom_breakout(df: pd.DataFrame, pattern: Dict) -> Dict:
    """
    检查双底形态是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 双底形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    # 检查索引是否存在
    if pattern['neckline_breakout'] not in df.index:
        return {'breakout': False}
    
    neckline_breakout_idx = df.index.get_loc(pattern['neckline_breakout'])
    
    # 确保有后续数据
    if neckline_breakout_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 检查后续3根K线是否确认突破
    post_pattern = df.iloc[neckline_breakout_idx:neckline_breakout_idx+3]
    
    # 确认突破条件：所有K线的收盘价都高于颈线
    confirmed = all(post_pattern['close'] > pattern['neckline_price'])
    
    if confirmed:
        return {
            'breakout': True,
            'direction': 'up',
            'price': post_pattern['close'].iloc[-1],
            'target': pattern['target_price'],
            'stop_loss': pattern['stop_loss'],
            'strength': pattern['strength']
        }
    else:
        return {'breakout': False}
