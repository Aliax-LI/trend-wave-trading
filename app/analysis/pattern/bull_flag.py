"""
牛旗形态识别模块

牛旗形态特征:
1. 前期有明显的上涨趋势（旗杆）
2. 随后出现小幅回调整理（旗面），通常呈下降通道或三角形
3. 整理期间成交量逐渐萎缩
4. 价格突破旗面上轨时成交量放大，确认突破

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_bull_flag(df: pd.DataFrame, min_pole_height_pct: float = 0.08,
                     max_flag_bars: int = 20, min_flag_bars: int = 7,
                     volume_threshold: float = 0.75, pole_bars_range: Tuple[int, int] = (5, 15)) -> List[Dict]:
    """
    检测牛旗形态
    
    参数:
        df: 包含OHLCV数据的DataFrame
        min_pole_height_pct: 旗杆最小高度（相对于价格的百分比）
        max_flag_bars: 旗面最大持续K线数
        min_flag_bars: 旗面最小持续K线数
        volume_threshold: 成交量萎缩阈值（相对于旗杆期间平均成交量）
    
    返回:
        List[Dict]: 检测到的牛旗形态列表，每个字典包含形态详情
    """
    if len(df) < min_flag_bars + 5:  # 确保有足够的数据
        return []
    
    results = []
    
    # 遍历可能的旗杆长度
    min_pole_bars, max_pole_bars = pole_bars_range
    
    for pole_bars in range(min_pole_bars, max_pole_bars + 1):
        # 遍历可能的旗杆结束点
        for pole_end in range(pole_bars, len(df) - min_flag_bars):
            # 检查前面的pole_bars根K线是否构成旗杆（强势上涨）
            pole_section = df.iloc[pole_end-pole_bars:pole_end+1]
            
            # 旗杆条件：价格明显上涨
            pole_start_price = pole_section['low'].iloc[0]
            pole_end_price = pole_section['high'].iloc[-1]
            pole_height_pct = (pole_end_price - pole_start_price) / pole_start_price
            
            if pole_height_pct < min_pole_height_pct:
                continue  # 旗杆高度不足
            
            # 检查是否有至少60%的K线是上涨的
            min_up_candles = int(len(pole_section) * 0.6)
            up_candles = sum(1 for i in range(len(pole_section)) 
                            if pole_section['close'].iloc[i] > pole_section['open'].iloc[i])
            if up_candles < min_up_candles:
                continue  # 上涨K线不足
                
            # 检查旗杆是否有明显的上升趋势
            x = np.array(range(len(pole_section)))
            y = pole_section['close'].values
            slope, _, r_value, _, _ = linregress(x, y)
            
            # 要求旗杆有明显的上升趋势，且拟合度较高
            if slope <= 0 or r_value < 0.7:
                continue
            
            # 旗杆期间的平均成交量
            pole_avg_volume = pole_section['volume'].mean()
            
            # 检查后续的K线是否构成旗面（横盘整理或小幅回调）
            for flag_length in range(min_flag_bars, min(max_flag_bars + 1, len(df) - pole_end)):
                flag_section = df.iloc[pole_end:pole_end+flag_length]
            
                # 旗面条件：价格在下降通道或横盘整理
                flag_high = flag_section['high'].max()
                flag_low = flag_section['low'].min()
                flag_range = flag_high - flag_low
                
                # 检查旗面是否为下降通道或横盘
                flag_start_close = flag_section['close'].iloc[0]
                flag_end_close = flag_section['close'].iloc[-1]
                
                # 旗面应该是横盘或小幅下跌，不应该上涨
                if flag_end_close > flag_start_close:
                    continue
                    
                # 旗面下跌幅度不应过大（相对于旗杆高度）
                flag_drop_pct = (flag_start_close - flag_end_close) / flag_start_close
                if flag_drop_pct > pole_height_pct * 0.5:  # 旗面回调不应超过旗杆高度的50%
                    continue
                
                # 确保旗面有回调特性（下降通道或横盘）
                # 计算旗面期间的线性趋势
                x = np.array(range(len(flag_section)))
                y = flag_section['close'].values
                flag_slope, _, flag_r_value, _, _ = linregress(x, y)
                
                # 旗面应该是下降趋势或横盘
                if flag_slope >= 0:  # 必须是下降趋势
                    continue
                    
                # 旗面的高度应该在合理范围内（相对于旗杆高度）
                flag_height_ratio = flag_range / (pole_end_price - pole_start_price)
                if flag_height_ratio < 0.2 or flag_height_ratio > 0.7:
                    continue
                
                # 检查旗面期间成交量是否萎缩
                flag_avg_volume = flag_section['volume'].mean()
                if flag_avg_volume > pole_avg_volume * volume_threshold:
                    continue  # 成交量没有足够萎缩
            
                # 检查是否有突破迹象（可选，作为额外确认）
                if pole_end + flag_length < len(df):
                    breakout_candle = df.iloc[pole_end + flag_length]
                    if (breakout_candle['close'] > flag_high and 
                        breakout_candle['volume'] > flag_avg_volume * 1.5):
                        # 检测到突破
                        breakout_confirmed = True
                    else:
                        breakout_confirmed = False
                else:
                    breakout_confirmed = False
                
                # 计算形态强度
                strength = _calculate_bull_flag_strength(
                    pole_height_pct, flag_length, flag_range / pole_end_price,
                    flag_avg_volume / pole_avg_volume,
                    abs(flag_slope), r_value
                )
                
                # 保存检测到的形态
                results.append({
                    'type': 'bull_flag',
                    'pole_start': df.index[pole_end - pole_bars],
                    'pole_end': df.index[pole_end],
                    'flag_start': df.index[pole_end],
                    'flag_end': df.index[pole_end + flag_length - 1],
                    'pole_height_pct': pole_height_pct,
                    'flag_length': flag_length,
                    'breakout_confirmed': breakout_confirmed,
                    'strength': strength,
                    'target_price': pole_end_price + pole_height_pct * pole_end_price,  # 价格目标：旗杆高度投射
                    'stop_loss': flag_low  # 止损位：旗面低点
                })
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _calculate_bull_flag_strength(pole_height_pct: float, flag_length: int, 
                                 flag_range_pct: float, volume_ratio: float,
                                 flag_slope: float = 0, r_value: float = 0) -> float:
    """
    计算牛旗形态的强度
    
    参数:
        pole_height_pct: 旗杆高度百分比
        flag_length: 旗面持续K线数
        flag_range_pct: 旗面振幅相对于价格的百分比
        volume_ratio: 旗面/旗杆成交量比率
    
    返回:
        float: 形态强度评分 (0-1)
    """
    # 旗杆强度 (0.4权重)
    pole_score = min(pole_height_pct / 0.15, 1.0) * 0.4
    
    # 旗面长度评分 (0.2权重)
    # 理想的旗面长度为10-20根K线
    length_score = (1.0 - abs(flag_length - 15) / 15) * 0.2
    length_score = max(0, min(length_score, 0.2))
    
    # 旗面振幅评分 (0.2权重)
    # 理想的旗面振幅为旗杆高度的30%-50%
    range_ratio = flag_range_pct / pole_height_pct
    range_score = (1.0 - abs(range_ratio - 0.4) / 0.4) * 0.2
    range_score = max(0, min(range_score, 0.2))
    
    # 成交量萎缩评分 (0.2权重)
    # 理想情况下，旗面成交量应为旗杆成交量的50%或更低
    volume_score = (1.0 - min(volume_ratio, 1.0)) * 0.2
    
    # 旗面下降趋势评分 (0.1权重)
    slope_score = min(flag_slope * 10, 1.0) * 0.1
    
    # 趋势拟合度评分 (0.1权重)
    fit_score = min(r_value, 1.0) * 0.1
    
    # 总分
    return pole_score + length_score + range_score + volume_score + slope_score + fit_score


def is_bull_flag_breakout(df: pd.DataFrame, flag_pattern: Dict) -> bool:
    """
    检查牛旗形态是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        flag_pattern: 牛旗形态字典
    
    返回:
        bool: 是否确认突破
    """
    flag_end_idx = df.index.get_loc(flag_pattern['flag_end'])
    
    # 确保有后续数据
    if flag_end_idx + 1 >= len(df):
        return False
    
    # 获取旗面的高点
    flag_section = df.loc[flag_pattern['flag_start']:flag_pattern['flag_end']]
    flag_high = flag_section['high'].max()
    
    # 检查后续K线是否突破旗面高点
    breakout_candle = df.iloc[flag_end_idx + 1]
    
    # 突破条件：收盘价高于旗面高点且成交量放大
    flag_avg_volume = flag_section['volume'].mean()
    return (breakout_candle['close'] > flag_high and 
            breakout_candle['volume'] > flag_avg_volume * 1.5)
