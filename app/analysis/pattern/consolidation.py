"""
两段式整理形态识别模块

两段式整理形态特征:
1. 价格经历两个明显的横盘整理阶段
2. 两个整理阶段之间有一个小幅波动的过渡期
3. 第二段整理的波动范围通常小于第一段
4. 成交量在整个形态中逐渐萎缩
5. 突破方向通常延续之前的主趋势

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_two_stage_consolidation(df: pd.DataFrame, min_length: int = 12, 
                                  max_length: int = 60, volatility_threshold: float = 0.04,
                                  transition_threshold: float = 0.03) -> List[Dict]:
    """
    检测两段式整理形态
    
    参数:
        df: 包含OHLCV数据的DataFrame
        min_length: 整个形态的最小长度（K线数量）
        max_length: 整个形态的最大长度（K线数量）
        volatility_threshold: 整理阶段的最大波动率（相对于价格的百分比）
        transition_threshold: 过渡期的最小波动率（相对于价格的百分比）
    
    返回:
        List[Dict]: 检测到的两段式整理形态列表，每个字典包含形态详情
    """
    if len(df) < min_length:
        return []
    
    results = []
    
    # 计算每个窗口的波动率
    volatility = []
    for i in range(len(df) - 5):
        window = df.iloc[i:i+5]
        vol = (window['high'].max() - window['low'].min()) / window['close'].mean()
        volatility.append(vol)
    
    # 填充前面的值
    volatility = [volatility[0]] * 5 + volatility
    df = df.copy()
    df['volatility'] = volatility
    
    # 寻找可能的整理区间起点
    for start_idx in range(len(df) - min_length):
        # 尝试不同的形态长度
        for pattern_length in range(min_length, min(max_length, len(df) - start_idx)):
            pattern_section = df.iloc[start_idx:start_idx + pattern_length]
            
            # 如果整个区间的波动率过大，则不是整理形态
            overall_volatility = (pattern_section['high'].max() - pattern_section['low'].min()) / pattern_section['close'].mean()
            if overall_volatility > volatility_threshold * 3:
                continue
            
            # 尝试不同的分割点，将整个区间分为两段
            best_split = None
            best_score = 0
            
            for split_ratio in [0.4, 0.45, 0.5, 0.55, 0.6]:
                split_idx = int(pattern_length * split_ratio)
                if split_idx < 5 or split_idx > pattern_length - 5:
                    continue
                    
                # 第一段整理
                stage1 = pattern_section.iloc[:split_idx]
                # 过渡期（中间20%的数据）
                transition_length = max(3, int(pattern_length * 0.2))
                transition_start = max(0, split_idx - transition_length // 2)
                transition_end = min(pattern_length, transition_start + transition_length)
                transition = pattern_section.iloc[transition_start:transition_end]
                # 第二段整理
                stage2 = pattern_section.iloc[split_idx:]
                
                # 计算各阶段的波动率
                stage1_vol = (stage1['high'].max() - stage1['low'].min()) / stage1['close'].mean()
                transition_vol = (transition['high'].max() - transition['low'].min()) / transition['close'].mean()
                stage2_vol = (stage2['high'].max() - stage2['low'].min()) / stage2['close'].mean()
                
                # 检查是否符合两段式整理的条件
                if (stage1_vol <= volatility_threshold and 
                    stage2_vol <= volatility_threshold and
                    transition_vol >= transition_threshold and
                    stage2_vol < stage1_vol):  # 第二段波动通常更小
                    
                    # 计算成交量变化
                    vol_change = stage2['volume'].mean() / stage1['volume'].mean()
                    
                    # 计算形态得分
                    score = _calculate_consolidation_score(
                        stage1_vol, stage2_vol, transition_vol, vol_change, pattern_length
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_split = {
                            'split_idx': split_idx,
                            'stage1_vol': stage1_vol,
                            'transition_vol': transition_vol,
                            'stage2_vol': stage2_vol,
                            'vol_change': vol_change
                        }
            
            # 如果找到了有效的分割点
            if best_split and best_score > 0.5:
                # 确定形态的趋势方向
                pre_pattern = df.iloc[max(0, start_idx-10):start_idx]
                if len(pre_pattern) > 5:
                    pre_trend = 'up' if pre_pattern['close'].iloc[-1] > pre_pattern['close'].iloc[0] else 'down'
                else:
                    pre_trend = 'neutral'
                
                # 确定突破方向的概率
                if pre_trend == 'up':
                    breakout_bias = 0.7  # 70%概率向上突破
                elif pre_trend == 'down':
                    breakout_bias = 0.3  # 30%概率向上突破
                else:
                    breakout_bias = 0.5  # 50%概率向上突破
                
                # 计算关键价格水平
                pattern_high = pattern_section['high'].max()
                pattern_low = pattern_section['low'].min()
                
                # 保存检测到的形态
                results.append({
                    'type': 'two_stage_consolidation',
                    'start': df.index[start_idx],
                    'end': df.index[start_idx + pattern_length - 1],
                    'split_point': df.index[start_idx + best_split['split_idx']],
                    'stage1_volatility': best_split['stage1_vol'],
                    'stage2_volatility': best_split['stage2_vol'],
                    'transition_volatility': best_split['transition_vol'],
                    'volume_change': best_split['vol_change'],
                    'strength': best_score,
                    'pre_trend': pre_trend,
                    'breakout_bias': breakout_bias,
                    'resistance': pattern_high,
                    'support': pattern_low
                })
                
                # 跳过重叠的区间
                start_idx += pattern_length // 2
                break
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _calculate_consolidation_score(stage1_vol: float, stage2_vol: float, 
                                  transition_vol: float, vol_change: float,
                                  pattern_length: int) -> float:
    """
    计算两段式整理形态的强度
    
    参数:
        stage1_vol: 第一阶段波动率
        stage2_vol: 第二阶段波动率
        transition_vol: 过渡期波动率
        vol_change: 成交量变化比例
        pattern_length: 形态总长度
    
    返回:
        float: 形态强度评分 (0-1)
    """
    # 波动率对比评分 (0.3权重)
    # 理想情况下，第二阶段波动率应该小于第一阶段，过渡期波动率应该大于两个阶段
    vol_ratio = stage2_vol / stage1_vol if stage1_vol > 0 else 1
    vol_score = (1.0 - min(vol_ratio, 1.0)) * 0.3
    
    # 过渡期波动率评分 (0.3权重)
    # 过渡期波动率应该明显高于整理阶段
    transition_score = min(transition_vol / max(stage1_vol, stage2_vol), 3.0) / 3.0 * 0.3
    
    # 成交量萎缩评分 (0.2权重)
    # 理想情况下，第二阶段成交量应该低于第一阶段
    volume_score = (1.0 - min(vol_change, 1.0)) * 0.2
    
    # 形态长度评分 (0.2权重)
    # 理想的形态长度为20-30天
    length_score = (1.0 - min(abs(pattern_length - 25) / 25, 1.0)) * 0.2
    
    # 总分
    return vol_score + transition_score + volume_score + length_score


def is_consolidation_breakout(df: pd.DataFrame, pattern: Dict) -> Dict:
    """
    检查两段式整理形态是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 两段式整理形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    pattern_end_idx = df.index.get_loc(pattern['end'])
    
    # 确保有后续数据
    if pattern_end_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 获取形态的高点和低点
    resistance = pattern['resistance']
    support = pattern['support']
    
    # 检查后续3根K线是否突破
    post_pattern = df.iloc[pattern_end_idx+1:pattern_end_idx+4]
    
    # 向上突破条件
    up_breakout = post_pattern['close'].iloc[-1] > resistance and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean()
    
    # 向下突破条件
    down_breakout = post_pattern['close'].iloc[-1] < support and post_pattern['volume'].mean() > df.iloc[pattern_end_idx-5:pattern_end_idx+1]['volume'].mean()
    
    if up_breakout:
        return {
            'breakout': True,
            'direction': 'up',
            'price': post_pattern['close'].iloc[-1],
            'target': resistance + (resistance - support),  # 目标价格：突破幅度等于形态高度
            'stop_loss': support
        }
    elif down_breakout:
        return {
            'breakout': True,
            'direction': 'down',
            'price': post_pattern['close'].iloc[-1],
            'target': support - (resistance - support),  # 目标价格：突破幅度等于形态高度
            'stop_loss': resistance
        }
    else:
        return {'breakout': False}
