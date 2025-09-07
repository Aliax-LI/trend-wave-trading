"""
双底牛旗形态识别模块

双底牛旗形态特征:
1. 首先形成双底（W底）形态，表示下跌趋势的终结
   - 两个低点位置接近
   - 第二个低点不低于第一个低点
   - 中间有一个明显的反弹高点（颈线）
2. 突破颈线后形成牛旗形态
   - 突破颈线后有一段明显的上涨（旗杆）
   - 随后出现小幅回调整理（旗面），通常呈下降通道
   - 整理期间成交量逐渐萎缩
3. 最终价格突破旗面上轨，继续上涨趋势

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress

# 导入双底和牛旗检测函数
from app.analysis.pattern.double_bottom import detect_double_bottom
from app.analysis.pattern.bull_flag import detect_bull_flag


def detect_double_bottom_bull_flag(df: pd.DataFrame, max_gap_bars: int = 15) -> List[Dict]:
    """
    检测双底牛旗复合形态
    
    参数:
        df: 包含OHLCV数据的DataFrame
        max_gap_bars: 双底与牛旗之间的最大K线数量
    
    返回:
        List[Dict]: 检测到的双底牛旗形态列表，每个字典包含形态详情
    """
    if len(df) < 30:  # 确保有足够的数据
        return []
    
    results = []
    
    # 首先检测双底形态
    double_bottoms = detect_double_bottom(df)
    if not double_bottoms:
        return []
    
    # 然后检测牛旗形态
    bull_flags = detect_bull_flag(df)
    if not bull_flags:
        return []
    
    # 遍历所有可能的双底和牛旗组合
    for db in double_bottoms:
        # 获取双底的颈线突破点
        if db['neckline_breakout'] not in df.index:
            continue
        db_neckline_breakout_idx = df.index.get_loc(db['neckline_breakout'])
        
        for bf in bull_flags:
            # 获取牛旗的旗杆起点
            if bf['pole_start'] not in df.index:
                continue
            bf_pole_start_idx = df.index.get_loc(bf['pole_start'])
            
            # 计算双底颈线突破点和牛旗旗杆起点之间的K线数量
            gap_bars = bf_pole_start_idx - db_neckline_breakout_idx
            
            # 检查双底和牛旗是否在合适的时间顺序和距离内
            if 0 <= gap_bars <= max_gap_bars:
                # 确保牛旗的旗杆起点价格高于双底的颈线
                db_neckline_price = db['neckline_price']
                bf_pole_start_price = df.loc[bf['pole_start']]['open']
                
                if bf_pole_start_price > db_neckline_price:
                    # 计算复合形态的强度
                    strength = _calculate_combined_strength(db, bf, gap_bars, max_gap_bars)
                    
                    # 保存检测到的形态
                    results.append({
                        'type': 'double_bottom_bull_flag',
                        'double_bottom': db,
                        'bull_flag': bf,
                        'start': db['first_bottom'],  # 形态开始于第一个底部
                        'end': bf['flag_end'],        # 形态结束于牛旗旗面结束
                        'strength': strength,
                        'target_price': bf['target_price'],  # 使用牛旗的目标价格
                        'stop_loss': min(db['second_bottom_price'], bf['stop_loss'])  # 使用更保守的止损位
                    })
    
    # 按强度排序
    results.sort(key=lambda x: x['strength'], reverse=True)
    
    return results


def _calculate_combined_strength(double_bottom: Dict, bull_flag: Dict, gap_bars: int, max_gap_bars: int) -> float:
    """
    计算双底牛旗复合形态的强度
    
    参数:
        double_bottom: 双底形态字典
        bull_flag: 牛旗形态字典
        gap_bars: 双底与牛旗之间的K线数量
        max_gap_bars: 允许的最大间隔K线数
    
    返回:
        float: 形态强度评分 (0-1)
    """
    # 双底强度 (0.4权重)
    db_strength = double_bottom['strength'] * 0.4
    
    # 牛旗强度 (0.4权重)
    bf_strength = bull_flag['strength'] * 0.4
    
    # 间隔评分 (0.2权重)
    # 间隔越小越好，最理想的情况是牛旗紧接着双底颈线突破
    gap_score = (1.0 - gap_bars / max_gap_bars) * 0.2
    
    # 总分
    return db_strength + bf_strength + gap_score


def is_double_bottom_bull_flag_breakout(df: pd.DataFrame, pattern: Dict) -> Dict:
    """
    检查双底牛旗形态是否已经突破
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 双底牛旗形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    # 使用牛旗部分的突破逻辑
    bull_flag = pattern['bull_flag']
    
    # 检查索引是否存在
    if bull_flag['flag_end'] not in df.index:
        return {'breakout': False}
        
    flag_end_idx = df.index.get_loc(bull_flag['flag_end'])
    
    # 确保有后续数据
    if flag_end_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 获取旗面的高点
    flag_section = df.loc[bull_flag['flag_start']:bull_flag['flag_end']]
    flag_high = flag_section['high'].max()
    
    # 检查后续3根K线是否突破
    post_pattern = df.iloc[flag_end_idx+1:flag_end_idx+4]
    
    # 向上突破条件
    up_breakout = post_pattern['close'].iloc[-1] > flag_high and post_pattern['volume'].mean() > flag_section['volume'].mean() * 1.5
    
    if up_breakout:
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
