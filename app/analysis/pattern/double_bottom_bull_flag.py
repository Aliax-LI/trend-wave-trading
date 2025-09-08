"""
双底牛旗形态识别模块 - 集成识别算法

双底牛旗形态特征（基于HTML示例优化）:
1. 旗杆阶段：明显的上涨趋势
   - 强势上涨，至少60%的K线为阳线
   - 上涨幅度至少8%
   - 良好的线性上升趋势（R² > 0.7）

2. 旗面整理阶段：横盘或小幅回调
   - 在旗面内部包含双底形态
   - 整理期间成交量萎缩
   - 价格振幅合理（不超过旗杆高度的60%）

3. 双底特征（在旗面内形成）：
   - 两个相近的低点（价格差异不超过3%）
   - 中间有明显的颈线反弹
   - 颈线高度至少2%

4. 突破确认：
   - 同时突破旗面上轨和双底颈线
   - 成交量放大确认（至少1.5倍）
   - 价格持续在突破位之上

- 集成检测：在旗面内直接检测双底，避免时间序列匹配问题
- 多重确认：价格+成交量+持续性的综合突破确认
- 动态目标：基于旗杆高度投射的目标价格计算
- 精确止损：使用双底低点作为止损基准

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import linregress


def detect_double_bottom_bull_flag(df: pd.DataFrame, 
                                  min_pole_height_pct: Optional[float] = None,
                                  max_flag_bars: int = 20, 
                                  min_flag_bars: int = 10,
                                  pole_bars_range: Tuple[int, int] = (5, 12),
                                  min_strength: float = 0.75,
                                  max_results: int = 5,
                                  adaptive_mode: bool = True) -> List[Dict]:
    """
    检测双底牛旗复合形态 - 基于HTML示例调整的识别算法
    
    形态特征：
    1. 旗杆：明显的上涨趋势
    2. 旗面：横盘整理期间内部包含双底形态
    3. 双底：两个相近的低点，中间有颈线反弹
    4. 突破：价格同时突破旗面上轨和双底颈线
    
    参数:
        df: 包含OHLCV数据的DataFrame
        min_pole_height_pct: 旗杆最小高度百分比（None时自动计算）
        max_flag_bars: 旗面最大K线数
        min_flag_bars: 旗面最小K线数
        pole_bars_range: 旗杆K线数范围
        min_strength: 最小强度要求
        max_results: 最大返回结果数
        adaptive_mode: 是否启用自适应模式
    
    返回:
        List[Dict]: 检测到的双底牛旗形态列表
    """
    if len(df) < 35:  # 确保有足够的数据
        return []
    
    # 自适应计算最小旗杆高度
    if adaptive_mode and min_pole_height_pct is None:
        min_pole_height_pct = _calculate_adaptive_pole_height(df)
    elif min_pole_height_pct is None:
        min_pole_height_pct = 0.08  # 默认8%
    
    results = []
    min_pole_bars, max_pole_bars = pole_bars_range
    
    # 遍历可能的旗杆长度和位置
    for pole_bars in range(min_pole_bars, max_pole_bars + 1):
        for pole_end in range(pole_bars, len(df) - max_flag_bars):
            # 1. 检测旗杆（强势上涨）
            pole_section = df.iloc[pole_end-pole_bars:pole_end+1]
            pole_start_price = pole_section['low'].iloc[0]
            pole_end_price = pole_section['high'].iloc[-1]
            pole_height_pct = (pole_end_price - pole_start_price) / pole_start_price
            
            if pole_height_pct < min_pole_height_pct:
                continue
            
            # 检查旗杆的上涨强度
            if not _is_valid_pole(pole_section):
                continue
            
            # 2. 检测旗面期间的双底形态
            for flag_length in range(min_flag_bars, min(max_flag_bars + 1, len(df) - pole_end)):
                flag_section = df.iloc[pole_end:pole_end+flag_length]
                
                # 检查旗面是否为横盘整理
                if not _is_valid_flag_range(flag_section, pole_height_pct):
                    continue
                
                # 3. 在旗面内检测双底形态
                double_bottom = _detect_double_bottom_in_flag(flag_section)
                if not double_bottom:
                    continue
                
                # 4. 验证双底在旗面中的位置合理性
                if not _validate_double_bottom_position(flag_section, double_bottom):
                    continue
                
                # 5. 计算复合形态强度
                strength = _calculate_integrated_strength(
                    pole_section, flag_section, double_bottom, pole_height_pct, min_pole_height_pct
                )
                
                # 跳过强度不足的形态
                if strength < min_strength:
                    continue
                
                # 6. 设置目标价格和止损位
                target_price = _calculate_target_price(pole_section, flag_section, double_bottom)
                stop_loss = _calculate_stop_loss(flag_section, double_bottom)
                
                # 保存检测到的形态
                results.append({
                    'type': 'double_bottom_bull_flag',
                    'pole_start': df.index[pole_end - pole_bars],
                    'pole_end': df.index[pole_end],
                    'flag_start': df.index[pole_end],
                    'flag_end': df.index[pole_end + flag_length - 1],
                    'first_bottom': df.index[pole_end + double_bottom['first_bottom_idx']],
                    'second_bottom': df.index[pole_end + double_bottom['second_bottom_idx']],
                    'neckline': df.index[pole_end + double_bottom['neckline_idx']],
                    'pole_height_pct': pole_height_pct,
                    'flag_length': flag_length,
                    'double_bottom_data': double_bottom,
                    'strength': strength,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'flag_high': flag_section['high'].max(),
                    'flag_low': flag_section['low'].min(),
                    'neckline_price': double_bottom['neckline_price']
                })
    
    # 过滤重复和弱形态
    filtered_results = _filter_overlapping_patterns(results, min_strength)
    
    # 按强度排序并限制结果数量
    filtered_results.sort(key=lambda x: x['strength'], reverse=True)
    
    return filtered_results[:max_results] if len(filtered_results) > max_results else filtered_results


def _calculate_adaptive_pole_height(df: pd.DataFrame) -> float:
    """
    根据市场波动特性自适应计算最小旗杆高度要求
    
    参数:
        df: OHLCV数据
    
    返回:
        float: 适应市场波动的最小旗杆高度百分比
    """
    if len(df) < 20:
        return 0.08  # 默认8%
    
    # 计算市场的整体波动特性
    price_changes = []
    for i in range(1, min(50, len(df))):  # 分析最近50根K线的波动
        daily_change = abs((df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1])
        price_changes.append(daily_change)
    
    if not price_changes:
        return 0.08
    
    # 计算平均日波动率和标准差
    avg_volatility = np.mean(price_changes)
    volatility_std = np.std(price_changes)
    
    # 计算价格的整体波动范围（最近20根K线）
    recent_high = df['high'].iloc[-20:].max()
    recent_low = df['low'].iloc[-20:].min()
    recent_close = df['close'].iloc[-1]
    range_volatility = (recent_high - recent_low) / recent_close
    
    # 根据波动特性分类市场
    if avg_volatility < 0.005 and range_volatility < 0.03:
        # 低波动市场（如稳定的大盘股、主要货币对）
        min_height = 0.03  # 3%
        market_type = "低波动"
    elif avg_volatility < 0.015 and range_volatility < 0.08:
        # 中等波动市场（如一般股票、次要货币对）
        min_height = 0.05  # 5%
        market_type = "中等波动"
    elif avg_volatility < 0.03 and range_volatility < 0.15:
        # 高波动市场（如小盘股、加密货币）
        min_height = 0.07  # 7%
        market_type = "高波动"
    else:
        # 极高波动市场（如投机性很强的资产）
        min_height = 0.10  # 10%
        market_type = "极高波动"
    
    # 微调：基于标准差进一步调整
    if volatility_std > avg_volatility * 2:  # 波动不稳定
        min_height *= 1.2  # 提高要求
    elif volatility_std < avg_volatility * 0.5:  # 波动很稳定
        min_height *= 0.8  # 降低要求
    
    # 确保在合理范围内
    min_height = max(0.02, min(0.12, min_height))  # 2%-12%之间
    
    # 可以在调试时输出市场分析结果
    # print(f"市场分析 - 类型: {market_type}, 平均波动: {avg_volatility:.3f}, 区间波动: {range_volatility:.3f}, 最小旗杆高度: {min_height:.1%}")
    
    return min_height


def _filter_overlapping_patterns(results: List[Dict], min_strength: float) -> List[Dict]:
    """
    过滤重叠和相似的形态，只保留最优质的形态
    
    参数:
        results: 检测到的形态列表
        min_strength: 最小强度阈值
    
    返回:
        List[Dict]: 过滤后的形态列表
    """
    if not results:
        return []
    
    # 首先按强度过滤
    high_quality_results = [r for r in results if r['strength'] >= min_strength]
    
    if len(high_quality_results) <= 1:
        return high_quality_results
    
    # 按强度排序
    high_quality_results.sort(key=lambda x: x['strength'], reverse=True)
    
    filtered = []
    for current in high_quality_results:
        is_overlapping = False
        
        for existing in filtered:
            # 检查时间重叠
            current_start = current['pole_start']
            current_end = current['flag_end']
            existing_start = existing['pole_start']
            existing_end = existing['flag_end']
            
            # 计算重叠程度
            overlap_start = max(current_start, existing_start)
            overlap_end = min(current_end, existing_end)
            
            if overlap_start < overlap_end:
                # 计算重叠比例
                current_duration = (current_end - current_start).total_seconds()
                existing_duration = (existing_end - existing_start).total_seconds()
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                
                overlap_ratio_current = overlap_duration / current_duration if current_duration > 0 else 0
                overlap_ratio_existing = overlap_duration / existing_duration if existing_duration > 0 else 0
                
                # 如果重叠超过50%，认为是相似形态
                if overlap_ratio_current > 0.5 or overlap_ratio_existing > 0.5:
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            filtered.append(current)
            # 限制最多返回3个不重叠的高质量形态
            if len(filtered) >= 3:
                break
    
    return filtered


def _is_valid_pole(pole_section: pd.DataFrame) -> bool:
    """
    验证旗杆是否为有效的上涨趋势
    
    参数:
        pole_section: 旗杆部分的DataFrame
    
    返回:
        bool: 是否为有效旗杆
    """
    # 检查上涨K线比例
    up_candles = sum(1 for i in range(len(pole_section)) 
                    if pole_section['close'].iloc[i] > pole_section['open'].iloc[i])
    up_ratio = up_candles / len(pole_section)
    
    if up_ratio < 0.6:  # 至少60%为上涨K线
        return False
    
    # 检查整体上升趋势
    x = np.array(range(len(pole_section)))
    y = pole_section['close'].values
    slope, _, r_value, _, _ = linregress(x, y)
    
    # 要求明显的上升趋势且拟合度高
    return slope > 0 and r_value > 0.7


def _is_valid_flag_range(flag_section: pd.DataFrame, pole_height_pct: float) -> bool:
    """
    验证旗面是否为有效的横盘整理
    
    参数:
        flag_section: 旗面部分的DataFrame
        pole_height_pct: 旗杆高度百分比
    
    返回:
        bool: 是否为有效旗面
    """
    flag_high = flag_section['high'].max()
    flag_low = flag_section['low'].min()
    flag_range = flag_high - flag_low
    flag_start_close = flag_section['close'].iloc[0]
    
    # 旗面振幅应该合理（不应过大或过小）
    flag_range_pct = flag_range / flag_start_close
    if flag_range_pct > pole_height_pct * 0.6 or flag_range_pct < 0.02:
        return False
    
    # 旗面应该是横盘或轻微下降
    flag_end_close = flag_section['close'].iloc[-1]
    if flag_end_close > flag_start_close * 1.02:  # 不应明显上涨
        return False
    
    return True


def _detect_double_bottom_in_flag(flag_section: pd.DataFrame) -> Optional[Dict]:
    """
    在旗面内检测双底形态 - 更严格的识别条件
    
    参数:
        flag_section: 旗面部分的DataFrame
    
    返回:
        Optional[Dict]: 双底形态数据，如果未找到则返回None
    """
    if len(flag_section) < 10:  # 需要足够的K线数
        return None
    
    # 寻找局部低点 - 使用更严格的条件
    lows = []
    window = 2  # 较小的窗口，适应旗面的短期特性
    
    for i in range(window, len(flag_section) - window):
        current_low = flag_section['low'].iloc[i]
        # 检查是否为显著的局部低点
        is_local_min = True
        for j in range(1, window + 1):
            if current_low > flag_section['low'].iloc[i - j] * 0.998:  # 需要明显更低
                is_local_min = False
                break
            if current_low > flag_section['low'].iloc[i + j] * 0.998:
                is_local_min = False
                break
        
        if is_local_min:
            # 检查是否有相应的成交量确认
            volume_confirmation = flag_section['volume'].iloc[i] > flag_section['volume'].iloc[i-1:i+2].mean() * 0.8
            if volume_confirmation:
                lows.append((i, current_low))
    
    if len(lows) < 2:
        return None
    
    # 寻找最佳的双底组合 - 更严格的条件
    best_double_bottom = None
    best_score = 0
    
    for i in range(len(lows) - 1):
        first_idx, first_price = lows[i]
        for j in range(i + 1, len(lows)):
            second_idx, second_price = lows[j]
            
            # 检查两个底部之间的距离
            separation = second_idx - first_idx
            if separation < 5 or separation > len(flag_section) * 0.7:  # 间隔要适中
                continue
            
            # 检查价格相似性 - 更严格
            avg_price = (first_price + second_price) / 2
            price_diff_pct = abs(second_price - first_price) / avg_price
            if price_diff_pct > 0.025:  # 差异不超过2.5%
                continue
            
            # 寻找中间的高点（颈线）
            between_section = flag_section.iloc[first_idx+1:second_idx]
            if between_section.empty:
                continue
            
            neckline_idx = between_section['high'].idxmax()
            relative_neckline_idx = first_idx + 1 + between_section.index.get_loc(neckline_idx)
            neckline_price = between_section.loc[neckline_idx, 'high']
            
            # 检查颈线高度 - 更严格
            bottom_price = min(first_price, second_price)
            neckline_height_pct = (neckline_price - bottom_price) / bottom_price
            if neckline_height_pct < 0.025:  # 颈线高度至少2.5%
                continue
            
            # 检查形态的对称性
            left_duration = relative_neckline_idx - first_idx
            right_duration = second_idx - relative_neckline_idx
            symmetry_score = 1 - abs(left_duration - right_duration) / max(left_duration, right_duration)
            
            if symmetry_score < 0.3:  # 要求一定的对称性
                continue
            
            # 计算双底质量分数
            quality_score = (
                (1 - price_diff_pct / 0.025) * 0.4 +  # 价格相似性权重40%
                min(neckline_height_pct / 0.05, 1) * 0.3 +  # 颈线高度权重30%
                symmetry_score * 0.3  # 对称性权重30%
            )
            
            if quality_score > best_score:
                best_score = quality_score
                best_double_bottom = {
                    'first_bottom_idx': first_idx,
                    'second_bottom_idx': second_idx,
                    'neckline_idx': relative_neckline_idx,
                    'first_bottom_price': first_price,
                    'second_bottom_price': second_price,
                    'neckline_price': neckline_price,
                    'price_diff_pct': price_diff_pct,
                    'neckline_height_pct': neckline_height_pct,
                    'quality_score': quality_score,
                    'symmetry_score': symmetry_score
                }
    
    # 只返回质量分数较高的双底
    if best_double_bottom and best_score > 0.6:
        return best_double_bottom
    
    return None


def _validate_double_bottom_position(flag_section: pd.DataFrame, double_bottom: Dict) -> bool:
    """
    验证双底在旗面中的位置是否合理
    
    参数:
        flag_section: 旗面部分的DataFrame
        double_bottom: 双底形态数据
    
    返回:
        bool: 位置是否合理
    """
    # 双底应该位于旗面的中下部分
    flag_high = flag_section['high'].max()
    flag_low = flag_section['low'].min()
    flag_range = flag_high - flag_low
    
    bottom_level = (double_bottom['first_bottom_price'] + double_bottom['second_bottom_price']) / 2
    bottom_position = (bottom_level - flag_low) / flag_range
    
    # 底部应该在旗面的下1/3到中部之间
    return 0.0 <= bottom_position <= 0.6


def _calculate_integrated_strength(pole_section: pd.DataFrame, flag_section: pd.DataFrame, 
                                 double_bottom: Dict, pole_height_pct: float, min_pole_height_pct: float) -> float:
    """
    计算集成的双底牛旗形态强度 - 自适应评分标准
    
    参数:
        pole_section: 旗杆部分数据
        flag_section: 旗面部分数据
        double_bottom: 双底形态数据
        pole_height_pct: 旗杆高度百分比
        min_pole_height_pct: 最小旗杆高度要求
    
    返回:
        float: 形态强度评分 (0-1)
    """
    # 旗杆强度 (0.25权重) - 自适应评分
    pole_score = 0
    if pole_height_pct >= min_pole_height_pct:
        # 根据实际的最小要求进行评分
        score_range = max(0.06, min_pole_height_pct * 1.5)  # 评分范围为最小要求的1.5倍
        pole_score = min((pole_height_pct - min_pole_height_pct) / score_range, 1.0) * 0.25
    
    # 双底质量 (0.35权重) - 增加权重
    db_quality_score = double_bottom.get('quality_score', 0.5)
    db_score = db_quality_score * 0.35
    
    # 旗面质量 (0.2权重)
    flag_range = flag_section['high'].max() - flag_section['low'].min()
    flag_start_price = flag_section['close'].iloc[0]
    flag_range_pct = flag_range / flag_start_price
    
    # 理想的旗面范围应该是旗杆高度的30-50%
    optimal_range_min = pole_height_pct * 0.3
    optimal_range_max = pole_height_pct * 0.5
    
    if optimal_range_min <= flag_range_pct <= optimal_range_max:
        range_score = 0.2
    else:
        # 超出理想范围的惩罚
        if flag_range_pct < optimal_range_min:
            range_score = (flag_range_pct / optimal_range_min) * 0.2
        else:
            range_score = max(0, (1 - (flag_range_pct - optimal_range_max) / optimal_range_max)) * 0.2
    
    # 成交量特征 (0.15权重) - 增加权重
    pole_avg_volume = pole_section['volume'].mean()
    flag_avg_volume = flag_section['volume'].mean()
    volume_ratio = flag_avg_volume / pole_avg_volume
    
    # 理想的成交量萎缩比例为0.5-0.8
    if 0.5 <= volume_ratio <= 0.8:
        volume_score = 0.15
    elif volume_ratio < 0.5:
        volume_score = (volume_ratio / 0.5) * 0.15
    else:
        volume_score = max(0, (1.2 - volume_ratio) / 0.4) * 0.15
    
    # 形态时间比例 (0.05权重)
    pole_duration = len(pole_section)
    flag_duration = len(flag_section)
    time_ratio = flag_duration / pole_duration
    
    # 理想的时间比例为1.0-2.0
    if 1.0 <= time_ratio <= 2.0:
        time_score = 0.05
    else:
        time_score = max(0, (1 - abs(time_ratio - 1.5) / 1.5)) * 0.05
    
    total_score = pole_score + db_score + range_score + volume_score + time_score
    
    # 应用额外的质量检查惩罚 - 自适应标准
    if pole_height_pct < min_pole_height_pct * 1.25:  # 旗杆相对较低
        total_score *= 0.8
    
    if double_bottom['price_diff_pct'] > 0.02:  # 双底差异太大
        total_score *= 0.9
    
    # 颈线高度要求根据市场波动调整
    min_neckline_height = max(0.015, min_pole_height_pct * 0.5)
    if double_bottom['neckline_height_pct'] < min_neckline_height:  # 颈线太低
        total_score *= 0.9
    
    return min(total_score, 1.0)


def _calculate_target_price(pole_section: pd.DataFrame, flag_section: pd.DataFrame, 
                          double_bottom: Dict) -> float:
    """
    计算目标价格
    
    参数:
        pole_section: 旗杆部分数据
        flag_section: 旗面部分数据
        double_bottom: 双底形态数据
    
    返回:
        float: 目标价格
    """
    # 使用旗杆高度投射法
    pole_height = pole_section['high'].iloc[-1] - pole_section['low'].iloc[0]
    breakout_level = max(flag_section['high'].max(), double_bottom['neckline_price'])
    
    return breakout_level + pole_height


def _calculate_stop_loss(flag_section: pd.DataFrame, double_bottom: Dict) -> float:
    """
    计算止损价格
    
    参数:
        flag_section: 旗面部分数据
        double_bottom: 双底形态数据
    
    返回:
        float: 止损价格
    """
    # 使用双底的较低点作为止损基准
    bottom_price = min(double_bottom['first_bottom_price'], double_bottom['second_bottom_price'])
    return bottom_price * 0.98  # 留2%缓冲


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
    检查双底牛旗形态是否已经突破 - 基于新的集成检测逻辑
    
    参数:
        df: 包含OHLCV数据的DataFrame
        pattern: 双底牛旗形态字典
    
    返回:
        Dict: 突破信息，包含方向、强度等
    """
    # 检查索引是否存在
    if pattern['flag_end'] not in df.index:
        return {'breakout': False}
        
    flag_end_idx = df.index.get_loc(pattern['flag_end'])
    
    # 确保有后续数据进行突破确认
    if flag_end_idx + 3 >= len(df):
        return {'breakout': False}
    
    # 获取关键价格水平
    flag_high = pattern['flag_high']  # 旗面上轨
    neckline_price = pattern['neckline_price']  # 双底颈线
    breakout_level = max(flag_high, neckline_price)  # 双重突破水平
    
    # 获取旗面期间的平均成交量
    flag_section = df.loc[pattern['flag_start']:pattern['flag_end']]
    flag_avg_volume = flag_section['volume'].mean()
    
    # 检查后续K线的突破情况
    post_pattern = df.iloc[flag_end_idx+1:flag_end_idx+4]
    
    # 突破确认条件
    breakout_conditions = []
    
    # 1. 价格突破条件
    price_breakout = any(post_pattern['close'] > breakout_level)
    breakout_conditions.append(price_breakout)
    
    # 2. 成交量放大条件
    volume_confirmation = any(post_pattern['volume'] > flag_avg_volume * 1.5)
    breakout_conditions.append(volume_confirmation)
    
    # 3. 持续性条件（最后一根K线仍在突破水平之上）
    sustainability = post_pattern['close'].iloc[-1] > breakout_level * 0.995
    breakout_conditions.append(sustainability)
    
    # 计算突破强度
    if sum(breakout_conditions) >= 2:  # 至少满足2个条件
        # 计算突破强度
        breakout_price = post_pattern['close'].iloc[-1]
        breakout_magnitude = (breakout_price - breakout_level) / breakout_level
        volume_ratio = post_pattern['volume'].mean() / flag_avg_volume
        
        # 综合突破强度
        breakout_strength = min(1.0, pattern['strength'] * 0.5 + 
                               min(breakout_magnitude * 10, 0.3) + 
                               min(volume_ratio / 3, 0.2))
        
        return {
            'breakout': True,
            'direction': 'up',
            'price': breakout_price,
            'target': pattern['target_price'],
            'stop_loss': pattern['stop_loss'],
            'strength': breakout_strength,
            'breakout_level': breakout_level,
            'flag_high': flag_high,
            'neckline_price': neckline_price,
            'volume_ratio': volume_ratio,
            'conditions_met': sum(breakout_conditions)
        }
    else:
        return {'breakout': False}
