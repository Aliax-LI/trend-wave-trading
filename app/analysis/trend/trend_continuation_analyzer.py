"""
趋势延续分析器
专门分析趋势的持续性和可能的转折点
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Tuple, List, Optional
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from app.models.trend_types import TrendDirection, TrendPhase


class ContinuationSignal(Enum):
    """趋势延续信号"""
    STRONG_CONTINUATION = "strong_continuation"    # 强烈延续
    WEAK_CONTINUATION = "weak_continuation"        # 弱延续
    CONSOLIDATION = "consolidation"                # 整理
    EARLY_REVERSAL = "early_reversal"             # 早期反转信号
    REVERSAL_WARNING = "reversal_warning"          # 反转警告


@dataclass
class ContinuationAnalysis:
    """趋势延续分析结果"""
    signal: ContinuationSignal
    momentum_strength: float  # 动量强度 (0-1)
    volume_confirmation: bool  # 成交量确认
    pullback_quality: str  # 回调质量
    breakout_potential: float  # 突破潜力 (0-1)
    entry_timing: str  # 入场时机
    continuation_probability: float  # 延续概率 (0-1)


class TrendContinuationAnalyzer:
    """趋势延续分析器"""
    
    def __init__(self, momentum_period: int = 14, volume_ma_period: int = 20):
        """
        初始化趋势延续分析器
        
        Args:
            momentum_period: 动量指标周期
            volume_ma_period: 成交量均线周期
        """
        self.momentum_period = momentum_period
        self.volume_ma_period = volume_ma_period
    
    def analyze_momentum_divergence(self, ohlc_data: pd.DataFrame) -> Tuple[str, float]:
        """
        分析动量背离
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            (背离类型, 背离强度)
        """
        if len(ohlc_data) < self.momentum_period * 2:
            return "数据不足", 0.0
            
        # 计算RSI和价格的背离
        rsi = talib.RSI(ohlc_data['close'].values, timeperiod=self.momentum_period)
        
        # 寻找最近的价格高点和低点
        recent_data = ohlc_data.tail(20)
        recent_rsi = rsi[-20:]
        
        price_highs = []
        rsi_highs = []
        price_lows = []
        rsi_lows = []
        
        # 简化的峰谷检测
        for i in range(2, len(recent_data) - 2):
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                price_highs.append((i, recent_data['high'].iloc[i]))
                rsi_highs.append((i, recent_rsi[i]))
                
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                price_lows.append((i, recent_data['low'].iloc[i]))
                rsi_lows.append((i, recent_rsi[i]))
        
        # 检查顶背离
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            latest_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            latest_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]
            
            if latest_price_high > prev_price_high and latest_rsi_high < prev_rsi_high:
                divergence_strength = abs(latest_rsi_high - prev_rsi_high) / 100
                return "顶背离", min(divergence_strength * 2, 1.0)
        
        # 检查底背离
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            latest_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            latest_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]
            
            if latest_price_low < prev_price_low and latest_rsi_low > prev_rsi_low:
                divergence_strength = abs(latest_rsi_low - prev_rsi_low) / 100
                return "底背离", min(divergence_strength * 2, 1.0)
        
        return "无背离", 0.0
    
    def analyze_volume_profile(self, ohlc_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        分析成交量特征
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            (成交量确认, 成交量强度)
        """
        if len(ohlc_data) < self.volume_ma_period:
            return False, 0.0
            
        volume_ma = talib.SMA(ohlc_data['volume'].values, timeperiod=self.volume_ma_period)
        recent_volume = ohlc_data['volume'].tail(5)
        recent_volume_ma = volume_ma[-5:]
        
        # 计算成交量比率
        volume_ratios = recent_volume.values / recent_volume_ma
        avg_volume_ratio = np.mean(volume_ratios)
        
        # 成交量确认条件
        volume_confirmation = avg_volume_ratio > 1.2  # 高于均线20%
        volume_strength = min(avg_volume_ratio / 2, 1.0)
        
        return volume_confirmation, volume_strength
    
    def analyze_pullback_quality(self, ohlc_data: pd.DataFrame, 
                                trend_direction: TrendDirection) -> Tuple[str, float]:
        """
        分析回调质量
        
        Args:
            ohlc_data: OHLC数据
            trend_direction: 趋势方向
            
        Returns:
            (回调质量, 质量评分)
        """
        if len(ohlc_data) < 10:
            return "数据不足", 0.0
            
        recent_data = ohlc_data.tail(10)
        
        # 计算ATR用于判断回调幅度
        atr = talib.ATR(ohlc_data['high'].values, ohlc_data['low'].values, 
                       ohlc_data['close'].values, timeperiod=14)
        current_atr = atr[-1]
        
        # 计算最近的回调幅度
        if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]:
            recent_high = recent_data['high'].max()
            current_price = recent_data['close'].iloc[-1]
            pullback_size = (recent_high - current_price) / current_atr
            
            if pullback_size < 0.5:
                return "微幅回调", 0.9  # 很好的趋势延续信号
            elif pullback_size < 1.0:
                return "浅度回调", 0.8  # 好的买入机会
            elif pullback_size < 2.0:
                return "中度回调", 0.6  # 需要谨慎
            else:
                return "深度回调", 0.3  # 趋势可能转变
                
        elif trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]:
            recent_low = recent_data['low'].min()
            current_price = recent_data['close'].iloc[-1]
            pullback_size = (current_price - recent_low) / current_atr
            
            if pullback_size < 0.5:
                return "微幅反弹", 0.9
            elif pullback_size < 1.0:
                return "浅度反弹", 0.8
            elif pullback_size < 2.0:
                return "中度反弹", 0.6
            else:
                return "深度反弹", 0.3
        
        return "横盘", 0.5
    
    def calculate_breakout_potential(self, ohlc_data: pd.DataFrame) -> float:
        """
        计算突破潜力
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            突破潜力评分 (0-1)
        """
        if len(ohlc_data) < 20:
            return 0.0
            
        # 计算布林带宽度
        close_prices = ohlc_data['close'].values
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
        
        current_bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        avg_bb_width = np.mean((bb_upper[-20:] - bb_lower[-20:]) / bb_middle[-20:])
        
        # 布林带收缩表示即将突破
        bb_squeeze_ratio = current_bb_width / avg_bb_width
        
        # 计算价格位置（接近布林带边缘表示突破概率高）
        current_price = close_prices[-1]
        bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        
        # 边缘位置得分更高
        edge_score = max(bb_position, 1 - bb_position) if bb_position > 0 and bb_position < 1 else 0
        
        # 综合评分
        breakout_potential = (
            (1 - bb_squeeze_ratio) * 0.6 +  # 布林带收缩
            edge_score * 0.4  # 价格位置
        )
        
        return min(max(breakout_potential, 0), 1)
    
    
    def analyze_continuation(self, ohlc_data: pd.DataFrame, 
                           trend_direction: TrendDirection,
                           trend_phase: TrendPhase) -> ContinuationAnalysis:
        """
        综合分析趋势延续性
        
        Args:
            ohlc_data: OHLC数据
            trend_direction: 当前趋势方向
            trend_phase: 当前趋势阶段
            
        Returns:
            趋势延续分析结果
        """
        try:
            # 各项分析
            divergence_type, divergence_strength = self.analyze_momentum_divergence(ohlc_data)
            volume_confirmation, volume_strength = self.analyze_volume_profile(ohlc_data)
            pullback_quality, quality_score = self.analyze_pullback_quality(ohlc_data, trend_direction)
            breakout_potential = self.calculate_breakout_potential(ohlc_data)
            
            # 计算延续概率
            continuation_factors = []
            
            # 趋势阶段权重
            if trend_phase == TrendPhase.BEGINNING:
                continuation_factors.append(0.8)
            elif trend_phase == TrendPhase.ACCELERATION:
                continuation_factors.append(0.9)
            elif trend_phase == TrendPhase.MATURITY:
                continuation_factors.append(0.6)
            elif trend_phase == TrendPhase.EXHAUSTION:
                continuation_factors.append(0.2)
            
            # 背离影响
            if "背离" in divergence_type:
                continuation_factors.append(1.0 - divergence_strength)
            else:
                continuation_factors.append(0.7)
            
            # 成交量确认
            continuation_factors.append(volume_strength)
            
            # 回调质量
            continuation_factors.append(quality_score)
            
            # 计算综合延续概率
            continuation_probability = np.mean(continuation_factors)
            
            # 确定延续信号
            if continuation_probability > 0.8 and quality_score > 0.8:
                signal = ContinuationSignal.STRONG_CONTINUATION
            elif continuation_probability > 0.6:
                signal = ContinuationSignal.WEAK_CONTINUATION
            elif "背离" in divergence_type and divergence_strength > 0.7:
                signal = ContinuationSignal.REVERSAL_WARNING
            elif continuation_probability < 0.3:
                signal = ContinuationSignal.EARLY_REVERSAL
            else:
                signal = ContinuationSignal.CONSOLIDATION
            
            # 入场时机判断
            if signal == ContinuationSignal.STRONG_CONTINUATION and "微幅" in pullback_quality:
                entry_timing = "立即入场"
            elif signal == ContinuationSignal.WEAK_CONTINUATION and "浅度" in pullback_quality:
                entry_timing = "谨慎入场"
            elif breakout_potential > 0.7:
                entry_timing = "等待突破"
            else:
                entry_timing = "观望"
            
            return ContinuationAnalysis(
                signal=signal,
                momentum_strength=1.0 - divergence_strength if "背离" in divergence_type else 0.7,
                volume_confirmation=volume_confirmation,
                pullback_quality=pullback_quality,
                breakout_potential=breakout_potential,
                entry_timing=entry_timing,
                continuation_probability=continuation_probability
            )
            
        except Exception as e:
            logger.error(f"趋势延续分析错误: {e}")
            return ContinuationAnalysis(
                signal=ContinuationSignal.CONSOLIDATION,
                momentum_strength=0.5,
                volume_confirmation=False,
                pullback_quality="未知",
                breakout_potential=0.5,
                entry_timing="观望",
                continuation_probability=0.5
            )
