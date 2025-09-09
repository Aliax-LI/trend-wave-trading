"""
价格行为分析器 - 用于补充EMA趋势分析器
专注于K线形态、支撑阻力、价格结构分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from app.models.trend_types import TrendDirection


class CandlePattern(Enum):
    """K线形态枚举"""
    DOJI = "doji"  # 十字星
    HAMMER = "hammer"  # 锤头
    SHOOTING_STAR = "shooting_star"  # 流星
    ENGULFING_BULL = "engulfing_bull"  # 看涨吞没
    ENGULFING_BEAR = "engulfing_bear"  # 看跌吞没
    INSIDE_BAR = "inside_bar"  # 内包线
    OUTSIDE_BAR = "outside_bar"  # 外包线
    PINBAR_BULL = "pinbar_bull"  # 看涨Pinbar
    PINBAR_BEAR = "pinbar_bear"  # 看跌Pinbar


class MarketStructure(Enum):
    """市场结构枚举"""
    HIGHER_HIGHS_HIGHER_LOWS = "hh_hl"  # 高高低低(上升趋势)
    LOWER_HIGHS_LOWER_LOWS = "lh_ll"    # 低高低低(下降趋势)
    SIDEWAYS = "sideways"                # 横盘
    STRUCTURE_BREAK = "structure_break"   # 结构破坏


@dataclass
class PriceActionSignal:
    """价格行为信号"""
    signal_type: str
    strength: float  # 信号强度 (0-1)
    confidence: float  # 信心度 (0-1)
    entry_price: float
    stop_loss: float
    take_profit: float
    description: str


class PriceActionAnalyzer:
    """价格行为分析器"""
    
    def __init__(self, lookback_period: int = 20):
        """
        初始化价格行为分析器
        
        Args:
            lookback_period: 回溯周期，用于分析市场结构
        """
        self.lookback_period = lookback_period
    
    def analyze_candle_patterns(self, ohlc_data: pd.DataFrame, 
                               num_candles: int = 3) -> List[CandlePattern]:
        """
        分析K线形态
        
        Args:
            ohlc_data: OHLC数据
            num_candles: 分析最近几根K线
            
        Returns:
            检测到的K线形态列表
        """
        patterns = []
        
        if len(ohlc_data) < num_candles:
            return patterns
            
        recent_data = ohlc_data.tail(num_candles)
        
        for i in range(len(recent_data)):
            candle = recent_data.iloc[i]
            pattern = self._identify_single_candle_pattern(candle)
            if pattern:
                patterns.append(pattern)
        
        # 检查多K线形态
        if len(recent_data) >= 2:
            multi_pattern = self._identify_multi_candle_pattern(recent_data.tail(2))
            if multi_pattern:
                patterns.append(multi_pattern)
                
        return patterns
    
    def _identify_single_candle_pattern(self, candle: pd.Series) -> Optional[CandlePattern]:
        """识别单根K线形态"""
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return None
            
        body_ratio = body / total_range
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # 十字星: 实体很小
        if body_ratio < 0.1:
            return CandlePattern.DOJI
            
        # 锤头: 下影线长，上影线短，实体在上半部分
        if (lower_shadow > body * 2 and 
            upper_shadow < body * 0.5 and
            max(open_price, close_price) > (high_price + low_price) / 2):
            return CandlePattern.HAMMER
            
        # 流星: 上影线长，下影线短，实体在下半部分
        if (upper_shadow > body * 2 and 
            lower_shadow < body * 0.5 and
            min(open_price, close_price) < (high_price + low_price) / 2):
            return CandlePattern.SHOOTING_STAR
            
        # Pinbar识别
        if body_ratio < 0.3:  # 实体相对较小
            if lower_shadow > upper_shadow * 2 and lower_shadow > body:
                return CandlePattern.PINBAR_BULL
            elif upper_shadow > lower_shadow * 2 and upper_shadow > body:
                return CandlePattern.PINBAR_BEAR
                
        return None
    
    def _identify_multi_candle_pattern(self, candles: pd.DataFrame) -> Optional[CandlePattern]:
        """识别多根K线形态"""
        if len(candles) < 2:
            return None
            
        prev_candle = candles.iloc[0]
        curr_candle = candles.iloc[1]
        
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        
        # 吞没形态
        if (curr_candle['open'] < prev_candle['close'] and 
            curr_candle['close'] > prev_candle['open'] and
            curr_body > prev_body * 1.2):  # 当前实体明显大于前一根
            return CandlePattern.ENGULFING_BULL
            
        if (curr_candle['open'] > prev_candle['close'] and 
            curr_candle['close'] < prev_candle['open'] and
            curr_body > prev_body * 1.2):
            return CandlePattern.ENGULFING_BEAR
            
        # 内包线: 当前K线完全在前一根K线内
        if (curr_candle['high'] <= prev_candle['high'] and 
            curr_candle['low'] >= prev_candle['low']):
            return CandlePattern.INSIDE_BAR
            
        # 外包线: 当前K线完全包含前一根K线
        if (curr_candle['high'] >= prev_candle['high'] and 
            curr_candle['low'] <= prev_candle['low'] and
            curr_body > prev_body):
            return CandlePattern.OUTSIDE_BAR
            
        return None
    
    def analyze_market_structure(self, ohlc_data: pd.DataFrame) -> MarketStructure:
        """
        分析市场结构
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            市场结构类型
        """
        if len(ohlc_data) < self.lookback_period:
            return MarketStructure.SIDEWAYS
            
        recent_data = ohlc_data.tail(self.lookback_period)
        
        # 找出高点和低点
        highs = self._find_swing_highs(recent_data)
        lows = self._find_swing_lows(recent_data)
        
        if len(highs) < 2 or len(lows) < 2:
            return MarketStructure.SIDEWAYS
            
        # 分析高点趋势
        recent_highs = sorted(highs.items(), key=lambda x: x[0])[-2:]
        recent_lows = sorted(lows.items(), key=lambda x: x[0])[-2:]
        
        higher_highs = recent_highs[1][1] > recent_highs[0][1]
        higher_lows = recent_lows[1][1] > recent_lows[0][1]
        lower_highs = recent_highs[1][1] < recent_highs[0][1]
        lower_lows = recent_lows[1][1] < recent_lows[0][1]
        
        if higher_highs and higher_lows:
            return MarketStructure.HIGHER_HIGHS_HIGHER_LOWS
        elif lower_highs and lower_lows:
            return MarketStructure.LOWER_HIGHS_LOWER_LOWS
        else:
            return MarketStructure.SIDEWAYS
    
    def _find_swing_highs(self, data: pd.DataFrame, window: int = 3) -> Dict[int, float]:
        """寻找摆动高点"""
        highs = {}
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            is_swing_high = True
            
            # 检查左右窗口
            for j in range(i - window, i + window + 1):
                if j != i and data['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                highs[i] = current_high
                
        return highs
    
    def _find_swing_lows(self, data: pd.DataFrame, window: int = 3) -> Dict[int, float]:
        """寻找摆动低点"""
        lows = {}
        for i in range(window, len(data) - window):
            current_low = data['low'].iloc[i]
            is_swing_low = True
            
            # 检查左右窗口
            for j in range(i - window, i + window + 1):
                if j != i and data['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                lows[i] = current_low
                
        return lows
    
    def find_support_resistance_levels(self, ohlc_data: pd.DataFrame, 
                                     touch_threshold: float = 0.002) -> Dict[str, List[float]]:
        """
        基于价格行为寻找支撑阻力位
        
        Args:
            ohlc_data: OHLC数据
            touch_threshold: 触及阈值(价格容差)
            
        Returns:
            支撑阻力位字典
        """
        if len(ohlc_data) < self.lookback_period:
            return {'support': [], 'resistance': []}
            
        recent_data = ohlc_data.tail(self.lookback_period * 2)  # 使用更多数据寻找关键位
        
        # 寻找重要的高低点
        highs = self._find_swing_highs(recent_data, window=2)
        lows = self._find_swing_lows(recent_data, window=2)
        
        # 聚类相近的价格水平
        resistance_levels = self._cluster_price_levels(list(highs.values()), touch_threshold)
        support_levels = self._cluster_price_levels(list(lows.values()), touch_threshold)
        
        # 验证支撑阻力位的有效性
        current_price = ohlc_data['close'].iloc[-1]
        
        valid_resistance = [level for level in resistance_levels if level > current_price]
        valid_support = [level for level in support_levels if level < current_price]
        
        return {
            'support': sorted(valid_support, reverse=True)[:3],  # 最近的3个支撑位
            'resistance': sorted(valid_resistance)[:3]  # 最近的3个阻力位
        }
    
    def _cluster_price_levels(self, prices: List[float], threshold: float) -> List[float]:
        """聚类相近的价格水平"""
        if not prices:
            return []
            
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 2:  # 至少2次触及才认为是有效位置
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [price]
        
        # 处理最后一个聚类
        if len(current_cluster) >= 2:
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        return clusters
    
    def generate_price_action_signals(self, ohlc_data: pd.DataFrame, 
                                    ema_trend: TrendDirection) -> List[PriceActionSignal]:
        """
        基于价格行为生成交易信号
        
        Args:
            ohlc_data: OHLC数据
            ema_trend: EMA趋势方向
            
        Returns:
            价格行为信号列表
        """
        signals = []
        
        if len(ohlc_data) < 10:
            return signals
            
        # 分析最近的K线形态
        patterns = self.analyze_candle_patterns(ohlc_data, num_candles=3)
        market_structure = self.analyze_market_structure(ohlc_data)
        sr_levels = self.find_support_resistance_levels(ohlc_data)
        
        current_candle = ohlc_data.iloc[-1]
        current_price = current_candle['close']
        
        # 基于趋势和价格行为生成信号
        for pattern in patterns:
            signal = self._pattern_to_signal(pattern, current_candle, ema_trend, 
                                           market_structure, sr_levels)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _pattern_to_signal(self, pattern: CandlePattern, current_candle: pd.Series,
                          ema_trend: TrendDirection, market_structure: MarketStructure,
                          sr_levels: Dict[str, List[float]]) -> Optional[PriceActionSignal]:
        """将K线形态转换为交易信号"""
        
        current_price = current_candle['close']
        high_price = current_candle['high']
        low_price = current_candle['low']
        
        # 只在趋势方向一致时生成信号
        if pattern in [CandlePattern.HAMMER, CandlePattern.PINBAR_BULL, CandlePattern.ENGULFING_BULL]:
            if ema_trend in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]:
                return PriceActionSignal(
                    signal_type="多头信号",
                    strength=0.8,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=low_price * 0.998,  # 止损位略低于最低点
                    take_profit=current_price * 1.015,  # 1.5%目标
                    description=f"看涨{pattern.value}形态 + {ema_trend.value}趋势"
                )
        
        elif pattern in [CandlePattern.SHOOTING_STAR, CandlePattern.PINBAR_BEAR, CandlePattern.ENGULFING_BEAR]:
            if ema_trend in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]:
                return PriceActionSignal(
                    signal_type="空头信号",
                    strength=0.8,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=high_price * 1.002,  # 止损位略高于最高点
                    take_profit=current_price * 0.985,  # 1.5%目标
                    description=f"看跌{pattern.value}形态 + {ema_trend.value}趋势"
                )
        
        return None
