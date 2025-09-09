"""
趋势分析器
整合EMA、价格行为和趋势延续分析
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from loguru import logger
from dataclasses import dataclass

from app.models.trend_types import TrendDirection, TrendPhase, TrendAnalysis
from app.analysis.trend.ema_analyzer import EMAAnalyzer
from app.analysis.trend.price_action_analyzer import PriceActionAnalyzer, PriceActionSignal
from app.analysis.trend.trend_continuation_analyzer import TrendContinuationAnalyzer, ContinuationAnalysis


@dataclass
class EnhancedTrendAnalysis:
    """趋势分析结果"""
    # 基础趋势分析
    basic_trend: TrendAnalysis
    
    # 价格行为分析
    price_action_signals: List[PriceActionSignal]
    key_support_resistance: Dict[str, List[float]]
    
    # 趋势延续分析
    continuation_analysis: ContinuationAnalysis
    
    # 综合评分
    overall_signal_strength: float  # 综合信号强度 (0-1)
    trend_quality_score: float     # 趋势质量评分 (0-1)
    entry_timing_score: float      # 入场时机评分 (0-1)


class EnhancedTrendAnalyzer:
    """
    增强版趋势分析器
    结合技术指标、价格行为和延续性分析
    """
    
    def __init__(self, window_obs: int = 80):
        """
        初始化增强版趋势分析器
        
        Args:
            window_obs: 观测窗口大小
        """
        self.window_obs = window_obs
        self.ema_analyzer = EMAAnalyzer(window_obs)
        self.price_action_analyzer = PriceActionAnalyzer(lookback_period=20)
        self.continuation_analyzer = TrendContinuationAnalyzer()
    
    def analyze_comprehensive_trend(self, observed_ohlc: pd.DataFrame) -> Optional[EnhancedTrendAnalysis]:
        """
        综合趋势分析 - 专为顺势短线交易设计
        
        Args:
            observed_ohlc: 观测周期的OHLC数据
            
        Returns:
            增强版趋势分析结果
        """
        try:
            if len(observed_ohlc) < 144:  # 确保有足够数据计算EMA144
                logger.warning("数据长度不足，无法进行综合分析")
                return None
            
            # 1. 基础EMA趋势分析
            basic_trend = self.ema_analyzer.analyze_trend(observed_ohlc)
            if not basic_trend:
                logger.error("基础趋势分析失败")
                return None
            
            # 2. 价格行为分析
            price_action_signals = self.price_action_analyzer.generate_price_action_signals(
                observed_ohlc, basic_trend.direction
            )
            
            # 3. 支撑阻力位分析
            key_sr_levels = self.price_action_analyzer.find_support_resistance_levels(observed_ohlc)
            
            # 4. 趋势延续性分析
            continuation_analysis = self.continuation_analyzer.analyze_continuation(
                observed_ohlc, basic_trend.direction, basic_trend.phase
            )
            
            # 5. 综合评分
            overall_strength, trend_quality, entry_timing = self._calculate_comprehensive_score(
                basic_trend, price_action_signals, continuation_analysis
            )
            
            return EnhancedTrendAnalysis(
                basic_trend=basic_trend,
                price_action_signals=price_action_signals,
                key_support_resistance=key_sr_levels,
                continuation_analysis=continuation_analysis,
                overall_signal_strength=overall_strength,
                trend_quality_score=trend_quality,
                entry_timing_score=entry_timing
            )
            
        except Exception as e:
            logger.error(f"综合趋势分析错误: {e}")
            return None
    
    def _calculate_comprehensive_score(self, basic_trend: TrendAnalysis, 
                                     price_signals: List[PriceActionSignal],
                                     continuation: ContinuationAnalysis) -> tuple:
        """
        计算综合评分
        
        Returns:
            (综合信号强度, 趋势质量评分, 入场时机评分)
        """
        # 1. 基础趋势强度评分
        base_score = basic_trend.confidence
        
        # 2. 价格行为确认评分
        price_action_score = 0.0
        if price_signals:
            avg_pa_strength = sum(signal.strength for signal in price_signals) / len(price_signals)
            avg_pa_confidence = sum(signal.confidence for signal in price_signals) / len(price_signals)
            price_action_score = (avg_pa_strength + avg_pa_confidence) / 2
        
        # 3. 趋势延续确认评分
        continuation_score = continuation.continuation_probability
        
        # 4. 综合信号强度 (权重分配)
        overall_strength = (
            base_score * 0.4 +           # 基础趋势 40%
            price_action_score * 0.3 +   # 价格行为 30%
            continuation_score * 0.3     # 延续性 30%
        )
        
        # 5. 趋势质量评分 (考虑趋势阶段和EMA排列)
        trend_quality = self._calculate_trend_quality(basic_trend, continuation)
        
        # 6. 入场时机评分 (基于延续性分析和回调质量)
        entry_timing = self._calculate_entry_timing_score(continuation)
        
        return overall_strength, trend_quality, entry_timing
    
    def _calculate_trend_quality(self, basic_trend: TrendAnalysis, 
                                continuation: ContinuationAnalysis) -> float:
        """
        计算趋势质量评分
        
        Args:
            basic_trend: 基础趋势分析
            continuation: 延续性分析
            
        Returns:
            趋势质量评分 (0-1)
        """
        quality_factors = []
        
        # EMA排列质量
        if "多头排列" in basic_trend.ema_alignment or "空头排列" in basic_trend.ema_alignment:
            quality_factors.append(0.9)
        elif "偏多" in basic_trend.ema_alignment or "偏空" in basic_trend.ema_alignment:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # 趋势阶段质量
        if basic_trend.phase == TrendPhase.BEGINNING:
            quality_factors.append(0.9)  # 趋势开始阶段质量最高
        elif basic_trend.phase == TrendPhase.ACCELERATION:
            quality_factors.append(0.8)  # 加速阶段质量很高
        elif basic_trend.phase == TrendPhase.MATURITY:
            quality_factors.append(0.6)  # 成熟阶段质量中等
        else:  # EXHAUSTION
            quality_factors.append(0.2)  # 衰竭阶段质量低
        
        # 动量质量
        quality_factors.append(continuation.momentum_strength)
        
        # 成交量确认
        if continuation.volume_confirmation:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_entry_timing_score(self, continuation: ContinuationAnalysis) -> float:
        """
        计算入场时机评分
        
        Args:
            continuation: 延续性分析
            
        Returns:
            入场时机评分 (0-1)
        """
        timing_score = 0.5  # 基础分数
        
        # 根据入场时机调整
        if continuation.entry_timing == "立即入场":
            timing_score = 0.9
        elif continuation.entry_timing == "谨慎入场":
            timing_score = 0.7
        elif continuation.entry_timing == "等待突破":
            timing_score = 0.6
        else:  # 观望
            timing_score = 0.2
        
        # 回调质量调整
        if "微幅" in continuation.pullback_quality:
            timing_score = min(timing_score + 0.1, 1.0)
        elif "浅度" in continuation.pullback_quality:
            timing_score = min(timing_score + 0.05, 1.0)
        elif "深度" in continuation.pullback_quality:
            timing_score = max(timing_score - 0.2, 0.0)
        
        # 突破潜力调整
        if continuation.breakout_potential > 0.7:
            timing_score = min(timing_score + 0.1, 1.0)
        
        return timing_score
    
    def get_trend_summary(self, analysis: EnhancedTrendAnalysis) -> str:
        """
        生成趋势分析摘要
        
        Args:
            analysis: 增强版趋势分析结果
            
        Returns:
            趋势分析摘要文本
        """
        summary = f"""
📊 【综合趋势分析报告】
{'='*50}

🎯 核心趋势信息
▫️ 趋势方向: {analysis.basic_trend.direction.value}
▫️ 趋势阶段: {analysis.basic_trend.phase.value}  
▫️ 趋势强度: {analysis.basic_trend.strength:.1%}
▫️ 趋势质量: {analysis.trend_quality_score:.1%}

📈 综合评分
▫️ 综合信号强度: {analysis.overall_signal_strength:.1%}
▫️ 入场时机评分: {analysis.entry_timing_score:.1%}
▫️ 延续概率: {analysis.continuation_analysis.continuation_probability:.1%}
▫️ 延续信号: {analysis.continuation_analysis.signal.value}

📋 技术指标状态
▫️ EMA排列: {analysis.basic_trend.ema_alignment}
▫️ 收敛状态: {analysis.basic_trend.convergence_status}
▫️ MACD信号: {analysis.basic_trend.macd_signal}
▫️ 布林带位置: {analysis.basic_trend.bollinger_position}

🔄 趋势延续分析
▫️ 入场时机: {analysis.continuation_analysis.entry_timing}
▫️ 回调质量: {analysis.continuation_analysis.pullback_quality}
▫️ 动量强度: {analysis.continuation_analysis.momentum_strength:.1%}
▫️ 成交量确认: {'✅' if analysis.continuation_analysis.volume_confirmation else '❌'}
▫️ 突破潜力: {analysis.continuation_analysis.breakout_potential:.1%}

🎯 关键价位
▫️ 支撑位: {', '.join([f'{level:.4f}' for level in analysis.key_support_resistance.get('support', [])])}
▫️ 阻力位: {', '.join([f'{level:.4f}' for level in analysis.key_support_resistance.get('resistance', [])])}
▫️ 关键位: {analysis.basic_trend.support_resistance['key_level']:.4f}

💡 价格行为信号
"""
        
        if analysis.price_action_signals:
            for signal in analysis.price_action_signals:
                summary += f"▫️ {signal.description} (强度: {signal.strength:.1%}, 信心: {signal.confidence:.1%})\n"
        else:
            summary += "▫️ 暂无明确价格行为信号\n"
        
        # 添加总体评价
        if analysis.overall_signal_strength > 0.8:
            summary += f"\n✅ 总体评价: 强势趋势，信号质量高"
        elif analysis.overall_signal_strength > 0.6:
            summary += f"\n⚠️ 总体评价: 中等强度趋势，需要关注"
        else:
            summary += f"\n❌ 总体评价: 趋势较弱，建议观望"
        
        return summary
