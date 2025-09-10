import talib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger

from app.models.trend_types import TrendDirection, TrendPhase, TrendAnalysis


class EMAAnalyzer:
    """EMA趋势分析器"""
    
    def __init__(self, window_obs: int = 100):
        """
        初始化EMA分析器
        
        Args:
            window_obs: 观测窗口大小，默认80
        """
        self.window_obs = window_obs
        self.ema_periods = {
            'ema_21': 21,
            'ema_55': 55,
            'ema_144': 144
        }
        
    def calculate_emas(self, ohlc_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算EMA指标
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            包含各EMA的字典
        """
        try:
            emas = {}
            close_prices = ohlc_data['close'].values
            
            for ema_name, period in self.ema_periods.items():
                emas[ema_name] = talib.EMA(close_prices, timeperiod=period)
                
            return emas
            
        except Exception as e:
            logger.error(f"计算EMA时发生错误: {e}")
            return {}
    
    def calculate_macd(self, ohlc_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        Args:
            ohlc_data: OHLC数据
            
        Returns:
            包含MACD线、信号线、柱状图的字典
        """
        try:
            close_prices = ohlc_data['close'].values
            macd_line, macd_signal, macd_histogram = talib.MACD(close_prices)
            
            return {
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram
            }
            
        except Exception as e:
            logger.error(f"计算MACD时发生错误: {e}")
            return {}
    
    def calculate_bollinger_bands(self, ohlc_data: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带指标
        
        Args:
            ohlc_data: OHLC数据
            period: 周期，默认20
            std_dev: 标准差倍数，默认2.0
            
        Returns:
            包含上轨、中轨、下轨的字典
        """
        try:
            close_prices = ohlc_data['close'].values
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=period, 
                                              nbdevup=std_dev, nbdevdn=std_dev)
            
            return {
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower
            }
            
        except Exception as e:
            logger.error(f"计算布林带时发生错误: {e}")
            return {}
    
    def analyze_ema_alignment(self, emas: Dict[str, pd.Series], 
                            window_data: pd.DataFrame) -> Tuple[str, float]:
        """
        增强版EMA排列分析 - 考虑趋势方向、斜率和持续性
        
        Args:
            emas: EMA数据
            window_data: 窗口数据
            
        Returns:
            (排列状态, 排列强度)
        """
        try:
            # 获取最新的EMA值
            latest_ema_21 = float(emas['ema_21'][-1])
            latest_ema_55 = float(emas['ema_55'][-1])
            latest_ema_144 = float(emas['ema_144'][-1])
            latest_price = float(window_data['close'].iloc[-1])
            
            # 计算EMA斜率 (最近5-10个周期的变化率)
            slope_window = min(10, len(emas['ema_21']) - 1)
            if slope_window >= 5:
                ema21_slope = (emas['ema_21'][-1] - emas['ema_21'][-slope_window]) / slope_window
                ema55_slope = (emas['ema_55'][-1] - emas['ema_55'][-slope_window]) / slope_window
                ema144_slope = (emas['ema_144'][-1] - emas['ema_144'][-slope_window]) / slope_window
            else:
                ema21_slope = ema55_slope = ema144_slope = 0
            
            # 计算排列持续性 (最近N个周期的排列稳定性)
            consistency_window = min(20, len(emas['ema_21']))
            bullish_count = 0  # 多头排列次数
            bearish_count = 0  # 空头排列次数
            
            for i in range(1, consistency_window + 1):
                try:
                    p = float(window_data['close'].iloc[-i])
                    e21 = float(emas['ema_21'][-i])
                    e55 = float(emas['ema_55'][-i])
                    e144 = float(emas['ema_144'][-i])
                    
                    if p > e21 > e55 > e144:
                        bullish_count += 1
                    elif p < e21 < e55 < e144:
                        bearish_count += 1
                except:
                    continue
            
            # 计算持续性比例
            consistency_ratio = max(bullish_count, bearish_count) / consistency_window
            
            # 判断基础排列状态
            perfect_bull = latest_price > latest_ema_21 > latest_ema_55 > latest_ema_144
            perfect_bear = latest_price < latest_ema_21 < latest_ema_55 < latest_ema_144
            partial_bull = latest_ema_21 > latest_ema_55 > latest_ema_144
            partial_bear = latest_ema_21 < latest_ema_55 < latest_ema_144
            
            # 计算综合强度 (0-1之间)
            strength = 0.0
            alignment = "混乱排列"
            
            if perfect_bull:
                alignment = "完美多头排列"
                # 基础强度
                strength = 0.7
                # 斜率加分 (所有EMA都向上)
                if ema21_slope > 0 and ema55_slope > 0 and ema144_slope > 0:
                    strength += 0.2
                # 持续性加分
                strength += consistency_ratio * 0.1
                # EMA间距加分 (间距越大越强)
                spacing_factor = min((latest_ema_21 - latest_ema_144) / latest_ema_144, 0.05) * 2
                strength += spacing_factor
                
            elif perfect_bear:
                alignment = "完美空头排列"
                strength = 0.7
                if ema21_slope < 0 and ema55_slope < 0 and ema144_slope < 0:
                    strength += 0.2
                strength += consistency_ratio * 0.1
                spacing_factor = min((latest_ema_144 - latest_ema_21) / latest_ema_144, 0.05) * 2
                strength += spacing_factor
                
            elif partial_bull:
                alignment = "偏多排列"
                strength = 0.4
                if ema21_slope > 0 and ema55_slope > 0:
                    strength += 0.15
                strength += consistency_ratio * 0.1
                # 价格位置调整
                if latest_price > latest_ema_21:
                    strength += 0.15
                    
            elif partial_bear:
                alignment = "偏空排列"
                strength = 0.4
                if ema21_slope < 0 and ema55_slope < 0:
                    strength += 0.15
                strength += consistency_ratio * 0.1
                if latest_price < latest_ema_21:
                    strength += 0.15
            else:
                # 混乱排列 - 检查是否在转折中
                if abs(ema21_slope) > abs(ema55_slope) and abs(ema21_slope) > abs(ema144_slope):
                    if ema21_slope > 0:
                        alignment = "可能转多"
                        strength = 0.3
                    else:
                        alignment = "可能转空"
                        strength = 0.3
                else:
                    alignment = "混乱排列"
                    strength = 0.1
            
            # 强度限制在0-1之间
            strength = max(0.0, min(1.0, strength))
            
            return alignment, strength
            
        except Exception as e:
            logger.error(f"分析EMA排列时发生错误: {e}")
            return "未知", 0.0
    
    def analyze_ema_convergence(self, emas: Dict[str, pd.Series]) -> Tuple[str, float]:
        """
        分析EMA收敛/发散状态
        
        Args:
            emas: EMA数据
            
        Returns:
            (收敛/发散状态, 收敛/发散强度)
        """
        try:
            # 计算EMA之间的距离变化
            window_size = min(20, len(emas['ema_21']) - 1)
            
            # 计算21-55之间的距离变化
            distance_21_55_recent = np.abs(emas['ema_21'][-window_size:] - emas['ema_55'][-window_size:])
            distance_55_144_recent = np.abs(emas['ema_55'][-window_size:] - emas['ema_144'][-window_size:])
            
            # 计算距离变化趋势
            trend_21_55 = np.polyfit(range(window_size), distance_21_55_recent, 1)[0]
            trend_55_144 = np.polyfit(range(window_size), distance_55_144_recent, 1)[0]
            
            avg_trend = (trend_21_55 + trend_55_144) / 2
            
            if avg_trend > 0.001:
                status = "发散"
                strength = min(abs(avg_trend) * 1000, 1.0)
            elif avg_trend < -0.001:
                status = "收敛"
                strength = min(abs(avg_trend) * 1000, 1.0)
            else:
                status = "平衡"
                strength = 0.5
                
            return status, strength
            
        except Exception as e:
            logger.error(f"分析EMA收敛/发散时发生错误: {e}")
            return "未知", 0.0
    
    def analyze_macd_momentum(self, macd_data: Dict[str, pd.Series]) -> Tuple[str, float]:
        """
        分析MACD动量
        
        Args:
            macd_data: MACD数据
            
        Returns:
            (MACD信号, 信号强度)
        """
        try:
            macd_line = macd_data['macd_line']
            macd_signal = macd_data['macd_signal']
            macd_histogram = macd_data['macd_histogram']
            
            # 获取最新值
            latest_macd = macd_line[-1]
            latest_signal = macd_signal[-1]
            latest_histogram = macd_histogram[-1]
            
            # 判断金叉死叉
            prev_macd = macd_line[-2] if len(macd_line) > 1 else latest_macd
            prev_signal = macd_signal[-2] if len(macd_signal) > 1 else latest_signal
            
            signal = ""
            strength = 0.0
            
            # 金叉信号
            if prev_macd <= prev_signal and latest_macd > latest_signal:
                signal = "MACD金叉"
                strength = 0.8
            # 死叉信号
            elif prev_macd >= prev_signal and latest_macd < latest_signal:
                signal = "MACD死叉"
                strength = 0.8
            # 零轴上方
            elif latest_macd > 0 and latest_signal > 0:
                signal = "MACD零轴上方"
                strength = 0.6
            # 零轴下方
            elif latest_macd < 0 and latest_signal < 0:
                signal = "MACD零轴下方"
                strength = 0.6
            else:
                signal = "MACD中性"
                strength = 0.3
                
            # 根据柱状图调整强度
            if abs(latest_histogram) > abs(macd_histogram[-2]):
                strength = min(strength + 0.1, 1.0)
                
            return signal, strength
            
        except Exception as e:
            logger.error(f"分析MACD动量时发生错误: {e}")
            return "MACD未知", 0.0
    
    def analyze_bollinger_position(self, bollinger_data: Dict[str, pd.Series], 
                                 current_price: float) -> Tuple[str, float]:
        """
        分析布林带位置
        
        Args:
            bollinger_data: 布林带数据
            current_price: 当前价格
            
        Returns:
            (布林带位置, 位置强度)
        """
        try:
            bb_upper = bollinger_data['bb_upper'][-1]
            bb_middle = bollinger_data['bb_middle'][-1]
            bb_lower = bollinger_data['bb_lower'][-1]
            
            # 计算相对位置
            bb_range = bb_upper - bb_lower
            if bb_range == 0:
                return "布林带异常", 0.0
                
            relative_position = (current_price - bb_lower) / bb_range
            
            if relative_position > 0.8:
                position = "接近上轨"
                strength = relative_position
            elif relative_position > 0.6:
                position = "上轨区域"
                strength = relative_position
            elif relative_position > 0.4:
                position = "中轨区域"
                strength = 0.5
            elif relative_position > 0.2:
                position = "下轨区域"
                strength = 1.0 - relative_position
            else:
                position = "接近下轨"
                strength = 1.0 - relative_position
                
            return position, strength
            
        except Exception as e:
            logger.error(f"分析布林带位置时发生错误: {e}")
            return "布林带未知", 0.0
    
    def determine_trend_direction(self, ema_alignment: str, ema_strength: float,
                                macd_signal: str, macd_strength: float) -> TrendDirection:
        """
        确定趋势方向
        
        Args:
            ema_alignment: EMA排列状态
            ema_strength: EMA排列强度
            macd_signal: MACD信号
            macd_strength: MACD强度
            
        Returns:
            趋势方向
        """
        # 多头趋势判断
        if "多头" in ema_alignment and ema_strength >= 0.7:
            if "金叉" in macd_signal or "零轴上方" in macd_signal:
                return TrendDirection.STRONG_UPTREND
            else:
                return TrendDirection.WEAK_UPTREND
        
        # 空头趋势判断
        elif "空头" in ema_alignment and ema_strength >= 0.7:
            if "死叉" in macd_signal or "零轴下方" in macd_signal:
                return TrendDirection.STRONG_DOWNTREND
            else:
                return TrendDirection.WEAK_DOWNTREND
        
        # 偏多偏空判断
        elif "偏多" in ema_alignment:
            return TrendDirection.WEAK_UPTREND
        elif "偏空" in ema_alignment:
            return TrendDirection.WEAK_DOWNTREND
        
        # 默认横盘
        else:
            return TrendDirection.SIDEWAYS
    
    def determine_trend_phase(self, convergence_status: str, convergence_strength: float,
                            trend_direction: TrendDirection) -> TrendPhase:
        """
        确定趋势阶段
        
        Args:
            convergence_status: 收敛/发散状态
            convergence_strength: 收敛/发散强度
            trend_direction: 趋势方向
            
        Returns:
            趋势阶段
        """
        if trend_direction == TrendDirection.SIDEWAYS:
            return TrendPhase.MATURITY
            
        if convergence_status == "发散" and convergence_strength > 0.6:
            return TrendPhase.BEGINNING
        elif convergence_status == "发散" and convergence_strength > 0.3:
            return TrendPhase.ACCELERATION
        elif convergence_status == "收敛" and convergence_strength > 0.6:
            return TrendPhase.EXHAUSTION
        else:
            return TrendPhase.MATURITY
    
    def analyze_trend(self, observed_ohlc: pd.DataFrame) -> Optional[TrendAnalysis]:
        """
        综合趋势分析
        
        Args:
            observed_ohlc: 观测周期的OHLC数据
            
        Returns:
            趋势分析结果
        """
        try:
            if len(observed_ohlc) < max(self.ema_periods.values()):
                logger.warning(f"数据长度不足，需要至少{max(self.ema_periods.values())}条数据")
                return None
            
            # 获取窗口数据
            window_data = observed_ohlc.tail(self.window_obs)
            
            # 计算技术指标
            emas = self.calculate_emas(observed_ohlc)
            macd_data = self.calculate_macd(observed_ohlc)
            bollinger_data = self.calculate_bollinger_bands(observed_ohlc)
            
            if not emas or not macd_data or not bollinger_data:
                logger.error("技术指标计算失败")
                return None
            
            # 获取窗口EMA数据
            window_emas = {}
            for name, ema_series in emas.items():
                window_emas[name] = ema_series[-self.window_obs:]
            
            # 分析各项指标
            ema_alignment, ema_strength = self.analyze_ema_alignment(window_emas, window_data)
            convergence_status, convergence_strength = self.analyze_ema_convergence(window_emas)
            macd_signal, macd_strength = self.analyze_macd_momentum(macd_data)
            bb_position, bb_strength = self.analyze_bollinger_position(
                bollinger_data, window_data['close'].iloc[-1]
            )
            
            # 确定趋势方向和阶段
            trend_direction = self.determine_trend_direction(
                ema_alignment, ema_strength, macd_signal, macd_strength
            )
            trend_phase = self.determine_trend_phase(
                convergence_status, convergence_strength, trend_direction
            )
            
            # 计算综合强度和信心度
            trend_strength = (ema_strength + macd_strength + bb_strength) / 3
            confidence = min(trend_strength * 1.2, 1.0)
            
            # 生成入场信号
            entry_signals = []
            if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]:
                if trend_phase == TrendPhase.BEGINNING:
                    entry_signals.append("趋势开始_多头入场")
                elif "金叉" in macd_signal:
                    entry_signals.append("MACD金叉_多头入场")
            elif trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]:
                if trend_phase == TrendPhase.BEGINNING:
                    entry_signals.append("趋势开始_空头入场")
                elif "死叉" in macd_signal:
                    entry_signals.append("MACD死叉_空头入场")
            
            # 支撑阻力位(简化版本，基于EMA)
            support_resistance = {
                'support': min(window_emas['ema_21'][-1], window_emas['ema_55'][-1]),
                'resistance': max(window_emas['ema_21'][-1], window_emas['ema_55'][-1]),
                'key_level': window_emas['ema_144'][-1]
            }
            
            return TrendAnalysis(
                direction=trend_direction,
                phase=trend_phase,
                strength=trend_strength,
                confidence=confidence,
                ema_alignment=ema_alignment,
                convergence_status=convergence_status,
                macd_signal=macd_signal,
                bollinger_position=bb_position,
                support_resistance=support_resistance,
                entry_signals=entry_signals
            )
            
        except Exception as e:
            logger.error(f"趋势分析时发生错误: {e}")
            return None
    
    def get_trend_summary(self, analysis: TrendAnalysis) -> str:
        """
        获取趋势分析摘要
        
        Args:
            analysis: 趋势分析结果
            
        Returns:
            趋势摘要文本
        """
        summary = f"""
📊 趋势分析摘要
================
🎯 趋势方向: {analysis.direction.value}
📈 趋势阶段: {analysis.phase.value}
💪 趋势强度: {analysis.strength:.2f}
🎯 信心度: {analysis.confidence:.2f}

📋 技术指标分析
================
📊 EMA排列: {analysis.ema_alignment}
🔄 收敛状态: {analysis.convergence_status}
📈 MACD信号: {analysis.macd_signal}
📊 布林带位置: {analysis.bollinger_position}

🎯 关键位置
================
🔻 支撑位: {analysis.support_resistance['support']:.4f}
🔺 阻力位: {analysis.support_resistance['resistance']:.4f}
🎯 关键位: {analysis.support_resistance['key_level']:.4f}

🚀 入场信号
================
{chr(10).join([f"• {signal}" for signal in analysis.entry_signals]) if analysis.entry_signals else "• 暂无明确入场信号"}
"""
        return summary
