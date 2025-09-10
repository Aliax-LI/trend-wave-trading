"""
数字货币日内交易核心趋势分析模块
结合重叠研究和动量指标，实现完整的趋势分析逻辑：
趋势方向 → 趋势时期 → 趋势延续

基于TALib库实现高精度技术指标计算
"""

import talib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from loguru import logger
from datetime import datetime
from dataclasses import dataclass

from app.models.trend_types import TrendDirection, TrendPhase, TrendAnalysis


@dataclass
class TrendContinuation:
    """趋势延续分析结果"""
    continuation_probability: float  # 延续概率 0-1
    continuation_strength: str       # 延续强度：强/中/弱
    key_levels: Dict[str, float]     # 关键支撑阻力位
    momentum_score: float           # 动量得分
    volume_confirmation: bool       # 成交量确认
    expected_duration: str          # 预期持续时间


class AdvancedTrendAnalyzer:
    """高级趋势分析器 - 数字货币日内交易专用"""
    
    def __init__(self, config: dict = None):
        """
        初始化高级趋势分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 数字货币专用EMA配置（适应高频波动）
        self.fast_emas = [8, 13, 21]      # 快速EMA - 捕捉短期趋势
        self.standard_emas = [21, 55, 144] # 标准EMA - 确定主趋势
        
        # 动量指标参数
        self.momentum_params = {
            'rsi_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'williams_period': 14,
            'cci_period': 14,
            'adx_period': 14
        }
        
        # 重叠研究参数
        self.overlap_params = {
            'sma_periods': [20, 50, 200],
            'bb_period': 20,
            'bb_std': 2.0,
            'sar_accel': 0.02,
            'sar_max': 0.2
        }
        
        logger.info("🎯 高级趋势分析器初始化完成")
    
    def calculate_overlap_studies(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        计算重叠研究指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            重叠研究指标字典
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            indicators = {}
            
            # 1. 快速EMA系列（8, 13, 21）
            for period in self.fast_emas:
                indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # 2. 标准EMA系列（21, 55, 144）
            for period in self.standard_emas:
                if f'ema_{period}' not in indicators:  # 避免重复计算EMA21
                    indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # 3. SMA系列
            for period in self.overlap_params['sma_periods']:
                indicators[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            
            # 4. 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, 
                timeperiod=self.overlap_params['bb_period'],
                nbdevup=self.overlap_params['bb_std'],
                nbdevdn=self.overlap_params['bb_std']
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # 5. 抛物线SAR
            indicators['sar'] = talib.SAR(
                high, low,
                acceleration=self.overlap_params['sar_accel'],
                maximum=self.overlap_params['sar_max']
            )
            
            # 6. HT_TRENDLINE (希尔伯特变换 - 瞬时趋势线)
            indicators['ht_trendline'] = talib.HT_TRENDLINE(close)
            
            # 7. MIDPOINT (中点)
            indicators['midpoint'] = talib.MIDPOINT(close, timeperiod=14)
            
            # 8. T3 (三重指数移动平均)
            indicators['t3'] = talib.T3(close, timeperiod=14, vfactor=0.7)
            
            logger.debug("✅ 重叠研究指标计算完成")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 计算重叠研究指标失败: {e}")
            return {}
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        计算动量指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            动量指标字典
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            indicators = {}
            
            # 1. RSI (相对强弱指标)
            indicators['rsi'] = talib.RSI(close, timeperiod=self.momentum_params['rsi_period'])
            
            # 2. STOCH (随机指标)
            stoch_k, stoch_d = talib.STOCH(
                high, low, close,
                fastk_period=self.momentum_params['stoch_k'],
                slowk_period=self.momentum_params['stoch_d'],
                slowd_period=self.momentum_params['stoch_d']
            )
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # 3. WILLIAMS %R
            indicators['williams_r'] = talib.WILLR(
                high, low, close,
                timeperiod=self.momentum_params['williams_period']
            )
            
            # 4. CCI (商品通道指标)
            indicators['cci'] = talib.CCI(
                high, low, close,
                timeperiod=self.momentum_params['cci_period']
            )
            
            # 5. ADX/DI (方向性运动指标)
            indicators['adx'] = talib.ADX(
                high, low, close,
                timeperiod=self.momentum_params['adx_period']
            )
            indicators['plus_di'] = talib.PLUS_DI(
                high, low, close,
                timeperiod=self.momentum_params['adx_period']
            )
            indicators['minus_di'] = talib.MINUS_DI(
                high, low, close,
                timeperiod=self.momentum_params['adx_period']
            )
            
            # 6. MACD
            macd_line, macd_signal, macd_histogram = talib.MACD(close)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_histogram
            
            # 7. MOM (动量)
            indicators['momentum'] = talib.MOM(close, timeperiod=10)
            
            # 8. ROC (变动率)
            indicators['roc'] = talib.ROC(close, timeperiod=10)
            
            # 9. BOP (均势指标)
            indicators['bop'] = talib.BOP(data['open'].values, high, low, close)
            
            # 10. 数字货币专用：成交量加权动量
            if len(volume) > 0:
                vwap = np.cumsum(close * volume) / np.cumsum(volume)
                indicators['vwap_momentum'] = (close / vwap - 1) * 100
            
            logger.debug("✅ 动量指标计算完成")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ 计算动量指标失败: {e}")
            return {}
    
    def analyze_trend_direction(self, overlap_data: Dict, momentum_data: Dict, 
                              current_price: float) -> Tuple[TrendDirection, float]:
        """
        分析趋势方向
        
        Args:
            overlap_data: 重叠研究数据
            momentum_data: 动量指标数据
            current_price: 当前价格
            
        Returns:
            (趋势方向, 方向强度)
        """
        try:
            direction_score = 0.0
            total_weight = 0.0
            
            # 1. EMA排列分析 (权重40%)
            ema_score = self._analyze_ema_alignment(overlap_data, current_price)
            direction_score += ema_score * 0.4
            total_weight += 0.4
            
            # 2. SAR位置分析 (权重10%)
            if 'sar' in overlap_data and not np.isnan(overlap_data['sar'][-1]):
                sar_score = 1.0 if current_price > overlap_data['sar'][-1] else -1.0
                direction_score += sar_score * 0.1
                total_weight += 0.1
            
            # 3. 布林带位置分析 (权重10%)
            bb_score = self._analyze_bollinger_trend(overlap_data, current_price)
            if not np.isnan(bb_score):
                direction_score += bb_score * 0.1
                total_weight += 0.1
            
            # 4. 动量指标综合分析 (权重40%)
            momentum_score = self._analyze_momentum_trend(momentum_data)
            direction_score += momentum_score * 0.4
            total_weight += 0.4
            
            # 标准化得分
            if total_weight > 0:
                final_score = direction_score / total_weight
            else:
                final_score = 0.0
            
            # 确定趋势方向 (为数字货币市场调整阈值)
            strength = abs(final_score)
            
            if final_score > 0.4:  # 强势上升趋势
                direction = TrendDirection.STRONG_UPTREND
            elif final_score > 0.15:  # 弱势上升趋势  
                direction = TrendDirection.WEAK_UPTREND
            elif final_score < -0.4:  # 强势下降趋势
                direction = TrendDirection.STRONG_DOWNTREND
            elif final_score < -0.15:  # 弱势下降趋势 (进一步降低阈值)
                direction = TrendDirection.WEAK_DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            logger.debug(f"趋势方向分析: {direction.value}, 强度: {strength:.3f}")
            return direction, strength
            
        except Exception as e:
            logger.error(f"❌ 趋势方向分析失败: {e}")
            return TrendDirection.SIDEWAYS, 0.0
    
    def _analyze_ema_alignment(self, overlap_data: Dict, current_price: float) -> float:
        """分析EMA排列得分"""
        try:
            # 获取关键EMA值
            ema8 = overlap_data.get('ema_8', [np.nan])[-1]
            ema13 = overlap_data.get('ema_13', [np.nan])[-1]
            ema21 = overlap_data.get('ema_21', [np.nan])[-1]
            ema55 = overlap_data.get('ema_55', [np.nan])[-1]
            ema144 = overlap_data.get('ema_144', [np.nan])[-1]
            
            score = 0.0
            
            # 完美多头排列
            if (current_price > ema8 > ema13 > ema21 > ema55 > ema144):
                score = 1.0
            # 强多头排列
            elif (current_price > ema21 > ema55 > ema144):
                score = 0.8
            # 偏多排列
            elif (ema21 > ema55):
                score = 0.4
            # 完美空头排列
            elif (current_price < ema8 < ema13 < ema21 < ema55 < ema144):
                score = -1.0
            # 强空头排列
            elif (current_price < ema21 < ema55 < ema144):
                score = -0.8
            # 偏空排列
            elif (ema21 < ema55):
                score = -0.4
            
            return score
            
        except Exception as e:
            logger.error(f"EMA排列分析失败: {e}")
            return 0.0
    
    def _analyze_bollinger_trend(self, overlap_data: Dict, current_price: float) -> float:
        """分析布林带趋势得分"""
        try:
            bb_upper = overlap_data.get('bb_upper', [np.nan])[-1]
            bb_lower = overlap_data.get('bb_lower', [np.nan])[-1]
            bb_middle = overlap_data.get('bb_middle', [np.nan])[-1]
            
            if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(bb_middle):
                return 0.0
            
            # 布林带宽度
            bb_width = bb_upper - bb_lower
            if bb_width == 0:
                return 0.0
            
            # 价格在布林带中的相对位置
            bb_position = (current_price - bb_lower) / bb_width
            
            if bb_position > 0.8:
                return 0.8  # 强势上升
            elif bb_position > 0.6:
                return 0.4  # 偏强
            elif bb_position < 0.2:
                return -0.8  # 强势下降
            elif bb_position < 0.4:
                return -0.4  # 偏弱
            else:
                return 0.0  # 中性
                
        except Exception as e:
            logger.error(f"布林带趋势分析失败: {e}")
            return 0.0
    
    def _analyze_momentum_trend(self, momentum_data: Dict) -> float:
        """分析动量趋势得分"""
        try:
            score = 0.0
            count = 0
            
            # RSI分析
            rsi = momentum_data.get('rsi', [np.nan])[-1]
            if not np.isnan(rsi):
                if rsi > 60:
                    score += 0.5
                elif rsi > 50:
                    score += 0.2
                elif rsi < 40:
                    score -= 0.5
                elif rsi < 50:
                    score -= 0.2
                count += 1
            
            # MACD分析
            macd_line = momentum_data.get('macd_line', [np.nan])[-1]
            macd_signal = momentum_data.get('macd_signal', [np.nan])[-1]
            if not np.isnan(macd_line) and not np.isnan(macd_signal):
                if macd_line > macd_signal and macd_line > 0:
                    score += 1.0
                elif macd_line > macd_signal:
                    score += 0.5
                elif macd_line < macd_signal and macd_line < 0:
                    score -= 1.0
                elif macd_line < macd_signal:
                    score -= 0.5
                count += 1
            
            # ADX分析
            adx = momentum_data.get('adx', [np.nan])[-1]
            plus_di = momentum_data.get('plus_di', [np.nan])[-1]
            minus_di = momentum_data.get('minus_di', [np.nan])[-1]
            if not np.isnan(adx) and not np.isnan(plus_di) and not np.isnan(minus_di):
                if adx > 25:  # 趋势强度足够
                    if plus_di > minus_di:
                        score += 0.8
                    else:
                        score -= 0.8
                count += 1
            
            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"动量趋势分析失败: {e}")
            return 0.0
    
    def analyze_trend_phase(self, overlap_data: Dict, momentum_data: Dict, 
                          trend_direction: TrendDirection) -> Tuple[TrendPhase, float]:
        """
        分析趋势时期/阶段
        
        Args:
            overlap_data: 重叠研究数据
            momentum_data: 动量指标数据
            trend_direction: 趋势方向
            
        Returns:
            (趋势阶段, 阶段强度)
        """
        try:
            if trend_direction == TrendDirection.SIDEWAYS:
                return TrendPhase.MATURITY, 0.5
            
            phase_score = 0.0
            
            # 1. ADX分析趋势强度
            adx = momentum_data.get('adx', [np.nan])[-1]
            if not np.isnan(adx):
                if adx > 40:
                    phase_score += 0.4  # 强趋势
                elif adx > 25:
                    phase_score += 0.2  # 中等趋势
                else:
                    phase_score -= 0.2  # 弱趋势
            
            # 2. RSI背离分析
            rsi = momentum_data.get('rsi', [np.nan])[-1]
            if not np.isnan(rsi):
                if rsi > 70 or rsi < 30:
                    phase_score -= 0.3  # 可能接近顶部/底部
                elif 40 < rsi < 60:
                    phase_score += 0.2  # 健康的动量
            
            # 3. 成交量确认
            vwap_momentum = momentum_data.get('vwap_momentum', [0])[-1]
            if abs(vwap_momentum) > 2:
                phase_score += 0.2  # 成交量支持
            
            # 4. EMA收敛/发散分析
            ema_divergence = self._analyze_ema_divergence(overlap_data)
            phase_score += ema_divergence * 0.3
            
            # 确定阶段
            if phase_score > 0.5:
                phase = TrendPhase.BEGINNING
            elif phase_score > 0.0:
                phase = TrendPhase.ACCELERATION
            elif phase_score > -0.3:
                phase = TrendPhase.MATURITY
            else:
                phase = TrendPhase.EXHAUSTION
            
            strength = abs(phase_score)
            logger.debug(f"趋势阶段分析: {phase.value}, 强度: {strength:.3f}")
            return phase, strength
            
        except Exception as e:
            logger.error(f"❌ 趋势阶段分析失败: {e}")
            return TrendPhase.MATURITY, 0.0
    
    def _analyze_ema_divergence(self, overlap_data: Dict) -> float:
        """分析EMA发散/收敛"""
        try:
            ema21 = overlap_data.get('ema_21', [])
            ema55 = overlap_data.get('ema_55', [])
            
            if len(ema21) < 10 or len(ema55) < 10:
                return 0.0
            
            # 计算最近10期的EMA距离变化
            recent_distance = np.abs(ema21[-10:] - ema55[-10:])
            distance_trend = np.polyfit(range(10), recent_distance, 1)[0]
            
            # 标准化
            return np.tanh(distance_trend * 1000)  # 发散为正，收敛为负
            
        except Exception as e:
            logger.error(f"EMA发散分析失败: {e}")
            return 0.0
    
    def analyze_trend_continuation(self, overlap_data: Dict, momentum_data: Dict,
                                 trend_direction: TrendDirection, 
                                 trend_phase: TrendPhase,
                                 data: pd.DataFrame) -> TrendContinuation:
        """
        分析趋势延续性
        
        Args:
            overlap_data: 重叠研究数据
            momentum_data: 动量指标数据
            trend_direction: 趋势方向
            trend_phase: 趋势阶段
            data: 原始数据
            
        Returns:
            趋势延续分析结果
        """
        try:
            continuation_prob = 0.5  # 基础概率
            momentum_score = 0.0
            
            # 1. 基于趋势阶段的基础概率
            phase_prob_map = {
                TrendPhase.BEGINNING: 0.8,
                TrendPhase.ACCELERATION: 0.7,
                TrendPhase.MATURITY: 0.5,
                TrendPhase.EXHAUSTION: 0.2
            }
            continuation_prob = phase_prob_map.get(trend_phase, 0.5)
            
            # 2. 动量确认
            momentum_score = self._calculate_momentum_score(momentum_data)
            continuation_prob += momentum_score * 0.2
            
            # 3. 成交量确认
            volume_confirmation = self._check_volume_confirmation(data, trend_direction)
            if volume_confirmation:
                continuation_prob += 0.1
            
            # 4. 关键技术位分析
            key_levels = self._identify_key_levels(overlap_data, data)
            
            # 5. 限制概率范围
            continuation_prob = max(0.0, min(1.0, continuation_prob))
            
            # 确定延续强度
            if continuation_prob > 0.7:
                strength = "强"
            elif continuation_prob > 0.5:
                strength = "中"
            else:
                strength = "弱"
            
            # 预期持续时间（基于数字货币市场特性）
            duration_map = {
                TrendPhase.BEGINNING: "2-4小时",
                TrendPhase.ACCELERATION: "1-3小时",
                TrendPhase.MATURITY: "30分钟-2小时",
                TrendPhase.EXHAUSTION: "15-60分钟"
            }
            expected_duration = duration_map.get(trend_phase, "不确定")
            
            return TrendContinuation(
                continuation_probability=continuation_prob,
                continuation_strength=strength,
                key_levels=key_levels,
                momentum_score=momentum_score,
                volume_confirmation=volume_confirmation,
                expected_duration=expected_duration
            )
            
        except Exception as e:
            logger.error(f"❌ 趋势延续分析失败: {e}")
            return TrendContinuation(0.5, "弱", {}, 0.0, False, "不确定")
    
    def _calculate_momentum_score(self, momentum_data: Dict) -> float:
        """计算综合动量得分"""
        try:
            scores = []
            
            # RSI动量
            rsi = momentum_data.get('rsi', [np.nan])[-1]
            if not np.isnan(rsi):
                if 40 < rsi < 60:
                    scores.append(0.8)
                elif 30 < rsi < 70:
                    scores.append(0.5)
                else:
                    scores.append(0.2)
            
            # MACD动量
            macd_hist = momentum_data.get('macd_histogram', [np.nan])[-1]
            if not np.isnan(macd_hist):
                if len(momentum_data.get('macd_histogram', [])) > 1:
                    prev_hist = momentum_data['macd_histogram'][-2]
                    if abs(macd_hist) > abs(prev_hist):
                        scores.append(0.8)
                    else:
                        scores.append(0.3)
            
            # CCI动量
            cci = momentum_data.get('cci', [np.nan])[-1]
            if not np.isnan(cci):
                if abs(cci) < 100:
                    scores.append(0.7)
                elif abs(cci) < 200:
                    scores.append(0.4)
                else:
                    scores.append(0.1)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"动量得分计算失败: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, data: pd.DataFrame, trend_direction: TrendDirection) -> bool:
        """检查成交量确认"""
        try:
            if len(data) < 20:
                return False
            
            recent_volume = data['volume'].tail(5).mean()
            avg_volume = data['volume'].tail(20).mean()
            
            # 数字货币市场：成交量应该支持趋势
            volume_ratio = recent_volume / avg_volume
            
            if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.STRONG_DOWNTREND]:
                return volume_ratio > 1.2  # 强趋势需要更多成交量确认
            else:
                return volume_ratio > 0.8   # 弱趋势需要适度成交量
            
        except Exception as e:
            logger.error(f"成交量确认检查失败: {e}")
            return False
    
    def _identify_key_levels(self, overlap_data: Dict, data: pd.DataFrame) -> Dict[str, float]:
        """识别关键技术位"""
        try:
            levels = {}
            current_price = data['close'].iloc[-1]
            
            # EMA关键位
            for period in [21, 55, 144]:
                ema_key = f'ema_{period}'
                if ema_key in overlap_data:
                    ema_value = overlap_data[ema_key][-1]
                    if not np.isnan(ema_value):
                        levels[f'EMA{period}'] = float(ema_value)
            
            # 布林带关键位
            if 'bb_upper' in overlap_data and 'bb_lower' in overlap_data:
                bb_upper = overlap_data['bb_upper'][-1]
                bb_lower = overlap_data['bb_lower'][-1]
                if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                    levels['布林上轨'] = float(bb_upper)
                    levels['布林下轨'] = float(bb_lower)
            
            # SAR关键位
            if 'sar' in overlap_data:
                sar_value = overlap_data['sar'][-1]
                if not np.isnan(sar_value):
                    levels['SAR'] = float(sar_value)
            
            # 近期高低点
            recent_data = data.tail(50)
            levels['近期高点'] = float(recent_data['high'].max())
            levels['近期低点'] = float(recent_data['low'].min())
            
            return levels
            
        except Exception as e:
            logger.error(f"关键位识别失败: {e}")
            return {}
    
    def comprehensive_trend_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        综合趋势分析 - 主入口函数
        
        Args:
            data: OHLCV数据
            
        Returns:
            完整的趋势分析结果
        """
        try:
            min_required = max(self.standard_emas)  # 144
            if len(data) < min_required:
                logger.warning(f"数据长度不足，需要至少{min_required}条数据，当前{len(data)}条")
                return {}
            
            current_price = data['close'].iloc[-1]
            logger.info(f"🎯 开始综合趋势分析，当前价格: {current_price:.4f}")
            
            # 1. 计算所有技术指标
            overlap_data = self.calculate_overlap_studies(data)
            momentum_data = self.calculate_momentum_indicators(data)
            
            if not overlap_data or not momentum_data:
                logger.error("技术指标计算失败")
                return {}
            
            # 2. 趋势方向分析
            trend_direction, direction_strength = self.analyze_trend_direction(
                overlap_data, momentum_data, current_price
            )
            
            # 3. 趋势阶段分析
            trend_phase, phase_strength = self.analyze_trend_phase(
                overlap_data, momentum_data, trend_direction
            )
            
            # 4. 趋势延续分析
            continuation = self.analyze_trend_continuation(
                overlap_data, momentum_data, trend_direction, trend_phase, data
            )
            
            # 5. 生成交易信号
            signals = self._generate_trading_signals(
                trend_direction, trend_phase, continuation, overlap_data, momentum_data
            )
            
            # 6. 构建完整分析结果
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'trend_direction': {
                    'direction': trend_direction.value,
                    'strength': float(direction_strength)
                },
                'trend_phase': {
                    'phase': trend_phase.value,
                    'strength': float(phase_strength)
                },
                'trend_continuation': {
                    'probability': continuation.continuation_probability,
                    'strength': continuation.continuation_strength,
                    'expected_duration': continuation.expected_duration,
                    'momentum_score': continuation.momentum_score,
                    'volume_confirmation': continuation.volume_confirmation
                },
                'key_levels': continuation.key_levels,
                'trading_signals': signals,
                'technical_summary': self._generate_technical_summary(
                    overlap_data, momentum_data, current_price
                )
            }
            
            logger.info(f"✅ 趋势分析完成: {trend_direction.value} - {trend_phase.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ 综合趋势分析失败: {e}")
            return {}
    
    def _generate_trading_signals(self, trend_direction: TrendDirection, trend_phase: TrendPhase,
                                continuation: TrendContinuation, overlap_data: Dict, 
                                momentum_data: Dict) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []
        
        try:
            # 基于趋势方向和阶段的信号
            if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.WEAK_UPTREND]:
                if trend_phase == TrendPhase.BEGINNING:
                    signals.append({
                        'type': '趋势跟随',
                        'direction': 'LONG',
                        'strength': 'HIGH',
                        'reason': '上升趋势开始阶段'
                    })
                elif trend_phase == TrendPhase.ACCELERATION and continuation.continuation_probability > 0.6:
                    signals.append({
                        'type': '趋势加速',
                        'direction': 'LONG',
                        'strength': 'MEDIUM',
                        'reason': '上升趋势加速，延续概率高'
                    })
            
            elif trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.WEAK_DOWNTREND]:
                if trend_phase == TrendPhase.BEGINNING:
                    signals.append({
                        'type': '趋势跟随',
                        'direction': 'SHORT', 
                        'strength': 'HIGH',
                        'reason': '下降趋势开始阶段'
                    })
                elif trend_phase == TrendPhase.ACCELERATION and continuation.continuation_probability > 0.6:
                    signals.append({
                        'type': '趋势加速',
                        'direction': 'SHORT',
                        'strength': 'MEDIUM',
                        'reason': '下降趋势加速，延续概率高'
                    })
            
            # 基于技术指标的信号
            self._add_momentum_signals(signals, momentum_data)
            self._add_overlap_signals(signals, overlap_data)
            
        except Exception as e:
            logger.error(f"交易信号生成失败: {e}")
        
        return signals
    
    def _add_momentum_signals(self, signals: List, momentum_data: Dict):
        """添加动量指标信号"""
        try:
            # RSI信号
            rsi = momentum_data.get('rsi', [np.nan])[-1]
            if not np.isnan(rsi):
                if rsi < 30:
                    signals.append({
                        'type': 'RSI超卖',
                        'direction': 'LONG',
                        'strength': 'MEDIUM',
                        'reason': f'RSI={rsi:.1f}，超卖区域'
                    })
                elif rsi > 70:
                    signals.append({
                        'type': 'RSI超买',
                        'direction': 'SHORT',
                        'strength': 'MEDIUM',
                        'reason': f'RSI={rsi:.1f}，超买区域'
                    })
            
            # MACD信号
            macd_line = momentum_data.get('macd_line', [np.nan])[-1]
            macd_signal = momentum_data.get('macd_signal', [np.nan])[-1]
            if not np.isnan(macd_line) and not np.isnan(macd_signal):
                if len(momentum_data.get('macd_line', [])) > 1:
                    prev_macd = momentum_data['macd_line'][-2]
                    prev_signal = momentum_data['macd_signal'][-2]
                    
                    # 金叉
                    if prev_macd <= prev_signal and macd_line > macd_signal:
                        signals.append({
                            'type': 'MACD金叉',
                            'direction': 'LONG',
                            'strength': 'HIGH',
                            'reason': 'MACD线上穿信号线'
                        })
                    # 死叉
                    elif prev_macd >= prev_signal and macd_line < macd_signal:
                        signals.append({
                            'type': 'MACD死叉',
                            'direction': 'SHORT',
                            'strength': 'HIGH',
                            'reason': 'MACD线下穿信号线'
                        })
                        
        except Exception as e:
            logger.error(f"动量信号添加失败: {e}")
    
    def _add_overlap_signals(self, signals: List, overlap_data: Dict):
        """添加重叠研究信号"""
        try:
            # SAR信号
            if 'sar' in overlap_data and len(overlap_data['sar']) > 1:
                current_sar = overlap_data['sar'][-1]
                prev_sar = overlap_data['sar'][-2]
                
                # 这里需要价格数据来判断SAR信号，简化处理
                pass
                
        except Exception as e:
            logger.error(f"重叠研究信号添加失败: {e}")
    
    def _generate_technical_summary(self, overlap_data: Dict, momentum_data: Dict, 
                                  current_price: float) -> Dict[str, Any]:
        """生成技术分析摘要"""
        try:
            summary = {
                'ema_alignment': self._get_ema_alignment_text(overlap_data, current_price),
                'momentum_status': self._get_momentum_status_text(momentum_data),
                'key_observations': []
            }
            
            # 关键观察点
            rsi = momentum_data.get('rsi', [np.nan])[-1]
            if not np.isnan(rsi):
                if rsi > 70:
                    summary['key_observations'].append(f"RSI={rsi:.1f} 显示超买状态")
                elif rsi < 30:
                    summary['key_observations'].append(f"RSI={rsi:.1f} 显示超卖状态")
            
            adx = momentum_data.get('adx', [np.nan])[-1]
            if not np.isnan(adx):
                if adx > 40:
                    summary['key_observations'].append(f"ADX={adx:.1f} 显示强趋势")
                elif adx < 20:
                    summary['key_observations'].append(f"ADX={adx:.1f} 显示震荡行情")
            
            return summary
            
        except Exception as e:
            logger.error(f"技术摘要生成失败: {e}")
            return {}
    
    def _get_ema_alignment_text(self, overlap_data: Dict, current_price: float) -> str:
        """获取EMA排列描述"""
        try:
            score = self._analyze_ema_alignment(overlap_data, current_price)
            if score > 0.8:
                return "强势多头排列"
            elif score > 0.4:
                return "偏多排列"
            elif score < -0.8:
                return "强势空头排列"
            elif score < -0.4:
                return "偏空排列"
            else:
                return "排列混乱"
        except:
            return "无法确定"
    
    def _get_momentum_status_text(self, momentum_data: Dict) -> str:
        """获取动量状态描述"""
        try:
            score = self._analyze_momentum_trend(momentum_data)
            if score > 0.5:
                return "动量强劲向上"
            elif score > 0.2:
                return "动量偏向上"
            elif score < -0.5:
                return "动量强劲向下"
            elif score < -0.2:
                return "动量偏向下"
            else:
                return "动量中性"
        except:
            return "无法确定"
