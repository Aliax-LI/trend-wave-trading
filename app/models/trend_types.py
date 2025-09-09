from enum import Enum
from dataclasses import dataclass
from typing import Dict, List


class TrendDirection(Enum):
    """趋势方向枚举"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class TrendPhase(Enum):
    """趋势阶段枚举"""
    BEGINNING = "beginning"  # 趋势开始
    ACCELERATION = "acceleration"  # 趋势加速
    MATURITY = "maturity"  # 趋势成熟
    EXHAUSTION = "exhaustion"  # 趋势衰竭


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    direction: TrendDirection
    phase: TrendPhase
    strength: float  # 趋势强度 (0-1)
    confidence: float  # 信心度 (0-1)
    ema_alignment: str  # EMA排列状态
    convergence_status: str  # 收敛/发散状态
    macd_signal: str  # MACD信号
    bollinger_position: str  # 布林带位置
    support_resistance: Dict  # 支撑阻力位
    entry_signals: List[str]  # 入场信号
