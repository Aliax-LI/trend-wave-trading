"""
交易形态识别模块
包含各种高级K线形态的识别算法

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

from .bull_flag import detect_bull_flag
from .bear_flag import detect_bear_flag
from .consolidation import detect_two_stage_consolidation
from .wedge import detect_wedge
from .narrow_range import detect_narrow_range_after_retracement

__all__ = [
    'detect_bull_flag',
    'detect_bear_flag',
    'detect_two_stage_consolidation',
    'detect_wedge',
    'detect_narrow_range_after_retracement'
]
