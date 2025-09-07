"""
形态可视化工具

提供交易形态的可视化功能，包括：
1. 牛旗/熊旗形态
2. 两段式整理形态
3. 三推楔形形态
4. 窄幅交易区间形态

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tzlocal
import mplfinance as mpf
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import ccxt.pro as ccxtpro
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入形态识别模块
from app.analysis.pattern.bull_flag import detect_bull_flag
from app.analysis.pattern.bear_flag import detect_bear_flag
from app.analysis.pattern.consolidation import detect_two_stage_consolidation
from app.analysis.pattern.wedge import detect_wedge
from app.analysis.pattern.narrow_range import detect_narrow_range_after_retracement
from app.analysis.pattern.double_bottom import detect_double_bottom
from app.analysis.pattern.double_bottom_bull_flag import detect_double_bottom_bull_flag

LOCAL_TZ = tzlocal.get_localzone()


def format_df(ohlcv_data) -> pd.DataFrame:
    """
    格式化OHLCV数据为标准DataFrame格式

    参数:
        ohlcv_data: 原始OHLCV数据列表

    返回:
        pd.DataFrame: 格式化后的DataFrame，包含datetime索引和数值列

    功能说明:
        - 将时间戳转换为本地时区的datetime索引
        - 确保所有价格和成交量数据为数值类型
        - 删除包含空值的行
    """
    if not ohlcv_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    # 转换时间戳为本地时区的datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(LOCAL_TZ)
    df.set_index('datetime', inplace=True)

    # 确保数值列为正确的数据类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 删除包含空值的行
    df.dropna(subset=numeric_columns, inplace=True)

    return df


exchange = getattr(ccxtpro, 'okx')(
            {
                "enableRateLimit": True,
                "newUpdates": True,
                "options": {
                    "defaultType": "swap",
                    'OHLCVLimit': 10000,
                },
                "httpProxy": "http://127.0.0.1:7890",
                'wsProxy': 'http://127.0.0.1:7890',
            }
        )


async def fetch_ohlcv(symbol, timeframe, since=None, limit=100, params={}):
    """
    从交易所获取OHLCV数据
    
    参数:
        symbol: 交易对符号
        timeframe: 时间周期
        limit: 获取的K线数量
        
    返回:
        pd.DataFrame: 格式化后的OHLCV数据
    """
    if params is None:
        params = {}
    ohlcv_data = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params=params)
    return format_df(ohlcv_data)


class PatternVisualizer:
    """交易形态可视化工具"""

    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = 'default'):
        """
        初始化可视化工具

        参数:
            figsize: 图表大小
            style: 图表样式，可选值: 'default', 'dark', 'light', 'pro'
        """
        self.figsize = figsize
        self.set_style(style)

    def set_style(self, style: str):
        """设置图表样式"""
        if style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'bullish': '#26a69a',  # 绿色
                'bearish': '#ef5350',  # 红色
                'neutral': '#9e9e9e',  # 灰色
                'volume': '#64b5f6',  # 蓝色
                'support': '#26a69a',  # 绿色
                'resistance': '#ef5350',  # 红色
                'pattern_area': 'yellow',  # 黄色
                'text': 'white'  # 白色
            }
        elif style == 'light':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = {
                'bullish': '#4caf50',  # 绿色
                'bearish': '#f44336',  # 红色
                'neutral': '#757575',  # 灰色
                'volume': '#2196f3',  # 蓝色
                'support': '#4caf50',  # 绿色
                'resistance': '#f44336',  # 红色
                'pattern_area': '#ffeb3b',  # 黄色
                'text': 'black'  # 黑色
            }
        elif style == 'pro':
            plt.style.use('ggplot')
            self.colors = {
                'bullish': '#00796b',  # 深绿色
                'bearish': '#d32f2f',  # 深红色
                'neutral': '#616161',  # 深灰色
                'volume': '#1976d2',  # 深蓝色
                'support': '#00796b',  # 深绿色
                'resistance': '#d32f2f',  # 深红色
                'pattern_area': '#fbc02d',  # 深黄色
                'text': '#212121'  # 深黑色
            }
        else:  # default
            plt.style.use('default')
            self.colors = {
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'gray',
                'volume': 'blue',
                'support': 'green',
                'resistance': 'red',
                'pattern_area': 'yellow',
                'text': 'black'
            }

    def visualize_pattern(self, df: pd.DataFrame, pattern: Dict, pattern_type: str,
                          show_volume: bool = True, show_pattern_info: bool = True,
                          context_bars: int = 20, save_path: Optional[str] = None):
        """
        可视化交易形态

        参数:
            df: 包含OHLCV数据的DataFrame
            pattern: 形态字典
            pattern_type: 形态类型，可选值: 'bull_flag', 'bear_flag', 'consolidation', 'wedge', 'narrow_range'
            show_volume: 是否显示成交量
            show_pattern_info: 是否显示形态信息
            context_bars: 形态前后显示的K线数量
            save_path: 保存图表的路径，如果为None则显示图表
        """
        # 确定绘图范围
        if 'start' in pattern and 'end' in pattern:
            start_idx = df.index.get_loc(pattern['start'])
            end_idx = df.index.get_loc(pattern['end'])
        elif 'pole_start' in pattern and 'flag_end' in pattern:
            start_idx = df.index.get_loc(pattern['pole_start'])
            end_idx = df.index.get_loc(pattern['flag_end'])
        else:
            # 如果没有明确的开始和结束，使用整个数据范围
            start_idx = 0
            end_idx = len(df) - 1

        # 向前和向后扩展一些数据点
        start_idx = max(0, start_idx - context_bars)
        end_idx = min(len(df) - 1, end_idx + context_bars)

        plot_df = df.iloc[start_idx:end_idx + 1].copy()

        # 准备mplfinance数据
        plot_df.index.name = 'Date'

        # 创建附加绘图
        apds = []

        # 根据不同形态类型添加特定标记
        if pattern_type == 'bull_flag':
            self._add_bull_flag_markers(plot_df, pattern, apds)
        elif pattern_type == 'bear_flag':
            self._add_bear_flag_markers(plot_df, pattern, apds)
        elif pattern_type == 'consolidation':
            self._add_consolidation_markers(plot_df, pattern, apds)
        elif pattern_type == 'wedge':
            self._add_wedge_markers(plot_df, pattern, apds)
        elif pattern_type == 'narrow_range':
            self._add_narrow_range_markers(plot_df, pattern, apds)

        # 设置图表样式
        mc = mpf.make_marketcolors(
            up=self.colors['bullish'],
            down=self.colors['bearish'],
            edge='inherit',
            wick='inherit',
            volume=self.colors['volume']
        )

        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=':',
            y_on_right=False
        )

        # 设置图表标题
        title = f"{pattern_type.replace('_', ' ').title()} Pattern"
        if 'strength' in pattern:
            title += f" (Strength: {pattern['strength']:.2f})"

        # 绘制图表
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            style=s,
            volume=show_volume,
            figsize=self.figsize,
            title=title,
            addplot=apds,
            returnfig=True
        )

        # 添加形态信息
        if show_pattern_info:
            self._add_pattern_info(fig, pattern, pattern_type, plot_df)

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def _add_bull_flag_markers(self, plot_df: pd.DataFrame, pattern: Dict, apds: List):
        """添加牛旗形态标记"""
        # 获取形态关键点的索引
        pole_start_idx = plot_df.index.get_loc(pattern['pole_start']) if pattern['pole_start'] in plot_df.index else 0
        pole_end_idx = plot_df.index.get_loc(pattern['pole_end']) if pattern['pole_end'] in plot_df.index else 0
        flag_end_idx = plot_df.index.get_loc(pattern['flag_end']) if pattern['flag_end'] in plot_df.index else 0

        # 创建旗杆区域
        pole_values = [np.nan] * len(plot_df)
        for i in range(pole_start_idx, pole_end_idx + 1):
            pole_values[i] = plot_df['high'].max() * 1.02
        
        # 创建旗面区域
        flag_values = [np.nan] * len(plot_df)
        for i in range(pole_end_idx, flag_end_idx + 1):
            flag_values[i] = plot_df['high'].max() * 1.02

        # 添加旗杆区域
        apds.append(
            mpf.make_addplot(
                pd.Series(pole_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color=self.colors['bullish'],
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加旗面区域
        apds.append(
            mpf.make_addplot(
                pd.Series(flag_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color=self.colors['pattern_area'],
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加目标价格和止损位
        target_line = [None] * len(plot_df)
        stop_line = [None] * len(plot_df)

        for i in range(flag_end_idx, len(plot_df)):
            target_line[i] = pattern['target_price']
            stop_line[i] = pattern['stop_loss']

        apds.append(
            mpf.make_addplot(
                pd.Series(target_line, index=plot_df.index),
                type='line',
                color=self.colors['bullish'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

        apds.append(
            mpf.make_addplot(
                pd.Series(stop_line, index=plot_df.index),
                type='line',
                color=self.colors['bearish'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

    def _add_bear_flag_markers(self, plot_df: pd.DataFrame, pattern: Dict, apds: List):
        """添加熊旗形态标记"""
        # 获取形态关键点的索引
        pole_start_idx = plot_df.index.get_loc(pattern['pole_start']) if pattern['pole_start'] in plot_df.index else 0
        pole_end_idx = plot_df.index.get_loc(pattern['pole_end']) if pattern['pole_end'] in plot_df.index else 0
        flag_end_idx = plot_df.index.get_loc(pattern['flag_end']) if pattern['flag_end'] in plot_df.index else 0

        # 创建旗杆区域
        pole_values = [np.nan] * len(plot_df)
        for i in range(pole_start_idx, pole_end_idx + 1):
            pole_values[i] = plot_df['high'].max() * 1.02
        
        # 创建旗面区域
        flag_values = [np.nan] * len(plot_df)
        for i in range(pole_end_idx, flag_end_idx + 1):
            flag_values[i] = plot_df['high'].max() * 1.02

        # 添加旗杆区域
        apds.append(
            mpf.make_addplot(
                pd.Series(pole_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color=self.colors['bearish'],
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加旗面区域
        apds.append(
            mpf.make_addplot(
                pd.Series(flag_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color=self.colors['pattern_area'],
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加目标价格和止损位
        target_line = [None] * len(plot_df)
        stop_line = [None] * len(plot_df)

        for i in range(flag_end_idx, len(plot_df)):
            target_line[i] = pattern['target_price']
            stop_line[i] = pattern['stop_loss']

        apds.append(
            mpf.make_addplot(
                pd.Series(target_line, index=plot_df.index),
                type='line',
                color=self.colors['bearish'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

        apds.append(
            mpf.make_addplot(
                pd.Series(stop_line, index=plot_df.index),
                type='line',
                color=self.colors['bullish'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

    def _add_consolidation_markers(self, plot_df: pd.DataFrame, pattern: Dict, apds: List):
        """添加两段式整理形态标记"""
        # 获取形态关键点的索引
        start_idx = plot_df.index.get_loc(pattern['start']) if pattern['start'] in plot_df.index else 0
        split_idx = plot_df.index.get_loc(pattern['split_point']) if pattern['split_point'] in plot_df.index else 0
        end_idx = plot_df.index.get_loc(pattern['end']) if pattern['end'] in plot_df.index else 0

        # 创建两段整理区域
        stage1_values = [np.nan] * len(plot_df)
        stage2_values = [np.nan] * len(plot_df)

        for i in range(start_idx, split_idx + 1):
            stage1_values[i] = plot_df['high'].max() * 1.02

        for i in range(split_idx, end_idx + 1):
            stage2_values[i] = plot_df['high'].max() * 1.02

        # 添加第一段整理区域
        apds.append(
            mpf.make_addplot(
                pd.Series(stage1_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color='blue',
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加第二段整理区域
        apds.append(
            mpf.make_addplot(
                pd.Series(stage2_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color='purple',
                secondary_y=False,
                alpha=0.2
            )
        )

        # 添加阻力位和支撑位
        resistance_line = [pattern['resistance']] * len(plot_df)
        support_line = [pattern['support']] * len(plot_df)

        apds.append(
            mpf.make_addplot(
                pd.Series(resistance_line, index=plot_df.index),
                type='line',
                color=self.colors['resistance'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

        apds.append(
            mpf.make_addplot(
                pd.Series(support_line, index=plot_df.index),
                type='line',
                color=self.colors['support'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

    def _add_wedge_markers(self, plot_df: pd.DataFrame, pattern: Dict, apds: List):
        """添加三推楔形形态标记"""
        # 获取形态关键点的索引
        start_idx = plot_df.index.get_loc(pattern['start']) if pattern['start'] in plot_df.index else 0
        end_idx = plot_df.index.get_loc(pattern['end']) if pattern['end'] in plot_df.index else 0

        # 提取高点和低点
        highs = pattern['highs']
        lows = pattern['lows']

        # 创建高点和低点的趋势线
        high_x = []
        high_y = []
        low_x = []
        low_y = []

        for date, price in highs:
            if date in plot_df.index:
                high_x.append(plot_df.index.get_loc(date))
                high_y.append(price)

        for date, price in lows:
            if date in plot_df.index:
                low_x.append(plot_df.index.get_loc(date))
                low_y.append(price)

        # 如果点数不足，无法绘制趋势线
        if len(high_x) < 2 or len(low_x) < 2:
            return

        # 计算高点趋势线
        high_slope = pattern['high_slope']
        high_intercept = high_y[0] - high_slope * high_x[0]

        # 计算低点趋势线
        low_slope = pattern['low_slope']
        low_intercept = low_y[0] - low_slope * low_x[0]

        # 创建趋势线数据
        high_line = [None] * len(plot_df)
        low_line = [None] * len(plot_df)

        for i in range(start_idx, end_idx + 1):
            high_line[i] = high_slope * i + high_intercept
            low_line[i] = low_slope * i + low_intercept

        # 添加高点趋势线
        apds.append(
            mpf.make_addplot(
                pd.Series(high_line, index=plot_df.index),
                type='line',
                color=self.colors['resistance'],
                width=1.5,
                panel=0,
                linestyle='-'
            )
        )

        # 添加低点趋势线
        apds.append(
            mpf.make_addplot(
                pd.Series(low_line, index=plot_df.index),
                type='line',
                color=self.colors['support'],
                width=1.5,
                panel=0,
                linestyle='-'
            )
        )

        # 标记高点
        high_points_series = [np.nan] * len(plot_df)
        
        for date, price in highs:
            if date in plot_df.index:
                idx = plot_df.index.get_loc(date)
                high_points_series[idx] = price

        apds.append(
            mpf.make_addplot(
                pd.Series(high_points_series, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='^',
                color=self.colors['resistance'],
                panel=0
            )
        )

        # 标记低点
        low_points_series = [np.nan] * len(plot_df)
        
        for date, price in lows:
            if date in plot_df.index:
                idx = plot_df.index.get_loc(date)
                low_points_series[idx] = price

        apds.append(
            mpf.make_addplot(
                pd.Series(low_points_series, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='v',
                color=self.colors['support'],
                panel=0
            )
        )

    def _add_narrow_range_markers(self, plot_df: pd.DataFrame, pattern: Dict, apds: List):
        """添加窄幅交易区间标记"""
        # 获取形态关键点的索引
        start_idx = plot_df.index.get_loc(pattern['start']) if pattern['start'] in plot_df.index else 0
        end_idx = plot_df.index.get_loc(pattern['end']) if pattern['end'] in plot_df.index else 0

        # 创建窄幅区间
        narrow_values = [np.nan] * len(plot_df)
        for i in range(start_idx, end_idx + 1):
            narrow_values[i] = plot_df['high'].max() * 1.02

        # 添加窄幅区间
        apds.append(
            mpf.make_addplot(
                pd.Series(narrow_values, index=plot_df.index),
                type='scatter',
                markersize=50,
                marker='_',
                panel=0,
                color=self.colors['pattern_area'],
                secondary_y=False,
                alpha=0.3
            )
        )

        # 添加阻力位和支撑位
        resistance_line = [pattern['resistance']] * len(plot_df)
        support_line = [pattern['support']] * len(plot_df)

        apds.append(
            mpf.make_addplot(
                pd.Series(resistance_line, index=plot_df.index),
                type='line',
                color=self.colors['resistance'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

        apds.append(
            mpf.make_addplot(
                pd.Series(support_line, index=plot_df.index),
                type='line',
                color=self.colors['support'],
                width=1.5,
                panel=0,
                linestyle='--'
            )
        )

        # 添加预期突破方向
        expected_direction = pattern['expected_breakout']
        arrow_start_idx = end_idx
        arrow_start_price = (pattern['resistance'] + pattern['support']) / 2

        if expected_direction == 'up':
            arrow_end_price = pattern['resistance'] * 1.02
            arrow_color = self.colors['bullish']
        else:
            arrow_end_price = pattern['support'] * 0.98
            arrow_color = self.colors['bearish']
            
        # 添加箭头指示预期突破方向
        # 由于mplfinance不直接支持箭头，我们可以使用特殊标记来指示方向
        arrow_x = [None] * len(plot_df)
        arrow_x[arrow_start_idx] = arrow_start_price
        
        apds.append(
            mpf.make_addplot(
                pd.Series(arrow_x, index=plot_df.index),
                type='scatter',
                markersize=100,
                marker='^' if expected_direction == 'up' else 'v',
                color=arrow_color,
                panel=0
            )
        )

    def _add_pattern_info(self, fig, pattern: Dict, pattern_type: str, plot_df: pd.DataFrame):
        """添加形态信息"""
        info_text = f"Pattern Type: {pattern_type.replace('_', ' ').title()}\n"

        # 添加强度信息
        if 'strength' in pattern:
            info_text += f"Strength: {pattern['strength']:.2f}\n"

        # 添加形态特定信息
        if pattern_type == 'bull_flag' or pattern_type == 'bear_flag':
            info_text += f"Pole Height: {pattern['pole_height_pct']:.2%}\n"
            info_text += f"Flag Length: {pattern['flag_length']} bars\n"
            info_text += f"Target Price: {pattern['target_price']:.2f}\n"
            info_text += f"Stop Loss: {pattern['stop_loss']:.2f}\n"

            # 计算风险收益比
            if 'flag_end_price' in pattern:
                flag_end_price = pattern['flag_end_price']
            elif 'flag_end' in pattern and pattern['flag_end'] in plot_df.index:
                # 使用旗面结束时的收盘价
                flag_end_idx = plot_df.index.get_loc(pattern['flag_end'])
                flag_end_price = plot_df['close'].iloc[flag_end_idx]
            else:
                # 如果无法获取旗面结束价格，使用最后一个收盘价
                flag_end_price = plot_df['close'].iloc[-1]
                
            risk = abs(pattern['stop_loss'] - flag_end_price)
            reward = abs(pattern['target_price'] - flag_end_price)
            if risk > 0:
                risk_reward = reward / risk
                info_text += f"Risk/Reward: 1:{risk_reward:.2f}\n"

        elif pattern_type == 'consolidation':
            info_text += f"Pre-Trend: {pattern['pre_trend']}\n"
            info_text += f"Breakout Bias: {'Upward' if pattern['breakout_bias'] > 0.5 else 'Downward'}\n"
            info_text += f"Support: {pattern['support']:.2f}\n"
            info_text += f"Resistance: {pattern['resistance']:.2f}\n"

        elif pattern_type == 'wedge':
            info_text += f"Type: {pattern['type'].replace('_', ' ').title()}\n"
            info_text += f"Expected Breakout: {'Upward' if pattern['expected_breakout'] == 'up' else 'Downward'}\n"
            info_text += f"High Angle: {pattern['high_angle']:.2f}°\n"
            info_text += f"Low Angle: {pattern['low_angle']:.2f}°\n"

        elif pattern_type == 'narrow_range':
            info_text += f"Trend: {'Upward' if pattern['trend'] == 'up' else 'Downward'}\n"
            info_text += f"Retracement: {pattern['retracement']:.2%}\n"
            info_text += f"Range Reduction: {pattern['range_reduction']:.2%}\n"
            info_text += f"Volume Reduction: {pattern['volume_reduction']:.2%}\n"
            info_text += f"Expected Breakout: {'Upward' if pattern['expected_breakout'] == 'up' else 'Downward'}\n"
            
        elif pattern_type == 'double_bottom':
            info_text += f"Depth: {pattern['depth_pct']:.2%}\n"
            info_text += f"Bottoms Diff: {pattern['price_diff_pct']:.2%}\n"
            info_text += f"Neckline Height: {pattern['neckline_height_pct']:.2%}\n"
            info_text += f"Volume Increase: {'Yes' if pattern['volume_increase'] else 'No'}\n"
            info_text += f"Target Price: {pattern['target_price']:.2f}\n"
            info_text += f"Stop Loss: {pattern['stop_loss']:.2f}\n"
            
        elif pattern_type == 'double_bottom_bull_flag':
            info_text += f"Combined Pattern\n"
            info_text += f"Target Price: {pattern['target_price']:.2f}\n"
            info_text += f"Stop Loss: {pattern['stop_loss']:.2f}\n"
            
            # 计算风险收益比
            current_price = plot_df['close'].iloc[-1]
            risk = abs(pattern['stop_loss'] - current_price)
            reward = abs(pattern['target_price'] - current_price)
            if risk > 0:
                risk_reward = reward / risk
                info_text += f"Risk/Reward: 1:{risk_reward:.2f}\n"

        # 添加文本框
        fig.text(0.02, 0.02, info_text, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))

    def visualize_all_patterns(self, df: pd.DataFrame, save_dir: Optional[str] = None, top_n: int = 1):
        """
        检测并可视化所有交易形态

        参数:
            df: 包含OHLCV数据的DataFrame
            save_dir: 保存图表的目录，如果为None则显示图表
            top_n: 每种形态保留的最强形态数量，默认为1（只保留最强的一个）
        """
        # 存储所有形态及其强度
        all_patterns = []
        
        # # 检测牛旗形态
        # bull_flags = detect_bull_flag(df)
        # for pattern in bull_flags:
        #     all_patterns.append(('bull_flag', pattern))
        
        # # 检测熊旗形态
        # bear_flags = detect_bear_flag(df)
        # for pattern in bear_flags:
        #     all_patterns.append(('bear_flag', pattern))
        
        # 检测两段式整理形态
        consolidations = detect_two_stage_consolidation(df)
        for pattern in consolidations:
            all_patterns.append(('consolidation', pattern))
        
        # 检测三推楔形形态
        wedges = detect_wedge(df)
        for pattern in wedges:
            all_patterns.append(('wedge', pattern))
        
        # 检测窄幅交易区间形态
        narrow_ranges = detect_narrow_range_after_retracement(df)
        for pattern in narrow_ranges:
            all_patterns.append(('narrow_range', pattern))
            
        # # 检测双底形态
        # double_bottoms = detect_double_bottom(df)
        # for pattern in double_bottoms:
        #     all_patterns.append(('double_bottom', pattern))
            
        # 检测双底牛旗形态
        double_bottom_bull_flags = detect_double_bottom_bull_flag(df)
        for pattern in double_bottom_bull_flags:
            all_patterns.append(('double_bottom_bull_flag', pattern))
        
        # 按强度排序所有形态
        all_patterns.sort(key=lambda x: x[1]['strength'], reverse=True)
        
        # 只保留前top_n个最强的形态
        top_patterns = all_patterns[:top_n]
        
        if not top_patterns:
            print("未检测到任何形态")
            return
        
        # 可视化选中的形态
        for i, (pattern_type, pattern) in enumerate(top_patterns):
            pattern_name = pattern_type.replace('_', ' ').title()
            save_path = os.path.join(save_dir, f"top_{i+1}_{pattern_name}.png") if save_dir else None
            self.visualize_pattern(df, pattern, pattern_type, save_path=save_path)
            print(f"绘制形态: {pattern_name}, 强度: {pattern['strength']:.2f}")


async def test_pattern_visualizer():
    """
    使用OKX数据测试形态可视化工具
    """
    import asyncio
    
    # 初始化可视化工具
    visualizer = PatternVisualizer(style='dark')
    
    # BTC/USDT永续合约的K线数据
    symbol = 'WLFI/USDT:USDT'  # OKX的BTC永续合约
    timeframe = '15m'
    limit = 100  # 获取100根K线
    banch_count = 10 # 获取批次数
    
    try:
        # 获取数据
        print(f"正在获取 {symbol} {timeframe} 数据...")
        
        # 批次获取K线数据
        batch_df_data = []
        since = None
        total_klines = 0
        
        for i in range(banch_count):
            print(f"获取第 {i+1}/{banch_count} 批次数据...")
            df = await fetch_ohlcv(symbol, timeframe, since, limit)
            
            if df.empty:
                print(f"批次 {i+1} 无数据，停止获取")
                break
                
            batch_df_data.append(df)
            total_klines += len(df)
            
            # 更新since参数，获取更早的数据
            since = int(df.index[0].timestamp() * 1000) - 1
        
        if not batch_df_data:
            print("获取数据失败，请检查网络连接或代理设置")
            return
            
        print(f"成功获取 {total_klines} 根K线数据，共 {len(batch_df_data)} 个批次")
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), 'pattern_images')
        os.makedirs(save_dir, exist_ok=True)
        
        # 对每个批次数据进行分析
        for batch_idx, batch_df in enumerate(batch_df_data):
            print(f"正在分析第 {batch_idx+1}/{len(batch_df_data)} 批次数据...")
            
            # 创建批次子目录
            batch_dir = os.path.join(save_dir, f"batch_{batch_idx+1}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # 检测并可视化该批次的所有形态
            print(f"正在检测和可视化第 {batch_idx+1} 批次的交易形态...")
            visualizer.visualize_all_patterns(batch_df, save_dir=batch_dir)
        
        print(f"形态可视化完成，图片保存在: {save_dir}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
    finally:
        # 关闭交易所连接
        await exchange.close()


if __name__ == "__main__":
    import asyncio
    
    # 运行测试函数
    asyncio.run(test_pattern_visualizer())
