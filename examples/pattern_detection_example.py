"""
交易形态识别示例

本示例展示如何使用交易形态识别模块来检测各种形态
包括：双底牛旗、双顶熊旗、两段式整理、三推楔形和回调形态末期的窄幅交易区间

作者: Cursor
日期: 2023-05-10
版本: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入形态识别模块
from app.analysis.pattern.bull_flag import detect_bull_flag, is_bull_flag_breakout
from app.analysis.pattern.bear_flag import detect_bear_flag, is_bear_flag_breakout
from app.analysis.pattern.consolidation import detect_two_stage_consolidation, is_consolidation_breakout
from app.analysis.pattern.wedge import detect_wedge, is_wedge_breakout
from app.analysis.pattern.narrow_range import detect_narrow_range_after_retracement, is_narrow_range_breakout


def load_sample_data(symbol='BTC/USDT', timeframe='1d', limit=200):
    """
    加载示例数据，如果没有实际数据，则生成模拟数据
    """
    try:
        # 尝试从文件加载数据
        df = pd.read_csv(f'data/{symbol.replace("/", "_")}_{timeframe}.csv', index_col=0, parse_dates=True)
        print(f"成功加载{symbol} {timeframe}数据，共{len(df)}条记录")
        return df
    except FileNotFoundError:
        print(f"未找到{symbol}数据，生成模拟数据...")
        
        # 生成模拟数据
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=limit, freq='D')
        
        # 生成价格数据
        close = 40000 + np.cumsum(np.random.normal(0, 400, limit))
        
        # 确保价格不会变为负数
        close = np.maximum(close, 100)
        
        # 生成高低价格
        high = close + np.random.uniform(100, 500, limit)
        low = close - np.random.uniform(100, 500, limit)
        
        # 确保低价不会高于收盘价，高价不会低于收盘价
        for i in range(limit):
            if low[i] > close[i]:
                low[i] = close[i] - np.random.uniform(10, 100)
            if high[i] < close[i]:
                high[i] = close[i] + np.random.uniform(10, 100)
        
        # 生成开盘价
        open_price = close - np.random.normal(0, 200, limit)
        
        # 确保开盘价在高低价之间
        for i in range(limit):
            open_price[i] = max(low[i], min(high[i], open_price[i]))
        
        # 生成成交量
        volume = np.random.uniform(1000, 10000, limit) * (1 + np.random.normal(0, 0.3, limit))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df


def plot_pattern(df, pattern, pattern_type):
    """
    绘制形态图表
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 确定绘图范围
    if 'start' in pattern and 'end' in pattern:
        start_idx = df.index.get_loc(pattern['start'])
        end_idx = df.index.get_loc(pattern['end'])
        # 向前和向后扩展一些数据点
        start_idx = max(0, start_idx - 20)
        end_idx = min(len(df) - 1, end_idx + 20)
    else:
        # 如果没有明确的开始和结束，使用整个数据范围
        start_idx = 0
        end_idx = len(df) - 1
    
    plot_df = df.iloc[start_idx:end_idx+1]
    
    # 绘制K线图
    for i in range(len(plot_df)):
        date = plot_df.index[i]
        open_price = plot_df['open'].iloc[i]
        close = plot_df['close'].iloc[i]
        high = plot_df['high'].iloc[i]
        low = plot_df['low'].iloc[i]
        
        # 确定K线颜色
        color = 'green' if close >= open_price else 'red'
        
        # 绘制K线实体
        ax1.plot([date, date], [open_price, close], color=color, linewidth=4)
        
        # 绘制上下影线
        ax1.plot([date, date], [low, high], color=color, linewidth=1)
    
    # 根据不同形态类型添加特定标记
    if pattern_type == 'bull_flag':
        # 标记旗杆和旗面
        pole_start = pattern['pole_start']
        pole_end = pattern['pole_end']
        flag_end = pattern['flag_end']
        
        ax1.axvspan(pole_start, pole_end, alpha=0.2, color='green', label='旗杆')
        ax1.axvspan(pole_end, flag_end, alpha=0.2, color='yellow', label='旗面')
        
        # 标记目标价格和止损位
        ax1.axhline(y=pattern['target_price'], color='green', linestyle='--', label=f"目标价: {pattern['target_price']:.2f}")
        ax1.axhline(y=pattern['stop_loss'], color='red', linestyle='--', label=f"止损位: {pattern['stop_loss']:.2f}")
    
    elif pattern_type == 'bear_flag':
        # 标记旗杆和旗面
        pole_start = pattern['pole_start']
        pole_end = pattern['pole_end']
        flag_end = pattern['flag_end']
        
        ax1.axvspan(pole_start, pole_end, alpha=0.2, color='red', label='旗杆')
        ax1.axvspan(pole_end, flag_end, alpha=0.2, color='yellow', label='旗面')
        
        # 标记目标价格和止损位
        ax1.axhline(y=pattern['target_price'], color='red', linestyle='--', label=f"目标价: {pattern['target_price']:.2f}")
        ax1.axhline(y=pattern['stop_loss'], color='green', linestyle='--', label=f"止损位: {pattern['stop_loss']:.2f}")
    
    elif pattern_type == 'two_stage_consolidation':
        # 标记两段整理区间
        start = pattern['start']
        split_point = pattern['split_point']
        end = pattern['end']
        
        ax1.axvspan(start, split_point, alpha=0.2, color='blue', label='第一段整理')
        ax1.axvspan(split_point, end, alpha=0.2, color='purple', label='第二段整理')
        
        # 标记阻力位和支撑位
        ax1.axhline(y=pattern['resistance'], color='red', linestyle='--', label=f"阻力位: {pattern['resistance']:.2f}")
        ax1.axhline(y=pattern['support'], color='green', linestyle='--', label=f"支撑位: {pattern['support']:.2f}")
    
    elif pattern_type == 'wedge':
        # 标记楔形区间
        start = pattern['start']
        end = pattern['end']
        
        # 绘制楔形趋势线
        dates = mdates.date2num(plot_df.index)
        start_idx_rel = np.where(dates >= mdates.date2num(start))[0][0]
        end_idx_rel = np.where(dates >= mdates.date2num(end))[0][0]
        
        x_vals = np.arange(start_idx_rel, end_idx_rel + 1)
        
        # 高点趋势线
        high_y_vals = pattern['high_slope'] * x_vals + (pattern['highs'][0][1] - pattern['high_slope'] * start_idx_rel)
        
        # 低点趋势线
        low_y_vals = pattern['low_slope'] * x_vals + (pattern['lows'][0][1] - pattern['low_slope'] * start_idx_rel)
        
        ax1.plot(plot_df.index[start_idx_rel:end_idx_rel+1], high_y_vals, 'r--', linewidth=2, label='高点趋势线')
        ax1.plot(plot_df.index[start_idx_rel:end_idx_rel+1], low_y_vals, 'g--', linewidth=2, label='低点趋势线')
        
        # 标记楔形类型
        wedge_type = "上升楔形" if pattern['type'] == 'rising_wedge' else "下降楔形"
        expected_breakout = "向下" if pattern['expected_breakout'] == 'down' else "向上"
        ax1.set_title(f"{wedge_type} (预期{expected_breakout}突破)")
    
    elif pattern_type == 'narrow_range':
        # 标记窄幅区间
        start = pattern['start']
        end = pattern['end']
        
        ax1.axvspan(start, end, alpha=0.3, color='yellow', label='窄幅区间')
        
        # 标记阻力位和支撑位
        ax1.axhline(y=pattern['resistance'], color='red', linestyle='--', label=f"阻力位: {pattern['resistance']:.2f}")
        ax1.axhline(y=pattern['support'], color='green', linestyle='--', label=f"支撑位: {pattern['support']:.2f}")
        
        # 标记预期突破方向
        expected_direction = "向上" if pattern['expected_breakout'] == 'up' else "向下"
        ax1.set_title(f"回调末期窄幅区间 (预期{expected_direction}突破)")
    
    # 绘制成交量
    ax2.bar(plot_df.index, plot_df['volume'], color='blue', alpha=0.5)
    ax2.set_ylabel('成交量')
    
    # 设置图表格式
    ax1.set_ylabel('价格')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # 格式化日期轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数
    """
    print("加载数据...")
    df = load_sample_data()
    
    print("\n1. 检测牛旗形态...")
    bull_flags = detect_bull_flag(df)
    if bull_flags:
        print(f"检测到{len(bull_flags)}个牛旗形态")
        strongest_bull_flag = bull_flags[0]
        print(f"最强牛旗形态 (强度: {strongest_bull_flag['strength']:.2f}):")
        print(f"  - 起始日期: {strongest_bull_flag['pole_start'].strftime('%Y-%m-%d')}")
        print(f"  - 结束日期: {strongest_bull_flag['flag_end'].strftime('%Y-%m-%d')}")
        print(f"  - 旗杆高度: {strongest_bull_flag['pole_height_pct']:.2%}")
        print(f"  - 目标价格: {strongest_bull_flag['target_price']:.2f}")
        print(f"  - 止损位置: {strongest_bull_flag['stop_loss']:.2f}")
        
        # 绘制图表
        plot_pattern(df, strongest_bull_flag, 'bull_flag')
    else:
        print("未检测到牛旗形态")
    
    print("\n2. 检测熊旗形态...")
    bear_flags = detect_bear_flag(df)
    if bear_flags:
        print(f"检测到{len(bear_flags)}个熊旗形态")
        strongest_bear_flag = bear_flags[0]
        print(f"最强熊旗形态 (强度: {strongest_bear_flag['strength']:.2f}):")
        print(f"  - 起始日期: {strongest_bear_flag['pole_start'].strftime('%Y-%m-%d')}")
        print(f"  - 结束日期: {strongest_bear_flag['flag_end'].strftime('%Y-%m-%d')}")
        print(f"  - 旗杆高度: {strongest_bear_flag['pole_height_pct']:.2%}")
        print(f"  - 目标价格: {strongest_bear_flag['target_price']:.2f}")
        print(f"  - 止损位置: {strongest_bear_flag['stop_loss']:.2f}")
        
        # 绘制图表
        plot_pattern(df, strongest_bear_flag, 'bear_flag')
    else:
        print("未检测到熊旗形态")
    
    print("\n3. 检测两段式整理形态...")
    consolidations = detect_two_stage_consolidation(df)
    if consolidations:
        print(f"检测到{len(consolidations)}个两段式整理形态")
        strongest_consolidation = consolidations[0]
        print(f"最强两段式整理形态 (强度: {strongest_consolidation['strength']:.2f}):")
        print(f"  - 起始日期: {strongest_consolidation['start'].strftime('%Y-%m-%d')}")
        print(f"  - 分割点: {strongest_consolidation['split_point'].strftime('%Y-%m-%d')}")
        print(f"  - 结束日期: {strongest_consolidation['end'].strftime('%Y-%m-%d')}")
        print(f"  - 前期趋势: {strongest_consolidation['pre_trend']}")
        print(f"  - 突破偏向: {'向上' if strongest_consolidation['breakout_bias'] > 0.5 else '向下'}")
        
        # 绘制图表
        plot_pattern(df, strongest_consolidation, 'two_stage_consolidation')
    else:
        print("未检测到两段式整理形态")
    
    print("\n4. 检测三推楔形形态...")
    wedges = detect_wedge(df)
    if wedges:
        print(f"检测到{len(wedges)}个楔形形态")
        strongest_wedge = wedges[0]
        wedge_type = "上升楔形" if strongest_wedge['type'] == 'rising_wedge' else "下降楔形"
        print(f"最强{wedge_type}形态 (强度: {strongest_wedge['strength']:.2f}):")
        print(f"  - 起始日期: {strongest_wedge['start'].strftime('%Y-%m-%d')}")
        print(f"  - 结束日期: {strongest_wedge['end'].strftime('%Y-%m-%d')}")
        print(f"  - 高点数量: {len(strongest_wedge['highs'])}")
        print(f"  - 低点数量: {len(strongest_wedge['lows'])}")
        print(f"  - 预期突破: {'向上' if strongest_wedge['expected_breakout'] == 'up' else '向下'}")
        
        # 绘制图表
        plot_pattern(df, strongest_wedge, 'wedge')
    else:
        print("未检测到楔形形态")
    
    print("\n5. 检测回调形态末期的窄幅交易区间...")
    narrow_ranges = detect_narrow_range_after_retracement(df)
    if narrow_ranges:
        print(f"检测到{len(narrow_ranges)}个窄幅交易区间")
        strongest_nr = narrow_ranges[0]
        print(f"最强窄幅交易区间 (强度: {strongest_nr['strength']:.2f}):")
        print(f"  - 起始日期: {strongest_nr['start'].strftime('%Y-%m-%d')}")
        print(f"  - 结束日期: {strongest_nr['end'].strftime('%Y-%m-%d')}")
        print(f"  - 主趋势: {'上升' if strongest_nr['trend'] == 'up' else '下降'}")
        print(f"  - 回调幅度: {strongest_nr['retracement']:.2%}")
        print(f"  - 波动减少: {strongest_nr['range_reduction']:.2%}")
        print(f"  - 成交量减少: {strongest_nr['volume_reduction']:.2%}")
        
        # 绘制图表
        plot_pattern(df, strongest_nr, 'narrow_range')
    else:
        print("未检测到窄幅交易区间")


if __name__ == "__main__":
    main()
