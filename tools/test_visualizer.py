"""
测试形态可视化工具

此脚本用于测试pattern_visualizer.py中的功能，使用OKX交易所数据
"""

import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入测试函数
from tools.pattern_visualizer import test_pattern_visualizer

if __name__ == "__main__":
    # 运行测试函数
    asyncio.run(test_pattern_visualizer())
