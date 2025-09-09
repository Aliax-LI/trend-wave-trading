#!/usr/bin/env python3
"""
趋势监控机器人启动脚本
快速启动趋势监控，实盘观测效果
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.bot.trend_monitor_bot import run_trend_monitor
from loguru import logger


def main():
    """主函数"""
    print("""
🤖 趋势监控机器人
==================
实时监控市场趋势变化，每60秒分析一次
分析报告将保存到 docs/ 文件夹

按 Ctrl+C 停止监控
""")
    
    # 配置参数
    symbol = "BTC/USDT:USDT"  # 可以修改为其他交易对
    exchange = "okx"     # 可以修改为其他交易所
    
    logger.info(f"🚀 准备启动趋势监控: {symbol} @ {exchange}")
    
    try:
        # 运行监控机器人
        asyncio.run(run_trend_monitor(symbol, exchange))
    except KeyboardInterrupt:
        logger.info("👋 监控已停止")
    except Exception as e:
        logger.error(f"❌ 启动错误: {e}")


if __name__ == "__main__":
    main()
