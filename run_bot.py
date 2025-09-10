#!/usr/bin/env python3
"""
数字货币日内交易策略数据系统启动脚本
"""

import asyncio
import signal
import sys
from datetime import datetime
from loguru import logger

from app.core.config_loader import load_strategy_config
from app.core.crypto_data_system import CryptoTradingDataSystem, StrategyDataService


class StrategyDataSystemRunner:
    """策略数据系统运行器"""
    
    def __init__(self, config_file: str = None):
        self.config = load_strategy_config(config_file)
        self.data_system = None
        self.strategy_service = None
        self.running = False
        
        # 配置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get("logging", {})
        
        # 移除默认handler
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stderr,
            level=log_config.get("level", "INFO"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>"
        )
        
        # 文件输出
        logger.add(
            "logs/strategy_data_system.log",
            rotation=log_config.get("file_rotation", "1 day"),
            retention=log_config.get("file_retention", "7 days"),
            level=log_config.get("level", "INFO"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
        
        if log_config.get("enable_debug_logs", False):
            logger.add(
                "logs/strategy_data_system_debug.log",
                rotation="1 day",
                retention="3 days",
                level="DEBUG"
            )
    
    async def initialize_system(self) -> bool:
        """初始化系统"""
        try:
            logger.info("🚀 启动系统")
            logger.info(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 获取配置
            exchange_config = self.config.get("exchange", {})
            symbols = self.config.get("symbols", [])
            
            logger.info(f"📊 交易所: {exchange_config.get('name', 'okx')}")
            logger.info(f"📈 监控币种: {', '.join(symbols)}")
            
            # 创建数据系统
            self.data_system = CryptoTradingDataSystem(
                exchange_name=exchange_config.get("name", "okx"),
                symbols=symbols,
                config={
                    "exchange_config": exchange_config.get("config", {}),
                    "data_loader_config": self.config.get("data_loader", {})
                }
            )
            
            # 初始化系统
            success = await self.data_system.initialize()
            if not success:
                logger.error("❌ 系统初始化失败")
                return False
            
            # 创建策略数据服务
            self.strategy_service = StrategyDataService(self.data_system)
            
            # 注册信号处理
            self.setup_signal_handlers()
            
            logger.info("✅ 系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统初始化异常: {e}")
            return False
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"📨 接收到信号 {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    async def run_monitoring_loop(self):
        """运行监控循环"""
        self.running = True
        logger.info("🔄 开始监控循环...")
        
        monitoring_config = self.config.get("monitoring", {})
        status_interval = monitoring_config.get("system_health", {}).get("status_check_interval", 30)
        
        status_counter = 0
        
        try:
            while self.running:
                # 每30秒显示一次状态
                status_counter += 1
                if status_counter >= (status_interval // 10):  # 假设每10秒一个循环
                    await self.show_system_status()
                    status_counter = 0
                
                # 检查系统健康
                if not self.data_system.is_running:
                    logger.warning("⚠️ 数据系统停止运行")
                    break
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
        except Exception as e:
            logger.error(f"❌ 监控循环异常: {e}")
        finally:
            logger.info("🛑 监控循环结束")
    
    async def show_system_status(self):
        """显示系统状态"""
        try:
            logger.info("📊 系统状态报告:")
            
            # 获取系统状态
            system_status = self.data_system.get_system_status()
            logger.info(f"  🔧 系统: 初始化={system_status['initialized']}, "
                       f"运行中={system_status['running']}")
            
            # 获取所有币种摘要
            summary = await self.strategy_service.get_multi_symbol_summary()
            logger.info("  📈 币种状态:")
            
            for symbol, info in summary.items():
                status_flags = []
                # if info['trend_ready']:
                #     status_flags.append("趋势✓")
                # if info['pattern_ready']:
                #     status_flags.append("形态✓")
                # if info['entry_ready']:
                #     status_flags.append("入场✓")
                
                # 获取当前趋势分析
                current_trend = self.strategy_service.get_current_trend_analysis(symbol)
                trend_info = ""
                if current_trend and current_trend['available']:
                    direction_en = current_trend['trend_direction'].get('direction', 'unknown')
                    phase_en = current_trend['trend_phase'].get('phase', 'unknown')
                    prob = current_trend['trend_continuation'].get('probability', 0.0)
                    
                    # 转换为中文
                    direction_cn = self._get_direction_chinese(direction_en)
                    phase_cn = self._get_phase_chinese(phase_en)
                    trend_info = f", 趋势={direction_cn}-{phase_cn}({prob:.0%})"
                
                logger.info(f"    {symbol}: {' '.join(status_flags)}, "
                           f"价格={info['current_price']:.2f}, "
                           f"费率={info['funding_rate']:.4%}, "
                           f"价格位={info['approaching_levels']}个{trend_info}")
            
            # 资金费率预警
            high_funding_symbols = []
            for symbol, rate_info in system_status['funding_rates'].items():
                rate = rate_info.get('rate', 0)
                if abs(rate) > 0.001:  # 0.1%
                    high_funding_symbols.append(f"{symbol}({rate:.4%})")
            
            if high_funding_symbols:
                logger.warning(f"💰 高资金费率预警: {', '.join(high_funding_symbols)}")
            
        except Exception as e:
            logger.error(f"❌ 状态报告生成失败: {e}")
    
    def _get_direction_chinese(self, direction_en: str) -> str:
        """获取趋势方向的中文名称"""
        chinese_names = {
            "strong_uptrend": "强势上升",
            "weak_uptrend": "弱势上升", 
            "sideways": "横盘震荡",
            "weak_downtrend": "弱势下降",
            "strong_downtrend": "强势下降"
        }
        return chinese_names.get(direction_en, direction_en)
    
    def _get_phase_chinese(self, phase_en: str) -> str:
        """获取趋势阶段的中文名称"""
        chinese_names = {
            "beginning": "起始期",
            "acceleration": "加速期",
            "maturity": "成熟期",
            "exhaustion": "衰竭期"
        }
        return chinese_names.get(phase_en, phase_en)
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 开始清理系统资源...")
        
        try:
            if self.data_system:
                await self.data_system.stop()
            
            logger.info("✅ 系统资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理异常: {e}")
    
    async def run(self):
        """运行系统"""
        try:
            # 初始化系统
            if not await self.initialize_system():
                return
            
            # 启动监控任务
            monitoring_task = asyncio.create_task(self.run_monitoring_loop())
            system_monitoring_task = asyncio.create_task(self.data_system.start_monitoring())
            
            # 等待任务完成
            await asyncio.gather(monitoring_task, system_monitoring_task)
            
        except KeyboardInterrupt:
            logger.info("⌨️ 用户中断")
        except Exception as e:
            logger.error(f"❌ 系统运行异常: {e}")
        finally:
            await self.cleanup()


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数字货币日内交易策略数据系统")
    parser.add_argument("--config", "-c", type=str, 
                       help="配置文件路径", 
                       default="config/strategy_data_config.yaml")
    parser.add_argument("--test", "-t", action="store_true",
                       help="运行测试模式")
    
    args = parser.parse_args()
    
    if args.test:
        # 测试模式
        logger.info("🧪 运行测试模式")
        from tests.test_strategy_data_system import test_strategy_data_system
        await test_strategy_data_system()
    else:
        # 正常运行模式
        runner = StrategyDataSystemRunner(args.config)
        await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
