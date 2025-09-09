"""
趋势监控机器人
定期执行趋势分析，实时监控市场趋势变化
用于实盘数据观测和策略验证
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from pathlib import Path

from loguru import logger

from app.core.data_loader import OhlcvDataLoader
from app.analysis.trend.enhanced_trend_analyzer import EnhancedTrendAnalyzer, EnhancedTrendAnalysis
from app.models.trend_types import TrendDirection, TrendPhase


def convert_numpy_types(obj: Any) -> Any:
    """
    将NumPy数据类型转换为Python原生类型，用于JSON序列化
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class TrendMonitorBot:
    """趋势监控机器人"""
    
    def __init__(self, symbol: str, exchange_name: str, config: Dict):
        """
        初始化趋势监控机器人
        
        Args:
            symbol: 交易对符号
            exchange_name: 交易所名称
            config: 配置参数
        """
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.config = config
        
        # 监控参数
        self.monitor_interval = config.get("monitor_interval", 60)  # 监控间隔(秒)
        self.window_obs = config.get("window_obs", 80)
        
        # 创建分析器
        self.trend_analyzer = EnhancedTrendAnalyzer(window_obs=self.window_obs)
        self.data_loader = None
        
        # 历史记录
        self.trend_history: List[Dict] = []
        self.last_analysis: Optional[EnhancedTrendAnalysis] = None
        
        # 统计信息
        self.analysis_count = 0
        self.start_time = None
        self.bot_running = False
        
        # 结果保存路径
        self.results_dir = Path("results/trend_monitoring")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 报告保存路径 - docs文件夹
        self.docs_dir = Path("docs")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # 基于币种的报告文件名（清理特殊字符）
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        self.report_filename = f"trend_analysis_{safe_symbol}.md"
        self.report_file_path = self.docs_dir / self.report_filename
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志"""
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        log_file = self.results_dir / f"trend_monitor_{safe_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 添加文件日志处理器
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            rotation="1 day",
            compression="zip"
        )
    
    async def start_monitoring(self):
        """启动趋势监控"""
        self.bot_running = True
        self.start_time = datetime.now(timezone.utc)
        
        logger.info(f"🤖 趋势监控机器人启动")
        logger.info(f"📊 监控品种: {self.symbol}")
        logger.info(f"🏢 交易所: {self.exchange_name}")
        logger.info(f"⏰ 监控间隔: {self.monitor_interval}秒")
        logger.info(f"🔍 观测窗口: {self.window_obs}条K线")
        
        # 创建数据加载器
        self.data_loader = OhlcvDataLoader(
            exchange_name=self.exchange_name,
            symbol=self.symbol,
            config=self.config
        )
        
        try:
            # 先初始化数据
            if not await self.data_loader.init_ohlcv():
                logger.error("❌ 数据初始化失败")
                return
            
            # 启动数据监控任务（非阻塞）
            data_monitor_task = asyncio.create_task(self.data_loader.watch_ohlcv())
            
            # 等待数据稳定
            await asyncio.sleep(5)
            
            # 开始趋势监控循环
            monitor_loop_task = asyncio.create_task(self._monitoring_loop())
            
            # 等待任何一个任务完成或异常
            done, pending = await asyncio.wait(
                [data_monitor_task, monitor_loop_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消未完成的任务
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 检查是否有异常
            for task in done:
                if task.exception():
                    raise task.exception()
            
        except KeyboardInterrupt:
            logger.info("🛑 用户中断监控")
        except Exception as e:
            logger.error(f"❌ 监控过程中发生错误: {e}")
        finally:
            await self.stop_monitoring()
    
    async def _monitoring_loop(self):
        """监控主循环"""
        logger.info("🔄 开始趋势监控循环...")
        
        while self.bot_running:
            try:
                # 执行趋势分析
                analysis_result = await self._perform_trend_analysis()
                
                if analysis_result:
                    # 检查趋势变化
                    await self._check_trend_changes(analysis_result)
                    
                    # 记录分析结果
                    await self._record_analysis_result(analysis_result)
                    
                    # 更新docs报告
                    await self._update_docs_report(analysis_result)
                    
                    # 输出监控状态
                    self._log_monitoring_status(analysis_result)
                    
                    self.last_analysis = analysis_result
                    self.analysis_count += 1
                else:
                    logger.warning("⚠️ 趋势分析失败，跳过本次监控")
                
                # 等待下次监控
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"❌ 监控循环错误: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _perform_trend_analysis(self) -> Optional[EnhancedTrendAnalysis]:
        """执行趋势分析"""
        try:
            # 获取观测数据
            observed_ohlc = self.data_loader.cache_ohlcv.get("observed")
            if observed_ohlc is None or observed_ohlc.empty:
                logger.warning("⚠️ 无法获取观测数据")
                logger.debug(f"缓存状态: {list(self.data_loader.cache_ohlcv.keys())}")
                return None
            
            logger.debug(f"📊 获取到观测数据: {len(observed_ohlc)}条")
            
            if len(observed_ohlc) < 144:  # 确保有足够数据
                logger.warning(f"⚠️ 数据不足，当前{len(observed_ohlc)}条，需要至少144条")
                return None
            
            # 执行综合趋势分析
            logger.debug("🔍 开始执行趋势分析...")
            analysis = self.trend_analyzer.analyze_comprehensive_trend(observed_ohlc)
            
            if analysis:
                logger.debug("✅ 趋势分析完成")
            else:
                logger.warning("❌ 趋势分析返回None")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 趋势分析执行错误: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    async def _check_trend_changes(self, current_analysis: EnhancedTrendAnalysis):
        """检查趋势变化"""
        if self.last_analysis is None:
            return
        
        # 检查趋势方向变化
        if current_analysis.basic_trend.direction != self.last_analysis.basic_trend.direction:
            logger.info(f"🔄 趋势方向变化: {self.last_analysis.basic_trend.direction.value} → {current_analysis.basic_trend.direction.value}")
        
        # 检查趋势阶段变化
        if current_analysis.basic_trend.phase != self.last_analysis.basic_trend.phase:
            logger.info(f"📈 趋势阶段变化: {self.last_analysis.basic_trend.phase.value} → {current_analysis.basic_trend.phase.value}")
        
        # 检查信号强度显著变化
        strength_change = abs(current_analysis.overall_signal_strength - self.last_analysis.overall_signal_strength)
        if strength_change > 0.2:  # 20%以上变化
            direction = "增强" if current_analysis.overall_signal_strength > self.last_analysis.overall_signal_strength else "减弱"
            logger.info(f"💪 信号强度{direction}: {self.last_analysis.overall_signal_strength:.1%} → {current_analysis.overall_signal_strength:.1%}")
        
        # 检查入场时机变化
        if current_analysis.continuation_analysis.entry_timing != self.last_analysis.continuation_analysis.entry_timing:
            logger.info(f"⏰ 入场时机变化: {self.last_analysis.continuation_analysis.entry_timing} → {current_analysis.continuation_analysis.entry_timing}")
    
    async def _record_analysis_result(self, analysis: EnhancedTrendAnalysis):
        """记录分析结果"""
        current_time = datetime.now(timezone.utc)
        
        # 构建记录
        record = {
            "timestamp": current_time.isoformat(),
            "symbol": self.symbol,
            "analysis_count": self.analysis_count,
            
            # 基础趋势信息
            "trend_direction": analysis.basic_trend.direction.value,
            "trend_phase": analysis.basic_trend.phase.value,
            "trend_strength": analysis.basic_trend.strength,
            "trend_confidence": analysis.basic_trend.confidence,
            
            # 综合评分
            "overall_signal_strength": analysis.overall_signal_strength,
            "trend_quality_score": analysis.trend_quality_score,
            "entry_timing_score": analysis.entry_timing_score,
            
            # 技术指标
            "ema_alignment": analysis.basic_trend.ema_alignment,
            "convergence_status": analysis.basic_trend.convergence_status,
            "macd_signal": analysis.basic_trend.macd_signal,
            "bollinger_position": analysis.basic_trend.bollinger_position,
            
            # 延续性分析
            "continuation_signal": analysis.continuation_analysis.signal.value,
            "continuation_probability": analysis.continuation_analysis.continuation_probability,
            "entry_timing": analysis.continuation_analysis.entry_timing,
            "pullback_quality": analysis.continuation_analysis.pullback_quality,
            "momentum_strength": analysis.continuation_analysis.momentum_strength,
            "volume_confirmation": analysis.continuation_analysis.volume_confirmation,
            "breakout_potential": analysis.continuation_analysis.breakout_potential,
            
            # 价格行为
            "price_action_signals_count": len(analysis.price_action_signals),
            "support_levels": [float(level) for level in analysis.key_support_resistance.get('support', [])],
            "resistance_levels": [float(level) for level in analysis.key_support_resistance.get('resistance', [])],
            
            # 关键价位
            "key_support": float(analysis.basic_trend.support_resistance.get('support', 0)),
            "key_resistance": float(analysis.basic_trend.support_resistance.get('resistance', 0)),
            "key_level": float(analysis.basic_trend.support_resistance.get('key_level', 0))
        }
        
        # 转换NumPy类型并添加到历史记录
        converted_record = convert_numpy_types(record)
        self.trend_history.append(converted_record)
        
        # 定期保存到文件
        if self.analysis_count % 10 == 0:  # 每10次分析保存一次
            await self._save_results_to_file()
    
    async def _update_docs_report(self, analysis: EnhancedTrendAnalysis):
        """更新docs文件夹中的趋势分析报告"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 获取当前价格信息
            observed_ohlc = self.data_loader.cache_ohlcv.get("observed")
            current_price = observed_ohlc['close'].iloc[-1] if observed_ohlc is not None and not observed_ohlc.empty else 0
            
            # 构建Markdown报告
            report_content = self._generate_markdown_report(analysis, current_time, current_price)
            
            # 写入文件
            with open(self.report_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.debug(f"📝 报告已更新: {self.report_file_path}")
            
        except Exception as e:
            logger.error(f"❌ 更新docs报告错误: {e}")
    
    def _generate_markdown_report(self, analysis: EnhancedTrendAnalysis, 
                                current_time: datetime, current_price: float) -> str:
        """生成Markdown格式的趋势分析报告"""
        
        # 趋势方向的emoji映射
        direction_emoji = {
            "strong_uptrend": "🚀",
            "weak_uptrend": "📈", 
            "sideways": "➡️",
            "weak_downtrend": "📉",
            "strong_downtrend": "💥"
        }
        
        # 趋势阶段的emoji映射
        phase_emoji = {
            "beginning": "🌅",
            "acceleration": "⚡",
            "maturity": "🌕",
            "exhaustion": "🌅"
        }
        
        # 信号强度评级
        def get_strength_rating(strength: float) -> str:
            if strength >= 0.8:
                return "🟢 极强"
            elif strength >= 0.6:
                return "🟡 较强"
            elif strength >= 0.4:
                return "🟠 中等"
            else:
                return "🔴 较弱"
        
        # 构建报告内容
        report = f"""# {self.symbol} 趋势分析报告

> **最后更新时间**: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
> **当前价格**: {current_price:.4f}  
> **分析次数**: {self.analysis_count}  
> **监控时长**: {self._get_monitoring_duration_minutes():.1f} 分钟

---

## 📊 核心趋势信息

| 指标 | 值 | 状态 |
|------|----|----- |
| **趋势方向** | {analysis.basic_trend.direction.value} | {direction_emoji.get(analysis.basic_trend.direction.value, '❓')} |
| **趋势阶段** | {analysis.basic_trend.phase.value} | {phase_emoji.get(analysis.basic_trend.phase.value, '❓')} |
| **趋势强度** | {analysis.basic_trend.strength:.1%} | {get_strength_rating(analysis.basic_trend.strength)} |
| **趋势质量** | {analysis.trend_quality_score:.1%} | {get_strength_rating(analysis.trend_quality_score)} |

---

## 🎯 综合评分

| 评分项目 | 数值 | 等级 |
|----------|------|------|
| **综合信号强度** | {analysis.overall_signal_strength:.1%} | {get_strength_rating(analysis.overall_signal_strength)} |
| **入场时机评分** | {analysis.entry_timing_score:.1%} | {get_strength_rating(analysis.entry_timing_score)} |
| **延续概率** | {analysis.continuation_analysis.continuation_probability:.1%} | {get_strength_rating(analysis.continuation_analysis.continuation_probability)} |

---

## 📋 技术指标状态

### EMA均线系统
- **EMA排列**: {analysis.basic_trend.ema_alignment}
- **收敛状态**: {analysis.basic_trend.convergence_status}

### 动量指标
- **MACD信号**: {analysis.basic_trend.macd_signal}
- **动量强度**: {analysis.continuation_analysis.momentum_strength:.1%}

### 波动性指标
- **布林带位置**: {analysis.basic_trend.bollinger_position}
- **突破潜力**: {analysis.continuation_analysis.breakout_potential:.1%}

---

## 🔄 趋势延续分析

| 项目 | 状态 |
|------|------|
| **延续信号** | {analysis.continuation_analysis.signal.value} |
| **入场时机** | {analysis.continuation_analysis.entry_timing} |
| **回调质量** | {analysis.continuation_analysis.pullback_quality} |
| **成交量确认** | {'✅ 确认' if analysis.continuation_analysis.volume_confirmation else '❌ 未确认'} |

---

## 🎯 关键价位

### 支撑阻力位
"""
        
        # 添加支撑阻力位信息
        if analysis.key_support_resistance.get('support'):
            report += "\n**支撑位**: "
            report += ", ".join([f"`{level:.4f}`" for level in analysis.key_support_resistance['support'][:3]])
        
        if analysis.key_support_resistance.get('resistance'):
            report += "\n\n**阻力位**: "
            report += ", ".join([f"`{level:.4f}`" for level in analysis.key_support_resistance['resistance'][:3]])
        
        # 关键技术位
        report += f"""

### 关键技术位
- **EMA关键位**: `{analysis.basic_trend.support_resistance.get('key_level', 0):.4f}`
- **支撑参考**: `{analysis.basic_trend.support_resistance.get('support', 0):.4f}`
- **阻力参考**: `{analysis.basic_trend.support_resistance.get('resistance', 0):.4f}`

---

## 💡 价格行为信号
"""
        
        # 添加价格行为信号
        if analysis.price_action_signals:
            for signal in analysis.price_action_signals:
                report += f"""
### {signal.signal_type}
- **描述**: {signal.description}
- **信号强度**: {signal.strength:.1%}
- **信心度**: {signal.confidence:.1%}
"""
        else:
            report += "\n> 暂无明确的价格行为信号\n"
        
        # 添加总体评价
        if analysis.overall_signal_strength > 0.8:
            overall_assessment = "🟢 **强势趋势，信号质量高，建议关注**"
        elif analysis.overall_signal_strength > 0.6:
            overall_assessment = "🟡 **中等强度趋势，需要谨慎关注**"
        elif analysis.overall_signal_strength > 0.4:
            overall_assessment = "🟠 **趋势较弱，建议等待更好信号**"
        else:
            overall_assessment = "🔴 **趋势不明确，建议观望**"
        
        report += f"""

---

## 📈 总体评价

{overall_assessment}

### 当前建议
根据当前分析结果，{analysis.continuation_analysis.entry_timing}。

---

## 📊 历史趋势统计

"""
        
        # 添加历史统计
        if len(self.trend_history) > 0:
            # 计算最近10次分析的趋势方向分布
            recent_records = self.trend_history[-10:] if len(self.trend_history) >= 10 else self.trend_history
            
            direction_counts = {}
            for record in recent_records:
                direction = record.get("trend_direction", "unknown")
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            report += "### 最近趋势分布\n"
            for direction, count in direction_counts.items():
                percentage = count / len(recent_records) * 100
                report += f"- **{direction}**: {count}次 ({percentage:.1f}%)\n"
        
        report += f"""

---

*报告由趋势监控机器人自动生成 | 监控交易所: {self.exchange_name} | 更新频率: {self.monitor_interval}秒*
"""
        
        return report

    async def _save_results_to_file(self):
        """保存结果到文件"""
        try:
            # 保存详细历史记录
            safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
            history_file = self.results_dir / f"trend_history_{safe_symbol}.json"
            
            # 确保历史数据经过类型转换
            converted_history = convert_numpy_types(self.trend_history)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(converted_history, f, indent=2, ensure_ascii=False)
            
            # 保存监控统计
            stats = {
                "symbol": self.symbol,
                "exchange": self.exchange_name,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "analysis_count": self.analysis_count,
                "monitoring_duration_minutes": self._get_monitoring_duration_minutes(),
                "config": self.config
            }
            
            # 转换统计数据类型
            converted_stats = convert_numpy_types(stats)
            stats_file = self.results_dir / f"monitoring_stats_{safe_symbol}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(converted_stats, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 结果已保存到: {history_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果文件错误: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
    
    def _log_monitoring_status(self, analysis: EnhancedTrendAnalysis):
        """输出监控状态"""
        current_time = datetime.now(timezone.utc)
        running_time = self._get_monitoring_duration_minutes()
        
        # 简化的状态信息
        status_info = f"""
🤖 [{current_time.strftime('%H:%M:%S')}] 趋势监控 #{self.analysis_count} | 运行时间: {running_time:.1f}分钟
📊 {analysis.basic_trend.direction.value} | {analysis.basic_trend.phase.value} | 强度: {analysis.overall_signal_strength:.1%}
⏰ {analysis.continuation_analysis.entry_timing} | 质量: {analysis.trend_quality_score:.1%} | 延续: {analysis.continuation_analysis.continuation_probability:.1%}
"""
        
        logger.info(status_info.strip())
        
        # 每10次分析输出详细摘要
        if self.analysis_count % 10 == 0:
            logger.info("📋 详细分析报告:")
            summary = self.trend_analyzer.get_trend_summary(analysis)
            for line in summary.split('\n'):
                if line.strip():
                    logger.info(line)
    
    def _get_monitoring_duration_minutes(self) -> float:
        """获取监控运行时长(分钟)"""
        if self.start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60
    
    async def stop_monitoring(self):
        """停止趋势监控"""
        self.bot_running = False
        
        # 停止数据加载器
        if self.data_loader:
            try:
                await self.data_loader.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ 数据加载器清理时发生错误: {e}")
        
        # 保存最终结果
        if self.trend_history:
            try:
                await self._save_results_to_file()
            except Exception as e:
                logger.warning(f"⚠️ 保存结果文件时发生错误: {e}")
        
        # 如果有最后一次分析结果，最终更新一次docs报告
        if self.last_analysis:
            try:
                await self._update_docs_report(self.last_analysis)
            except Exception as e:
                logger.warning(f"⚠️ 更新docs报告时发生错误: {e}")
        
        # 输出监控总结
        if self.analysis_count > 0:
            self._log_monitoring_summary()
        
        logger.info("✅ 趋势监控机器人已停止")
        if self.last_analysis:
            logger.info(f"📝 最终报告已保存到: {self.report_file_path}")
    
    def _log_monitoring_summary(self):
        """输出监控总结"""
        if self.analysis_count == 0:
            return
        
        duration = self._get_monitoring_duration_minutes()
        
        # 统计趋势方向分布
        direction_counts = {}
        phase_counts = {}
        
        for record in self.trend_history:
            direction = record.get("trend_direction", "unknown")
            phase = record.get("trend_phase", "unknown")
            
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # 平均信号强度
        avg_strength = sum(r.get("overall_signal_strength", 0) for r in self.trend_history) / len(self.trend_history)
        
        summary = f"""
📊 【趋势监控总结】
{'='*50}
⏰ 监控时长: {duration:.1f} 分钟
📈 分析次数: {self.analysis_count}
📊 平均信号强度: {avg_strength:.1%}

🎯 趋势方向分布:
{chr(10).join([f"▫️ {direction}: {count}次 ({count/self.analysis_count:.1%})" for direction, count in direction_counts.items()])}

📈 趋势阶段分布:
{chr(10).join([f"▫️ {phase}: {count}次 ({count/self.analysis_count:.1%})" for phase, count in phase_counts.items()])}

💾 结果文件保存在: {self.results_dir}
"""
        
        logger.info(summary)


async def run_trend_monitor(symbol: str = "BTC/USDT", exchange_name: str = "okx"):
    """运行趋势监控机器人"""
    
    # 配置参数
    config = {
        "exchange_config": {
            # 在这里添加你的交易所配置
            # "sandbox": True,  # 使用沙盒环境
            # "apiKey": "your_api_key",
            # "secret": "your_secret",
            # "password": "your_passphrase"  # OKX需要
        },
        "watch_timeframes": {
            "observed": "15m",   # 观测周期
            "trading": "5m",     # 交易周期
            "admission": "1m"    # 准入周期
        },
        "window_obs": 80,        # 观测窗口
        "monitor_interval": 60   # 监控间隔(秒)
    }
    
    # 创建并启动监控机器人
    bot = TrendMonitorBot(symbol, exchange_name, config)
    
    try:
        await bot.start_monitoring()
    except KeyboardInterrupt:
        logger.info("🛑 用户中断程序")
    finally:
        await bot.stop_monitoring()


if __name__ == "__main__":
    # 运行趋势监控机器人
    asyncio.run(run_trend_monitor())
