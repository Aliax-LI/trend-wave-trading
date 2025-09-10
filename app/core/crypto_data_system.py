"""
数字货币交易数据系统
集成策略优化的数据加载器和监控器
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import pandas as pd

from app.core.strategy_data_loader import StrategyOptimizedLoader, PerpetualMonitor, OrderMonitor
from app.core.exchange_client import ExchangeClient
from app.analysis.trend.advanced_trend_analyzer import AdvancedTrendAnalyzer


class CryptoTradingDataSystem:
    """数字货币交易数据系统
    
    集成ExchangeClient、StrategyOptimizedLoader、PerpetualMonitor、OrderMonitor
    为交易策略提供统一的数据服务
    """
    
    def __init__(self, exchange_name: str = "okx", symbols: List[str] = None, config: Dict = None):
        self.exchange_name = exchange_name
        self.symbols = symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        self.config = config or {}
        
        # 核心组件
        self.exchange_client = ExchangeClient(exchange_name, self.config.get("exchange_config", {}))
        self.data_loaders: Dict[str, StrategyOptimizedLoader] = {}
        self.perpetual_monitor = PerpetualMonitor(self.exchange_client)
        self.order_monitor = OrderMonitor()
        
        # 趋势分析器
        self.trend_analyzer = AdvancedTrendAnalyzer(self.config.get("trend_analyzer_config", {}))
        
        # 趋势分析结果缓存
        self.trend_analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        
        logger.info(f"🎯 数字货币交易数据系统创建: {exchange_name}, 币种: {len(self.symbols)}个")
    
    async def initialize(self) -> bool:
        """初始化系统"""
        try:
            logger.info("🚀 开始初始化数字货币交易数据系统...")
            
            # 为每个币种创建数据加载器
            for symbol in self.symbols:
                logger.info(f"📊 初始化数据加载器: {symbol}")
                
                loader = StrategyOptimizedLoader(
                    exchange_name=self.exchange_name,
                    symbol=symbol,
                    config=self.config.get("data_loader_config", {})
                )
                
                self.data_loaders[symbol] = loader
                
                # 启动数据监控
                loader_task = asyncio.create_task(loader.watch_ohlcv())
                
                # 等待数据准备就绪
                data_ready = await loader.wait_for_data_ready(timeout=60)
                if not data_ready:
                    logger.warning(f"⚠️ {symbol} 数据未完全准备就绪，但继续运行")
                
                # 启动资金费率监控
                await self.perpetual_monitor.monitor_funding_rate(symbol)
            
            self.is_initialized = True
            self.is_running = True
            logger.info("✅ 数字货币交易数据系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            self.is_initialized = False
            return False
    
    async def start_monitoring(self):
        """启动系统监控"""
        if not self.is_initialized:
            logger.error("❌ 系统未初始化，无法启动监控")
            return
        
        logger.info("🔄 启动系统监控...")
        
        # 启动定期任务
        monitoring_tasks = [
            asyncio.create_task(self._funding_rate_monitor_task()),
            asyncio.create_task(self._system_health_monitor_task()),
            asyncio.create_task(self._cleanup_task()),
            asyncio.create_task(self._trend_analysis_monitor_task())
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            logger.error(f"❌ 监控任务异常: {e}")
    
    async def _funding_rate_monitor_task(self):
        """资金费率监控任务"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    await self.perpetual_monitor.monitor_funding_rate(symbol)
                
                # 每5分钟检查一次资金费率
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"资金费率监控异常: {e}")
                await asyncio.sleep(60)
    
    async def _system_health_monitor_task(self):
        """系统健康监控任务"""
        while self.is_running:
            try:
                # 检查数据加载器状态
                for symbol, loader in self.data_loaders.items():
                    if not loader.is_running:
                        logger.warning(f"⚠️ 数据加载器停止运行: {symbol}")
                
                # 每30秒检查一次系统健康
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"系统健康监控异常: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """清理任务"""
        while self.is_running:
            try:
                # 清理过期的价格位
                self.order_monitor.cleanup_old_levels(max_age_hours=24)
                
                # 每小时清理一次
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"清理任务异常: {e}")
                await asyncio.sleep(300)
    
    async def _trend_analysis_monitor_task(self):
        """趋势分析监控任务"""
        # 初始等待，让数据加载完成
        await asyncio.sleep(60)
        
        while self.is_running:
            try:
                logger.info("🎯 执行定期趋势分析...")
                
                # 批量执行趋势分析
                results = await self.batch_trend_analysis(force_refresh=False)
                
                # 记录分析结果摘要
                for symbol, result in results.items():
                    if 'error' not in result:
                        direction_en = result.get('trend_direction', {}).get('direction', 'unknown')
                        phase_en = result.get('trend_phase', {}).get('phase', 'unknown')
                        prob = result.get('trend_continuation', {}).get('probability', 0.0)
                        
                        # 转换为中文显示
                        direction_cn = self._get_direction_chinese(direction_en)
                        phase_cn = self._get_phase_chinese(phase_en)
                        
                        logger.info(f"📊 {symbol}: {direction_cn} - {phase_cn} (延续概率: {prob:.1%})")
                    else:
                        logger.warning(f"⚠️ {symbol} 趋势分析失败: {result['error']}")
                
                # 每10分钟执行一次趋势分析
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"趋势分析监控异常: {e}")
                await asyncio.sleep(60)
    
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
    
    def get_trend_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取趋势分析数据（1小时）"""
        if symbol in self.data_loaders:
            return self.data_loaders[symbol].cache_ohlcv.get('trend_filter')
        return None
    
    def get_pattern_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取形态识别数据（15分钟）"""
        if symbol in self.data_loaders:
            return self.data_loaders[symbol].cache_ohlcv.get('signal_main')
        return None
    
    def get_entry_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取入场时机数据（5分钟）"""
        if symbol in self.data_loaders:
            return self.data_loaders[symbol].cache_ohlcv.get('entry_timing')
        return None
    
    async def perform_trend_analysis(self, symbol: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """执行趋势分析"""
        try:
            # 检查缓存
            if not force_refresh and symbol in self.trend_analysis_cache:
                cache_time = self.trend_analysis_cache[symbol].get('timestamp')
                if cache_time:
                    # 缓存5分钟有效
                    cache_dt = datetime.fromisoformat(cache_time.replace('Z', '+00:00'))
                    if (datetime.now() - cache_dt).total_seconds() < 300:
                        logger.debug(f"🔄 使用缓存的趋势分析结果: {symbol}")
                        return self.trend_analysis_cache[symbol]
            
            # 获取趋势分析数据（1小时）
            trend_data = self.get_trend_data(symbol)
            if trend_data is None or trend_data.empty:
                logger.warning(f"⚠️ 无法获取趋势数据: {symbol}")
                return None
            
            if len(trend_data) < 144:  # 确保数据长度足够
                logger.warning(f"⚠️ 趋势数据长度不足: {symbol}, 需要144条，当前{len(trend_data)}条")
                return None
            
            logger.info(f"🎯 开始趋势分析: {symbol}, 数据长度: {len(trend_data)}")
            
            # 执行综合趋势分析
            analysis_result = self.trend_analyzer.comprehensive_trend_analysis(trend_data)
            
            if analysis_result:
                # 添加币种信息
                analysis_result['symbol'] = symbol
                analysis_result['data_length'] = len(trend_data)
                
                # 更新缓存
                self.trend_analysis_cache[symbol] = analysis_result
                
                logger.info(f"✅ 趋势分析完成: {symbol} - {analysis_result.get('trend_direction', {}).get('direction', 'unknown')}")
                return analysis_result
            else:
                logger.error(f"❌ 趋势分析失败: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 趋势分析异常: {symbol} - {e}")
            return None
    
    def get_latest_trend_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新的趋势分析结果"""
        return self.trend_analysis_cache.get(symbol)
    
    async def batch_trend_analysis(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """批量执行趋势分析"""
        results = {}
        
        for symbol in self.symbols:
            try:
                analysis = await self.perform_trend_analysis(symbol, force_refresh)
                if analysis:
                    results[symbol] = analysis
                else:
                    results[symbol] = {'error': '分析失败'}
            except Exception as e:
                logger.error(f"❌ 批量趋势分析失败: {symbol} - {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'exchange': self.exchange_name,
            'symbols': self.symbols,
            'data_loaders': {},
            'funding_rates': {},
            'active_price_levels': {},
            'trend_analysis': {}
        }
        
        # 数据加载器状态
        for symbol, loader in self.data_loaders.items():
            status['data_loaders'][symbol] = loader.get_data_status()
        
        # 资金费率状态
        for symbol in self.symbols:
            rate = self.perpetual_monitor.get_current_funding_rate(symbol)
            time_to_settlement = self.perpetual_monitor.get_time_to_settlement(symbol)
            status['funding_rates'][symbol] = {
                'rate': rate,
                'rate_pct': f"{rate:.4%}" if rate else "N/A",
                'time_to_settlement_minutes': time_to_settlement
            }
        
        # 活跃价格位状态
        for symbol in self.symbols:
            active_levels = self.order_monitor.get_active_levels(symbol)
            status['active_price_levels'][symbol] = len(active_levels)
        
        # 趋势分析状态
        for symbol in self.symbols:
            analysis = self.get_latest_trend_analysis(symbol)
            if analysis:
                status['trend_analysis'][symbol] = {
                    'trend_direction': analysis.get('trend_direction', {}).get('direction', 'unknown'),
                    'trend_phase': analysis.get('trend_phase', {}).get('phase', 'unknown'),
                    'continuation_probability': analysis.get('trend_continuation', {}).get('probability', 0.0),
                    'last_updated': analysis.get('timestamp', 'unknown'),
                    'current_price': analysis.get('current_price', 0.0)
                }
            else:
                status['trend_analysis'][symbol] = {
                    'trend_direction': 'no_data',
                    'trend_phase': 'no_data',
                    'continuation_probability': 0.0,
                    'last_updated': 'never',
                    'current_price': 0.0
                }
        
        return status
    
    def register_strategy_callback(self, symbol: str, callback_type: str, callback_func):
        """注册策略回调函数"""
        if symbol in self.data_loaders:
            self.data_loaders[symbol].register_strategy_callback(callback_type, callback_func)
        else:
            logger.warning(f"⚠️ 未找到币种数据加载器: {symbol}")
    
    async def stop(self):
        """停止系统"""
        logger.info("🛑 停止数字货币交易数据系统...")
        
        self.is_running = False
        
        # 停止所有数据加载器
        for symbol, loader in self.data_loaders.items():
            try:
                await loader.cleanup()
            except Exception as e:
                logger.error(f"停止数据加载器失败 {symbol}: {e}")
        
        # 关闭交易所连接
        try:
            await self.exchange_client.close_exchange()
        except Exception as e:
            logger.error(f"关闭交易所连接失败: {e}")
        
        logger.info("✅ 数字货币交易数据系统已停止")


class StrategyDataService:
    """为交易策略提供数据服务"""
    
    def __init__(self, data_system: CryptoTradingDataSystem):
        self.data_system = data_system
        logger.info("🎯 策略数据服务初始化完成")
    
    async def get_trend_analysis_data(self, symbol: str) -> Dict[str, Any]:
        """获取趋势分析所需数据"""
        trend_data = self.data_system.get_trend_data(symbol)
        
        if trend_data is None or trend_data.empty:
            return {
                'hourly_data': pd.DataFrame(),
                'ema_ready': False,
                'funding_rate': 0.0,
                'data_ready': False,
                'data_count': 0
            }
        
        return {
            'hourly_data': trend_data,
            'ema_ready': len(trend_data) >= 144,
            'funding_rate': self.data_system.perpetual_monitor.get_current_funding_rate(symbol),
            'data_ready': True,
            'latest_price': float(trend_data['close'].iloc[-1]),
            'data_count': len(trend_data),
            'latest_time': trend_data.index[-1]
        }
    
    async def perform_comprehensive_trend_analysis(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """执行综合趋势分析"""
        return await self.data_system.perform_trend_analysis(symbol, force_refresh)
    
    def get_current_trend_analysis(self, symbol: str) -> Dict[str, Any]:
        """获取当前趋势分析结果"""
        analysis = self.data_system.get_latest_trend_analysis(symbol)
        if not analysis:
            return {
                'available': False,
                'message': '暂无趋势分析数据'
            }
        
        return {
            'available': True,
            'symbol': symbol,
            'trend_direction': analysis.get('trend_direction', {}),
            'trend_phase': analysis.get('trend_phase', {}),
            'trend_continuation': analysis.get('trend_continuation', {}),
            'key_levels': analysis.get('key_levels', {}),
            'trading_signals': analysis.get('trading_signals', []),
            'technical_summary': analysis.get('technical_summary', {}),
            'timestamp': analysis.get('timestamp'),
            'current_price': analysis.get('current_price')
        }
    
    async def get_trend_analysis_summary(self) -> Dict[str, Dict[str, Any]]:
        """获取所有币种的趋势分析摘要"""
        summary = {}
        
        for symbol in self.data_system.symbols:
            analysis = self.get_current_trend_analysis(symbol)
            
            if analysis['available']:
                summary[symbol] = {
                    'trend_direction': analysis['trend_direction'].get('direction', 'unknown'),
                    'direction_strength': analysis['trend_direction'].get('strength', 0.0),
                    'trend_phase': analysis['trend_phase'].get('phase', 'unknown'),
                    'phase_strength': analysis['trend_phase'].get('strength', 0.0),
                    'continuation_probability': analysis['trend_continuation'].get('probability', 0.0),
                    'continuation_strength': analysis['trend_continuation'].get('strength', 'unknown'),
                    'current_price': analysis.get('current_price', 0.0),
                    'signal_count': len(analysis['trading_signals']),
                    'last_updated': analysis.get('timestamp')
                }
            else:
                summary[symbol] = {
                    'trend_direction': 'no_data',
                    'direction_strength': 0.0,
                    'trend_phase': 'no_data',
                    'phase_strength': 0.0,
                    'continuation_probability': 0.0,
                    'continuation_strength': 'no_data',
                    'current_price': 0.0,
                    'signal_count': 0,
                    'last_updated': 'never'
                }
        
        return summary
    
    async def get_pattern_analysis_data(self, symbol: str) -> Dict[str, Any]:
        """获取形态识别所需数据"""
        pattern_data = self.data_system.get_pattern_data(symbol)
        
        if pattern_data is None or pattern_data.empty:
            return {
                'fifteen_min_data': pd.DataFrame(),
                'pattern_ready': False,
                'current_price': 0.0,
                'data_ready': False,
                'data_count': 0
            }
        
        return {
            'fifteen_min_data': pattern_data,
            'pattern_ready': len(pattern_data) >= 200,
            'current_price': float(pattern_data['close'].iloc[-1]),
            'data_ready': True,
            'data_count': len(pattern_data),
            'latest_time': pattern_data.index[-1]
        }
    
    async def get_entry_signal_data(self, symbol: str) -> Dict[str, Any]:
        """获取入场信号所需数据"""
        entry_data = self.data_system.get_entry_data(symbol)
        
        if entry_data is None or entry_data.empty:
            return {
                'five_min_data': pd.DataFrame(),
                'signal_ready': False,
                'approaching_levels': [],
                'current_price': 0.0,
                'data_ready': False,
                'data_count': 0
            }
        
        current_price = float(entry_data['close'].iloc[-1])
        approaching_levels = self.data_system.order_monitor.check_price_approach(symbol, current_price)
        
        return {
            'five_min_data': entry_data,
            'signal_ready': len(entry_data) >= 50,
            'approaching_levels': approaching_levels,
            'current_price': current_price,
            'data_ready': True,
            'data_count': len(entry_data),
            'latest_time': entry_data.index[-1]
        }
    
    async def register_order_level(self, symbol: str, price: float, level_type: str, description: str = ""):
        """注册挂单关键位"""
        self.data_system.order_monitor.add_price_level(symbol, price, level_type, description)
    
    async def get_risk_management_data(self, symbol: str) -> Dict[str, Any]:
        """获取风险管理数据"""
        entry_data = self.data_system.get_entry_data(symbol)
        
        if entry_data is None or entry_data.empty:
            return {
                'current_price': 0.0,
                'current_volume': 0.0,
                'funding_rate': 0.0,
                'atr': 0.0,
                'data_ready': False
            }
        
        current_price = float(entry_data['close'].iloc[-1])
        current_volume = float(entry_data['volume'].iloc[-1])
        funding_rate = self.data_system.perpetual_monitor.get_current_funding_rate(symbol)
        atr = self.calculate_atr(entry_data)
        
        return {
            'current_price': current_price,
            'current_volume': current_volume,
            'funding_rate': funding_rate,
            'atr': atr,
            'data_ready': len(entry_data) >= 50,
            'time_to_funding_settlement': self.data_system.perpetual_monitor.get_time_to_settlement(symbol)
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR（平均真实范围）"""
        if data is None or data.empty or len(data) < period:
            return 0.0
        
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return 0.0
    
    async def get_multi_symbol_summary(self) -> Dict[str, Dict[str, Any]]:
        """获取多币种数据摘要"""
        summary = {}
        
        for symbol in self.data_system.symbols:
            try:
                trend_data = await self.get_trend_analysis_data(symbol)
                pattern_data = await self.get_pattern_analysis_data(symbol)
                entry_data = await self.get_entry_signal_data(symbol)
                
                summary[symbol] = {
                    'trend_ready': trend_data['ema_ready'],
                    'pattern_ready': pattern_data['pattern_ready'],
                    'entry_ready': entry_data['signal_ready'],
                    'current_price': entry_data['current_price'],
                    'funding_rate': trend_data['funding_rate'],
                    'approaching_levels': len(entry_data['approaching_levels'])
                }
                
            except Exception as e:
                logger.error(f"获取 {symbol} 摘要失败: {e}")
                summary[symbol] = {
                    'trend_ready': False,
                    'pattern_ready': False,
                    'entry_ready': False,
                    'current_price': 0.0,
                    'funding_rate': 0.0,
                    'approaching_levels': 0
                }
        
        return summary
