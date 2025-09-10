"""
策略优化的数据加载器
专为数字货币日内交易策略服务，基于现有OhlcvDataLoader进行优化
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from loguru import logger
import pandas as pd

from app.core.data_loader import OhlcvDataLoader
from app.core.exchange_client import ExchangeClient


class StrategyOptimizedLoader(OhlcvDataLoader):
    """策略优化的数据加载器
    
    专门为顺势复杂回调策略服务，优化数据结构和回调机制
    """
    
    def __init__(self, exchange_name: str, symbol: str, config: dict = None):
        # 使用策略专用的时间周期配置
        strategy_config = config or {}
        strategy_config["watch_timeframes"] = {
            "trend_filter": "1h",      # 趋势过滤 - 1小时
            "signal_main": "15m",      # 主信号生成 - 15分钟
            "entry_timing": "5m"       # 入场时机 - 5分钟
        }
        strategy_config["window_obs"] = 200  # 增加观测窗口，确保EMA144可用
        
        super().__init__(exchange_name, symbol, strategy_config)
        
        # 策略相关的实时监控
        self.current_funding_rate = None
        self.price_alerts = []
        self.volume_alerts = []
        self.last_price_check = {}
        
        # 策略回调函数
        self.strategy_callbacks = {
            'trend_change': None,
            'pattern_update': None, 
            'entry_signal': None
        }
        
        logger.info(f"🎯 策略优化数据加载器初始化: {symbol}")
    
    def register_strategy_callback(self, callback_type: str, callback_func: Callable):
        """注册策略回调函数
        
        Args:
            callback_type: 回调类型 ('trend_change', 'pattern_update', 'entry_signal')
            callback_func: 回调函数
        """
        if callback_type in self.strategy_callbacks:
            self.strategy_callbacks[callback_type] = callback_func
            logger.info(f"✅ 注册策略回调: {callback_type}")
        else:
            logger.warning(f"⚠️ 未知回调类型: {callback_type}")
    
    async def ohlcvc_callback(self, symbol: str, timeframe: str, ohlcvc_data):
        """增强版实时数据回调"""
        # 调用原始数据更新
        await super().ohlcvc_callback(symbol, timeframe, ohlcvc_data)
        
        # 策略相关的实时处理
        try:
            # 基于配置的时间框架判断，而不是硬编码
            if timeframe == self.watch_timeframes.get("entry_timing"):
                await self.check_entry_signals(symbol, ohlcvc_data)
            elif timeframe == self.watch_timeframes.get("signal_main"):
                await self.check_pattern_signals(symbol, ohlcvc_data)
            elif timeframe == self.watch_timeframes.get("trend_filter"):
                await self.check_trend_signals(symbol, ohlcvc_data)
        except Exception as e:
            logger.error(f"策略回调处理失败: {e}")
    
    async def check_entry_signals(self, symbol: str, ohlcvc_data):
        """检查入场信号 - 基于配置的入场时机数据"""
        try:
            close_price = float(ohlcvc_data[4])
            volume = float(ohlcvc_data[5])
            
            # 价格突破监控
            if self.is_price_breakout(symbol, close_price):
                logger.info(f"🚀 价格突破信号: {symbol} {close_price}")
                if self.strategy_callbacks['entry_signal']:
                    await self.strategy_callbacks['entry_signal']('price_breakout', {
                        'symbol': symbol,
                        'price': close_price,
                        'timestamp': datetime.now()
                    })
            
            # 成交量异常监控
            if self.is_volume_spike(symbol, volume):
                logger.info(f"📊 成交量突破信号: {symbol} {volume}")
                if self.strategy_callbacks['entry_signal']:
                    await self.strategy_callbacks['entry_signal']('volume_spike', {
                        'symbol': symbol,
                        'volume': volume,
                        'timestamp': datetime.now()
                    })
            
            # 更新最后价格检查
            self.last_price_check[symbol] = {
                'price': close_price,
                'volume': volume,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"入场信号检查失败: {e}")
    
    async def check_pattern_signals(self, symbol: str, ohlcvc_data):
        """检查形态信号 - 基于配置的主信号数据"""
        try:
            close_price = float(ohlcvc_data[4])
            
            # 为形态识别提供实时数据更新通知
            logger.debug(f"{self.watch_timeframes.get('signal_main')}数据更新: {symbol} @ {close_price}")
            
            if self.strategy_callbacks['pattern_update']:
                await self.strategy_callbacks['pattern_update']({
                    'symbol': symbol,
                    'timeframe': self.watch_timeframes.get('signal_main'),
                    'price': close_price,
                    'data_ready': len(self.cache_ohlcv.get('signal_main', [])) >= 200
                })
                
        except Exception as e:
            logger.error(f"形态信号检查失败: {e}")
    
    async def check_trend_signals(self, symbol: str, ohlcvc_data):
        """检查趋势信号 - 基于配置的趋势过滤数据"""
        try:
            close_price = float(ohlcvc_data[4])
            
            # 为趋势分析提供实时数据更新通知
            logger.debug(f"{self.watch_timeframes.get('trend_filter')}数据更新: {symbol} @ {close_price}")
            
            if self.strategy_callbacks['trend_change']:
                await self.strategy_callbacks['trend_change']({
                    'symbol': symbol,
                    'timeframe': self.watch_timeframes.get('trend_filter'),
                    'price': close_price,
                    'data_ready': len(self.cache_ohlcv.get('trend_filter', [])) >= 144
                })
                
        except Exception as e:
            logger.error(f"趋势信号检查失败: {e}")
    
    def is_price_breakout(self, symbol: str, current_price: float, threshold: float = 0.005) -> bool:
        """检测价格突破 - 0.5%阈值"""
        if symbol not in self.last_price_check:
            return False
        
        last_check = self.last_price_check[symbol]
        time_diff = (datetime.now() - last_check['timestamp']).total_seconds()
        
        # 至少间隔1分钟才检查突破
        if time_diff < 60:
            return False
        
        price_change = abs(current_price - last_check['price']) / last_check['price']
        return price_change > threshold
    
    def is_volume_spike(self, symbol: str, current_volume: float, spike_ratio: float = 2.0) -> bool:
        """检测成交量异常 - 2倍异常阈值"""
        # 获取5分钟数据计算平均成交量
        entry_data = self.cache_ohlcv.get('entry_timing')
        if entry_data is None or entry_data.empty or len(entry_data) < 20:
            return False
        
        # 计算最近20个周期的平均成交量
        avg_volume = entry_data['volume'].tail(20).mean()
        return current_volume > avg_volume * spike_ratio
    
    def get_data_status(self) -> Dict[str, Any]:
        """获取数据状态"""
        status = {}
        for key, timeframe in self.watch_timeframes.items():
            data = self.cache_ohlcv.get(key)
            if data is not None and not data.empty:
                status[key] = {
                    'timeframe': timeframe,
                    'data_length': len(data),
                    'latest_time': data.index[-1].isoformat() if len(data) > 0 else None,
                    'latest_price': float(data['close'].iloc[-1]) if len(data) > 0 else None,
                    'ready_for_analysis': len(data) >= self.get_min_data_length(key)
                }
            else:
                status[key] = {
                    'timeframe': timeframe,
                    'data_length': 0,
                    'ready_for_analysis': False
                }
        
        return status
    
    def get_min_data_length(self, cache_key: str) -> int:
        """获取分析所需的最小数据长度"""
        min_lengths = {
            'trend_filter': 144,    # 需要144周期计算EMA144
            'signal_main': 200,     # 需要200周期进行形态识别
            'entry_timing': 50      # 需要50周期进行入场信号
        }
        return min_lengths.get(cache_key, 50)
    
    async def wait_for_data_ready(self, timeout: int = 300) -> bool:
        """等待数据准备就绪
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否所有数据都准备就绪
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = self.get_data_status()
            all_ready = all(s['ready_for_analysis'] for s in status.values())
            
            if all_ready:
                logger.info("✅ 所有策略数据已准备就绪")
                return True
            
            # 每10秒检查一次
            await asyncio.sleep(10)
            
            # 显示进度
            ready_count = sum(1 for s in status.values() if s['ready_for_analysis'])
            total_count = len(status)
            logger.info(f"📊 数据准备进度: {ready_count}/{total_count}")
        
        logger.warning("⚠️ 数据准备超时")
        return False


class PerpetualMonitor:
    """永续合约实时监控"""
    
    def __init__(self, exchange_client: ExchangeClient):
        self.exchange_client = exchange_client
        self.funding_rates = {}
        self.monitoring_symbols = set()
        
    async def monitor_funding_rate(self, symbol: str):
        """监控资金费率"""
        try:
            rate = await self.exchange_client.get_funding_rate(symbol)
            if rate is not None:
                self.funding_rates[symbol] = {
                    'rate': rate,
                    'timestamp': datetime.now(),
                    'next_settlement': self.calculate_next_settlement()
                }
                
                # 高费率预警
                if abs(rate) > 0.001:  # 0.1%
                    logger.warning(f"💰 高资金费率预警: {symbol} {rate:.4%}")
                else:
                    logger.debug(f"💰 资金费率更新: {symbol} {rate:.4%}")
                    
                self.monitoring_symbols.add(symbol)
                return rate
                    
        except Exception as e:
            logger.error(f"获取资金费率失败 {symbol}: {e}")
        
        return None
    
    def get_current_funding_rate(self, symbol: str) -> float:
        """获取当前资金费率"""
        if symbol in self.funding_rates:
            return self.funding_rates[symbol]['rate']
        return 0.0
    
    def calculate_next_settlement(self) -> datetime:
        """计算下次资金费率结算时间"""
        now = datetime.now()
        # 大多数交易所每8小时结算一次 (00:00, 08:00, 16:00 UTC)
        hour = now.hour
        next_hour = ((hour // 8) + 1) * 8
        if next_hour >= 24:
            next_settlement = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_settlement = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        return next_settlement
    
    def get_time_to_settlement(self, symbol: str) -> int:
        """获取距离下次结算的时间（分钟）"""
        if symbol in self.funding_rates:
            next_settlement = self.funding_rates[symbol]['next_settlement']
            time_diff = next_settlement - datetime.now()
            return max(0, int(time_diff.total_seconds() / 60))
        return 0


class OrderMonitor:
    """挂单监控器"""
    
    def __init__(self):
        self.price_levels = {}  # 关键价格位
        self.order_zones = {}   # 挂单区域
        
    def add_price_level(self, symbol: str, price: float, level_type: str, description: str = ""):
        """添加关键价格位
        
        Args:
            symbol: 交易对符号
            price: 价格
            level_type: 类型 ('support', 'resistance', 'entry', 'stop', 'target')
            description: 描述
        """
        if symbol not in self.price_levels:
            self.price_levels[symbol] = []
            
        self.price_levels[symbol].append({
            'price': price,
            'type': level_type,
            'description': description,
            'timestamp': datetime.now(),
            'triggered': False
        })
        
        logger.info(f"📍 添加关键价格位: {symbol} {price} ({level_type}) - {description}")
    
    def check_price_approach(self, symbol: str, current_price: float, 
                           threshold: float = 0.005) -> list:
        """检查价格接近关键位
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
            threshold: 接近阈值（默认0.5%）
            
        Returns:
            list: 接近的价格位列表
        """
        if symbol not in self.price_levels:
            return []
            
        approaching_levels = []
        for level in self.price_levels[symbol]:
            if level['triggered']:
                continue
                
            distance = abs(current_price - level['price']) / level['price']
            if distance <= threshold:
                approaching_levels.append({
                    **level,
                    'distance_pct': distance,
                    'current_price': current_price
                })
                
        return approaching_levels
    
    def mark_level_triggered(self, symbol: str, price: float, tolerance: float = 0.002):
        """标记价格位已触发
        
        Args:
            symbol: 交易对符号
            price: 触发价格
            tolerance: 容忍度（默认0.2%）
        """
        if symbol not in self.price_levels:
            return
            
        for level in self.price_levels[symbol]:
            if level['triggered']:
                continue
                
            distance = abs(price - level['price']) / level['price']
            if distance <= tolerance:
                level['triggered'] = True
                level['triggered_time'] = datetime.now()
                level['triggered_price'] = price
                
                logger.info(f"🎯 价格位触发: {symbol} {level['type']} @ {price} "
                          f"(目标: {level['price']}, 偏差: {distance:.3%})")
    
    def cleanup_old_levels(self, max_age_hours: int = 24):
        """清理过期的价格位"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for symbol in self.price_levels:
            original_count = len(self.price_levels[symbol])
            self.price_levels[symbol] = [
                level for level in self.price_levels[symbol]
                if level['timestamp'] > cutoff_time
            ]
            
            removed_count = original_count - len(self.price_levels[symbol])
            if removed_count > 0:
                logger.debug(f"🧹 清理过期价格位: {symbol} 移除 {removed_count} 个")
    
    def get_active_levels(self, symbol: str) -> list:
        """获取活跃的价格位"""
        if symbol not in self.price_levels:
            return []
            
        return [level for level in self.price_levels[symbol] if not level['triggered']]
