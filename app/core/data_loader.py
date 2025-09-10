import asyncio

from app.core.exchange_client import ExchangeClient
from loguru import logger
import pandas as pd

class OhlcvDataLoader:

    def __init__(self, exchange_name, symbol, config={}):
        self.symbol = symbol
        # 分离交易所配置和数据加载器配置
        exchange_config = config.get("exchange_config", {})
        self.exchange_client = ExchangeClient(exchange_name=exchange_name, config=exchange_config)
        self.is_running = False
        self.cache_ohlcv = {}
        self.watch_timeframes = config.get("watch_timeframes", {
            "observed": "15m",
            "trading": "5m",
            "admission": "1m"
        })
        self.window_obs = config.get("window_obs", 80)
        self._watch_task_ids = []
        
        self._external_callbacks = {}

    async def ohlcvc_callback(self, symbol, timeframe, ohlcvc_data):
        """回调函数：处理实时OHLCVC数据更新
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            ohlcvc_data: OHLCVC数据 [timestamp, open, high, low, close, volume, count]
        """
        if symbol != self.symbol or not ohlcvc_data or len(ohlcvc_data) < 6:
            return
            
        if not hasattr(self, '_timeframe_to_key_map'):
            self._timeframe_to_key_map = {v: k for k, v in self.watch_timeframes.items()}
        
        cache_key = self._timeframe_to_key_map.get(timeframe)
        if cache_key is None:
            logger.warning(f"未找到时间框架 {timeframe} 对应的缓存key")
            return
            
        if cache_key not in self.cache_ohlcv:
            logger.warning(f"缓存中不存在key: {cache_key}")
            return
            
        try:
            timestamp, open_val, high_val, low_val, close_val, volume_val = ohlcvc_data[:6]
            
            if not self._validate_ohlcv_data(open_val, high_val, low_val, close_val, volume_val):
                logger.warning(f"数据验证失败: {symbol} {timeframe}")
                return
            new_row_data = {
                'timestamp': timestamp,
                'open': float(open_val),
                'high': float(high_val), 
                'low': float(low_val),
                'close': float(close_val),
                'volume': float(volume_val)
            }
            # 格式化数据
            new_df = self.exchange_client.format_df([list(new_row_data.values())[:6]])
            if not new_df.empty:
                await self._update_cache_data(cache_key, new_df, timeframe)
        except Exception as e:
            logger.error(f"更新缓存数据时发生错误: {e}")
            logger.debug(f"原始ohlcvc数据: {ohlcvc_data}")

    @staticmethod
    def _validate_ohlcv_data(open_val, high_val, low_val, close_val, volume_val) -> bool:
        """验证OHLCV数据的有效性
        
        Args:
            open_val, high_val, low_val, close_val, volume_val: OHLCV值
            
        Returns:
            bool: 数据是否有效
        """
        try:
            o, h, l, c, v = float(open_val), float(high_val), float(low_val), float(close_val), float(volume_val)
            # 基本数值验证
            if any(x <= 0 for x in [o, h, l, c, v]):
                return False
            # OHLC逻辑验证
            if not (l <= o <= h and l <= c <= h):
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    async def _update_cache_data(self, cache_key: str, new_df: pd.DataFrame, timeframe: str):
        """更新缓存数据
        
        Args:
            cache_key: 缓存键
            new_df: 新数据DataFrame
            timeframe: 时间框架
        """
        current_df = self.cache_ohlcv[cache_key]
        if not current_df.empty:
            new_timestamp = new_df.index[0]
            # 检查是否是同一时间戳的数据更新
            if new_timestamp == current_df.index[-1]:
                # 更新最后一行数据（同一时间戳的实时更新）
                current_df.iloc[-1] = new_df.iloc[0]
                logger.debug(f"更新了 {cache_key} 的最后一行数据 (时间戳: {new_timestamp})")
            else:
                # 添加新的一行数据
                combined_df = pd.concat([current_df, new_df])
                # 保持最近500条数据
                self.cache_ohlcv[cache_key] = combined_df.tail(500)
                logger.debug(f"添加了 {cache_key} 的新数据行，当前共 {len(self.cache_ohlcv[cache_key])} 条")
                
                # 优化5: 触发数据更新事件 (为策略服务预留接口)
                await self._on_new_data_added(cache_key, timeframe, new_df.iloc[0])
        else:
            # 如果当前缓存为空，直接设置新数据
            self.cache_ohlcv[cache_key] = new_df
            logger.debug(f"初始化了 {cache_key} 的缓存数据")
    
    async def _on_new_data_added(self, cache_key: str, timeframe: str, new_row: pd.Series):
        """新数据添加时的事件处理 (为扩展预留)
        Args:
            cache_key: 缓存键
            timeframe: 时间框架
            new_row: 新行数据
        """
        for callback_name, callback_func in self._external_callbacks.items():
            try:
                if asyncio.iscoroutinefunction(callback_func):
                    await callback_func(self.symbol, timeframe, cache_key, new_row)
                else:
                    callback_func(self.symbol, timeframe, cache_key, new_row)
            except Exception as e:
                logger.error(f"外部回调函数 {callback_name} 执行失败: {e}")
    
    def register_callback(self, name: str, callback_func):
        """注册外部回调函数
        
        Args:
            name: 回调函数名称
            callback_func: 回调函数，签名为 func(symbol, timeframe, cache_key, new_row)
        """
        self._external_callbacks[name] = callback_func
        logger.info(f"✅ 注册外部回调函数: {name}")
    
    def unregister_callback(self, name: str):
        """注销外部回调函数
        
        Args:
            name: 回调函数名称
        """
        if name in self._external_callbacks:
            del self._external_callbacks[name]
            logger.info(f"🗑️ 注销外部回调函数: {name}")
        else:
            logger.warning(f"⚠️ 回调函数 {name} 不存在")
    
    def get_data_statistics(self) -> dict:
        """获取数据统计信息
        
        Returns:
            dict: 各时间框架的数据统计
        """
        stats = {}
        for cache_key, timeframe in self.watch_timeframes.items():
            data = self.cache_ohlcv.get(cache_key)
            if data is not None and not data.empty:
                stats[cache_key] = {
                    'timeframe': timeframe,
                    'data_count': len(data),
                    'latest_time': data.index[-1].isoformat() if len(data) > 0 else None,
                    'latest_price': float(data['close'].iloc[-1]) if len(data) > 0 else None,
                    'price_range': {
                        'high': float(data['high'].max()),
                        'low': float(data['low'].min())
                    } if len(data) > 0 else None
                }
            else:
                stats[cache_key] = {
                    'timeframe': timeframe,
                    'data_count': 0,
                    'latest_time': None,
                    'latest_price': None,
                    'price_range': None
                }
        return stats

    async def init_ohlcv(self):
        try:
            data_limit = max(self.window_obs * 3, 300)  # 至少300条数据确保EMA144可用
            if len(self.watch_timeframes) != 3:
                self.is_running = False
                return False
            tasks = [
                self.exchange_client.get_ohlcv_data(self.symbol, watch_timeframe, limit=data_limit)
                for watch_timeframe in self.watch_timeframes.values()
            ]
            results = await asyncio.gather(*tasks)
            for i, k in enumerate(self.watch_timeframes.keys()):
                self.cache_ohlcv[k] = results[i]

            logger.info(f"✅ 数据初始化成功: {self.symbol} ({len(results[0])}/{len(results[1])}/{len(results[2])} 条)")
            return True
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            self.is_running = False
            return False

    async def watch_ohlcv(self):
        """启动OHLCV数据监控"""
        self.is_running = True
        logger.info("✅ 开始数据监控")
        
        try:
            # 初始化数据
            if not await self.init_ohlcv():
                logger.error("❌ 无法初始化数据，停止监控")
                return
            
            # 验证时间框架配置
            if len(self.watch_timeframes) != 3:
                logger.error(f"❌ 时间框架配置错误，期望3个，实际{len(self.watch_timeframes)}个")
                self.is_running = False
                return
            
            # 启动观测任务
            for watch_timeframe in self.watch_timeframes.values():
                task_id = self.exchange_client.start_watch_ohlcvc(
                    self.symbol, 
                    watch_timeframe, 
                    callback=self.ohlcvc_callback
                )
                self._watch_task_ids.append(task_id)
                logger.info(f"📊 启动观测任务: {self.symbol}-{watch_timeframe}, ID: {task_id}")
            
            # 保持运行状态，直到被停止
            logger.info("🔄 数据监控运行中...")
            while self.is_running:
                # 检查观测任务状态
                watch_status = self.exchange_client.get_watch_status()
                active_tasks = sum(1 for status in watch_status.values() if status)
                
                if active_tasks == 0:
                    logger.warning("⚠️ 所有观测任务都已停止")
                    break
                    
                # 每10秒检查一次状态
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ 数据监控过程中发生错误: {e}")
        finally:
            # 清理资源
            await self.stop_watch_ohlcv()
    
    async def stop_watch_ohlcv(self):
        """停止OHLCV数据监控"""
        self.is_running = False
        
        if hasattr(self, '_watch_task_ids'):
            logger.info("🛑 停止所有观测任务...")
            for task_id in self._watch_task_ids:
                symbol, timeframe = task_id.split('_', 1)
                await self.exchange_client.stop_watch_ohlcvc(symbol, timeframe)
            self._watch_task_ids = []
            
        logger.info("✅ 数据监控已停止")
    
    async def cleanup(self):
        """清理所有资源"""
        try:
            # 停止数据监控
            await self.stop_watch_ohlcv()
            
            # 关闭交易所连接
            if hasattr(self, 'exchange_client'):
                await self.exchange_client.close_exchange()
                
            # 清空缓存数据
            self.cache_ohlcv.clear()
            
            logger.info("🧹 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理时发生错误: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
        if exc_type:
            logger.error(f"❌ 上下文管理器退出时发生异常: {exc_type.__name__}: {exc_val}")
        return False


