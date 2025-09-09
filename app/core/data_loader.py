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

    async def ohlcvc_callback(self, symbol, timeframe, ohlcvc_data):
        """回调函数：处理实时OHLCVC数据更新
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            ohlcvc_data: OHLCVC数据 [timestamp, open, high, low, close, volume, count]
        """
        if symbol != self.symbol:
            return
        # 找到对应的缓存key
        cache_key = None
        for k, v in self.watch_timeframes.items():
            if v == timeframe:
                cache_key = k
                break
                
        if cache_key is None:
            logger.warning(f"未找到时间框架 {timeframe} 对应的缓存key")
            return
            
        if cache_key not in self.cache_ohlcv:
            logger.warning(f"缓存中不存在key: {cache_key}")
            return
            
        try:
            if ohlcvc_data and len(ohlcvc_data) >= 6:
                new_row_data = {
                    'timestamp': ohlcvc_data[0],
                    'open': float(ohlcvc_data[1]),
                    'high': float(ohlcvc_data[2]), 
                    'low': float(ohlcvc_data[3]),
                    'close': float(ohlcvc_data[4]),
                    'volume': float(ohlcvc_data[5])
                }
                
                # 格式化数据
                new_df = self.exchange_client.format_df([list(new_row_data.values())[:6]])
                
                if not new_df.empty:
                    # 获取当前缓存的数据
                    current_df = self.cache_ohlcv[cache_key]
                    
                    # 合并新数据
                    if not current_df.empty:
                        # 检查是否是同一时间戳的数据更新
                        new_timestamp = new_df.index[0]
                        if new_timestamp == current_df.index[-1]:
                            # 更新最后一行数据（同一时间戳的实时更新）
                            current_df.iloc[-1] = new_df.iloc[0]
                            # logger.debug(f"更新了 {cache_key} 的最后一行数据 (时间戳: {new_timestamp})")
                        else:
                            # 添加新的一行数据
                            combined_df = pd.concat([current_df, new_df])
                            # 保持最近500条数据
                            self.cache_ohlcv[cache_key] = combined_df.tail(500)
                            logger.debug(f"添加了 {cache_key} 的新数据行，当前共 {len(self.cache_ohlcv[cache_key])} 条")
                    else:
                        # 如果当前缓存为空，直接设置新数据
                        self.cache_ohlcv[cache_key] = new_df
                        logger.debug(f"初始化了 {cache_key} 的缓存数据")
                        
        except Exception as e:
            logger.error(f"更新缓存数据时发生错误: {e}")
            logger.debug(f"原始ohlcvc数据: {ohlcvc_data}")

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
        self._watch_task_ids = []  # 存储观测任务ID
        
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


