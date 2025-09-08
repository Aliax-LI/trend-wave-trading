import asyncio
from datetime import datetime, timezone
import tzlocal
import ccxt.pro as ccxt
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

class ExchangeClient:

    def __init__(self, exchange_name: str, config=None):
        if config is None:
            config = {}
        self.local_tz = tzlocal.get_localzone()
        self.config: Dict[str, Any] = {
            "enableRateLimit": True,
            "newUpdates": True,
            "timeout": 6000,
            'httpProxy': config.get('httpProxy', "http://127.0.0.1:7890"),
            'options': config.get('options', {
                "defaultType": "swap",
                'OHLCVLimit': 10000,
            }),
        }
        self.exchange = self._init_exchange(exchange_name, config)
        self.cache_ohlcvc = {}
        self._watch_tasks = {}  # 存储观测任务
        self._stop_events = {}  # 存储停止事件

    def format_df(self, ohlcv_data) -> pd.DataFrame:
        """格式化OHLCV数据"""
        if not ohlcv_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(self.local_tz)
        df.set_index('datetime', inplace=True)

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=numeric_columns, inplace=True)

        return df

    def _init_exchange(self, exchange_name, config):
        if 'api_key' in config.keys():
            self.config['apiKey'] = config['api_key']
        if 'api_secret' in config.keys():
            self.config['apiSecret'] = config['api_secret']
        if 'passphrase' in config.keys():
            self.config['password'] = config['passphrase']
        logger.info(f"初始化交易所: {exchange_name}, 时区：{self.local_tz.key}")
        return getattr(ccxt, exchange_name)(self.config)

    async def get_ohlcv_data(self, symbol, timeframe='15m', limit=100) -> pd.DataFrame:
        batch_count, last_count = int(limit / 100), limit % 100
        since, time_interval = None, None
        all_ohlcvs = []
        for i in range(batch_count):
            ohlcv_data = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
            all_ohlcvs.extend(ohlcv_data)
            if time_interval is None:
                time_interval = int(ohlcv_data[1][0] - ohlcv_data[0][0])
            since = ohlcv_data[0][0] - time_interval * limit
        if last_count > 0:
            since = since - time_interval * last_count
            await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=last_count)
            all_ohlcvs.extend(ohlcv_data)
        return self.format_df(all_ohlcvs)

    async def fetch_historical_data(self, symbol, timeframe='15m', days=1) -> pd.DataFrame:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        current_since = start_ts
        all_ohlcv_data = []
        while current_since < end_ts:
            batch_data = await self.exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=300
            )
            if not batch_data:
                break
            all_ohlcv_data.extend(batch_data)
            logger.info(f"获取历史ohlcv：{len(all_ohlcv_data)} 条数据")
            current_since = batch_data[-1][0] + 1
            if batch_data[-1][0] > end_ts:
                break
            await asyncio.sleep(0.3)
        return self.format_df(all_ohlcv_data)

    async def _watch_ohlcvc_internal(self, symbol: str, timeframe: str = '15m', 
                                     callback: Optional[Callable] = None):
        """内部观测方法"""
        task_key = f"{symbol}_{timeframe}"
        stop_event = self._stop_events.get(task_key)
        
        try:
            while not stop_event.is_set():
                try:
                    # 使用超时避免无限等待
                    trades = await asyncio.wait_for(
                        self.exchange.watch_trades(symbol), 
                        timeout=30.0
                    )
                    # 获取最新数据，格式为：['timestamp', 'open', 'high', 'low', 'close', 'volume', 'count']
                    ohlcvc_data = self.exchange.build_ohlcvc(trades, timeframe)[0]
                    logger.debug(f"最新ohlcvc数据：{ohlcvc_data}")
                    self.cache_ohlcvc[symbol] = ohlcvc_data
                    
                    # 如果有回调函数，调用它
                    if callback:
                        await callback(symbol, ohlcvc_data)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"观测 {symbol} 超时，重新尝试...")
                    continue
                except Exception as e:
                    logger.error(f"观测 {symbol} 时发生错误: {e}")
                    await asyncio.sleep(5)  # 出错时等待5秒后重试
                    
        except asyncio.CancelledError:
            logger.info(f"观测任务 {task_key} 被取消")
            raise
        finally:
            # 清理资源
            if task_key in self._watch_tasks:
                del self._watch_tasks[task_key]
            if task_key in self._stop_events:
                del self._stop_events[task_key]

    def start_watch_ohlcvc(self, symbol: str, timeframe: str = '15m', 
                          callback: Optional[Callable] = None) -> str:
        """启动观测OHLCVC数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            callback: 可选的回调函数，接收(symbol, ohlcvc_data)参数
            
        Returns:
            str: 任务标识符
        """
        task_key = f"{symbol}_{timeframe}"
        
        # 如果任务已存在，先停止它
        if task_key in self._watch_tasks:
            logger.warning(f"任务 {task_key} 已存在，将先停止现有任务")
            self.stop_watch_ohlcvc(symbol, timeframe)
        
        # 创建停止事件
        stop_event = asyncio.Event()
        self._stop_events[task_key] = stop_event
        
        # 创建并启动任务
        task = asyncio.create_task(
            self._watch_ohlcvc_internal(symbol, timeframe, callback)
        )
        self._watch_tasks[task_key] = task
        
        logger.info(f"启动观测任务: {task_key}")
        return task_key
    
    def stop_watch_ohlcvc(self, symbol: str, timeframe: str = '15m') -> bool:
        """停止观测OHLCVC数据
        
        Args:
            symbol: 交易对符号  
            timeframe: 时间框架
            
        Returns:
            bool: 是否成功停止任务
        """
        task_key = f"{symbol}_{timeframe}"
        
        if task_key not in self._watch_tasks:
            logger.warning(f"任务 {task_key} 不存在")
            return False
        
        # 设置停止事件
        if task_key in self._stop_events:
            self._stop_events[task_key].set()
        
        # 取消任务
        task = self._watch_tasks[task_key]
        task.cancel()
        
        logger.info(f"停止观测任务: {task_key}")
        return True
    
    def stop_all_watch_tasks(self):
        """停止所有观测任务"""
        task_keys = list(self._watch_tasks.keys())
        for task_key in task_keys:
            symbol, timeframe = task_key.split('_', 1)
            self.stop_watch_ohlcvc(symbol, timeframe)
        logger.info("已停止所有观测任务")
    
    def get_watch_status(self) -> Dict[str, bool]:
        """获取观测任务状态
        
        Returns:
            Dict[str, bool]: 任务标识符到运行状态的映射
        """
        status = {}
        for task_key, task in self._watch_tasks.items():
            status[task_key] = not task.done()
        return status
    
    async def watch_ohlcvc(self, symbol: str, timeframe: str = '15m', 
                          callback: Optional[Callable] = None):
        """兼容性方法：直接观测OHLCVC数据（阻塞式）
        
        注意：这是阻塞式方法，建议使用 start_watch_ohlcvc 进行非阻塞观测
        """
        logger.warning("使用了阻塞式观测方法，建议使用 start_watch_ohlcvc")
        await self._watch_ohlcvc_internal(symbol, timeframe, callback)

    def close_exchange(self):
        """关闭交易所连接并清理所有观测任务"""
        # 先停止所有观测任务
        self.stop_all_watch_tasks()
        # 关闭交易所连接
        self.exchange.close()



async def data_callback(symbol: str, ohlcvc_data):
    """示例回调函数"""
    logger.info(f"收到 {symbol} 的新数据: {ohlcvc_data}")

async def main():
    """优化后的使用示例"""
    exchange_client = ExchangeClient("okx")
    
    try:
        # 启动观测任务（非阻塞）
        task_id1 = exchange_client.start_watch_ohlcvc(
            symbol='BTC/USDT:USDT', 
            timeframe='15m', 
            callback=data_callback
        )
        
        task_id2 = exchange_client.start_watch_ohlcvc(
            symbol='ETH/USDT:USDT', 
            timeframe='5m'
        )
        
        logger.info(f"启动了观测任务: {task_id1}, {task_id2}")
        
        # 查看任务状态
        status = exchange_client.get_watch_status()
        logger.info(f"任务状态: {status}")
        
        # 运行一段时间
        await asyncio.sleep(60)
        
        # 停止特定任务
        exchange_client.stop_watch_ohlcvc('BTC/USDT:USDT', '15m')
        
        # 再运行一段时间
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        # 停止所有任务并关闭连接
        exchange_client.close_exchange()

if __name__ == '__main__':
    asyncio.run(main())

