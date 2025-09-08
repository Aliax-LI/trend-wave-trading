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
        self.max_limit = 100 # API最大限制为100
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

    # ================================ 公共操作 ================================

    async def get_ohlcv_data(self, symbol, timeframe='15m', limit=100) -> pd.DataFrame:
        """获取OHLCV数据
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            limit: 获取的K线数量
            
        Returns:
            pd.DataFrame: 格式化后的OHLCV数据
        """
        # 检查缓存
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if hasattr(self, '_ohlcv_cache') and cache_key in self._ohlcv_cache:
            cache_entry = self._ohlcv_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < 30:
                return cache_entry['data']
        
        all_ohlcv_data = []
        
        if limit <= self.max_limit:
            # 单次获取
            ohlcv_data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            all_ohlcv_data = ohlcv_data
        else:
            # 分批获取
            remaining = limit
            since = None
            
            while remaining > 0:
                current_limit = min(remaining, self.max_limit)
                batch_data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=current_limit, since=since)
                
                if not batch_data:
                    break
                
                all_ohlcv_data.extend(batch_data)
                remaining -= len(batch_data)
                
                if len(batch_data) < current_limit:
                    break
                
                # 计算下次请求的since参数
                if len(batch_data) >= 2:
                    time_interval = batch_data[1][0] - batch_data[0][0]
                    since = batch_data[0][0] - time_interval
                else:
                    break
                
                await asyncio.sleep(0.1)
        
        result_df = self.format_df(all_ohlcv_data)
        
        # 更新缓存
        if not hasattr(self, '_ohlcv_cache'):
            self._ohlcv_cache = {}
        self._ohlcv_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': result_df
        }
        
        return result_df

    async def fetch_ticker(self, symbol, key="last"):
        ticker = await self.exchange.fetch_ticker(symbol)
        return getattr(ticker, key)

    async def fetch_historical_data(self, symbol, timeframe='15m', days=1) -> pd.DataFrame:
        """获取历史OHLCV数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            days: 获取多少天的历史数据
            
        Returns:
            pd.DataFrame: 格式化后的历史OHLCV数据
        """
        # 检查缓存
        cache_key = f"{symbol}_{timeframe}_{days}_historical"
        if hasattr(self, '_historical_cache') and cache_key in self._historical_cache:
            cache_entry = self._historical_cache[cache_key]
            # 如果缓存时间在5分钟内，直接返回缓存数据
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < 300:
                logger.info(f"使用缓存的历史数据: {symbol} {timeframe} {days}天")
                return cache_entry['data']
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_ohlcv_data = []
        current_since = start_ts
        
        while current_since < end_ts:
            batch_data = await self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=self.max_limit)
            
            if not batch_data:
                break
            
            valid_data = [d for d in batch_data if d[0] <= end_ts]
            all_ohlcv_data.extend(valid_data)
            
            if len(batch_data) < self.max_limit or batch_data[-1][0] >= end_ts:
                break
            
            current_since = batch_data[-1][0] + 1
            await asyncio.sleep(0.1)
        
        result_df = self.format_df(all_ohlcv_data)
        
        # 更新缓存
        if not hasattr(self, '_historical_cache'):
            self._historical_cache = {}
        self._historical_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': result_df
        }
        
        logger.info(f"成功获取历史数据: {len(result_df)} 条 ({symbol} {timeframe} {days}天)")
        return result_df

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
    
    async def stop_watch_ohlcvc(self, symbol: str, timeframe: str = '15m') -> bool:
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
        await self.exchange.un_watch_trades(symbol)
        logger.info(f"停止观测任务: {task_key}")
        return True
    
    async def stop_all_watch_tasks(self):
        """停止所有观测任务"""
        task_keys = list(self._watch_tasks.keys())
        for task_key in task_keys:
            symbol, timeframe = task_key.split('_', 1)
            await self.stop_watch_ohlcvc(symbol, timeframe)
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

    def clear_cache(self, cache_type=None):
        """清除数据缓存
        
        Args:
            cache_type: 要清除的缓存类型，可以是 'ohlcv'、'historical' 或 None (清除所有)
        """
        if cache_type is None or cache_type == 'ohlcv':
            if hasattr(self, '_ohlcv_cache'):
                self._ohlcv_cache = {}
                logger.info("已清除OHLCV数据缓存")
                
        if cache_type is None or cache_type == 'historical':
            if hasattr(self, '_historical_cache'):
                self._historical_cache = {}
                logger.info("已清除历史数据缓存")
    
    async def close_exchange(self):
        """关闭交易所连接并清理所有观测任务"""
        # 先停止所有观测任务
        await self.stop_all_watch_tasks()
        # 清除缓存
        self.clear_cache()
        # 关闭交易所连接
        await self.exchange.close()
        logger.info("已关闭交易所连接")


    # ================================ 私有操作 ================================

    async def convert_contract_coin(self, convert_type: int, symbol: str, amount: str, price: str,
                                    op_type : str ="open"):
        await self.exchange.load_markets()
        formatted_amount = self.exchange.amount_to_precision(symbol, amount)
        formatted_price = self.exchange.price_to_precision(symbol, price)
        print(formatted_amount, formatted_price)
        # 币张转换
        resp_data = await self.exchange.publicGetPublicConvertContractCoin(
            {
                "type": convert_type,
                "instId": symbol,
                "sz": amount,
                "px": price,
                'opType': op_type,

            }
        )
        return resp_data.get('data')[0].get('sz')

    async def get_max_leverage(self, symbol: str):
        await self.exchange.load_markets()
        symbol_market = self.exchange.market(symbol)
        return symbol_market['limits']['leverage']['max']


    async def set_leverage(self, symbol: str, leverage = None):
        if leverage is None:
            leverage = await self.get_max_leverage(symbol)
        leverage_resp = await self.exchange.set_leverage(leverage, symbol=symbol)
        logger.info(f"🔧 设置杠杆倍数: {leverage}x, 响应：: {leverage_resp}")

    async def create_limit_order(self, symbol: str, price: float, amount: float, signal_type, take_profit, stop_loss):
        pos_side = "long" if signal_type == "buy" else "short"
        # 订单参数
        order_params = {
            'tdMode': 'isolated',  # 逐仓保证金模式
            'posSide': pos_side,  # 持仓方向
            'attachAlgoOrds': [  # 附加止盈止损算法订单
                {
                    'attachAlgoClOrdId': "first_tp_trigger",  # 止盈止损策略ID
                    'tpTriggerPx': take_profit,  # 第一止盈触发价
                    'tpOrdPx': -1,  # 止盈委托价(-1表示市价)
                },
                {
                    'attachAlgoClOrdId': "auto_tp_sl_trigger", # 止盈止损策略ID
                    'tpTriggerPx': take_profit,  # 第二止盈触发价
                    'tpOrdPx': -1,  # 止盈委托价(-1表示市价)
                    'slTriggerPx': stop_loss,  # 止损触发价
                    'slOrdPx': -1  # 止损委托价(-1表示市价)
                }
            ]
        }
        # 创建带止盈止损的限价单
        order_response = await self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=signal_type,
            amount=amount,
            price=price,
            params=order_params
        )
        logger.info(f"🔧 创建带止盈止损的限价单，响应: {order_response}")

    async def modify_tp_sl(self, symbol: str, take_profit, stop_loss, algo_order_id: str= "auto_tp_sl_trigger"):
        """
        修改止盈止损价
        """
        modify_params = {
            'instId': symbol,
            'algoClOrdId': algo_order_id,
            'newTpTriggerPx': take_profit,  # 新止盈触发价
            'newTpOrdPx': -1,  # 市价止盈
            'newSlTriggerPx': stop_loss,  # 新止损触发价
            'newSlOrdPx': -1,  # 市价止损
        }
        # 调用OKX的修改算法订单API
        modify_response = await self.exchange.privatePostTradeAmendAlgos(modify_params)
        logger.info(f"🔧 修改止盈止损价，响应: {modify_response}")

    async def get_funding_rate(self, symbol):
        rate_response = await self.exchange.fetch_funding_rate(symbol)
        funding_rate = rate_response.get('info', {}).get('fundingRate')

        if funding_rate:
            funding_rate_percent = float(funding_rate) * 100
            logger.info(f"📊 {symbol} 资金费率: {funding_rate_percent:.4f}%")
            return float(funding_rate)
        else:
            logger.warning(f"⚠️ 无法获取 {symbol} 的资金费率")
            return None




async def data_callback(symbol: str, ohlcvc_data):
    """示例回调函数"""
    logger.info(f"收到 {symbol} 的新数据: {ohlcvc_data}")

async def main():
    """优化后的使用示例"""
    exchange_client = ExchangeClient("okx")
    
    try:
        # 获取OHLCV数据
        ohlcv_data = await exchange_client.get_ohlcv_data(
            symbol='BTC/USDT:USDT', 
            timeframe='15m', 
            limit=450  # 最大限制为100
        )
        logger.info(f"获取到 {len(ohlcv_data)} 条OHLCV数据")
        # 获取历史数据
        historical_data = await exchange_client.fetch_historical_data(
            symbol='BTC/USDT:USDT', 
            timeframe='15m', 
            days=1
        )
        logger.info(f"获取到 {len(historical_data)} 条历史数据")
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
        await asyncio.sleep(30)
        
        # 停止特定任务
        await exchange_client.stop_watch_ohlcvc('BTC/USDT:USDT', '15m')
        # 再运行一段时间
        await asyncio.sleep(15)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        # 停止所有任务并关闭连接
        await exchange_client.close_exchange()

async def main2():
    exchange_client = ExchangeClient("okx")
    resp_data = await exchange_client.convert_contract_coin(1, 'BTC-USD-SWAP', "10", "112072.1")
    resp_data = exchange_client.get_max_leverage("BTC-USD-SWAP")
    print(resp_data)
    await exchange_client.close_exchange()

if __name__ == '__main__':
    asyncio.run(main2())

