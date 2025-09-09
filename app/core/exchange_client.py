import asyncio
from datetime import datetime, timezone
import tzlocal
import ccxt.pro as ccxt
from typing import Dict, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
            self.config['secret'] = config['api_secret']
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
                    
                    if trades and len(trades) > 0:
                        # 构建OHLCVC数据，格式为：[timestamp, open, high, low, close, volume, count]
                        ohlcvc_list = self.exchange.build_ohlcvc(trades, timeframe)
                        
                        if ohlcvc_list and len(ohlcvc_list) > 0:
                            # 获取最新的OHLCVC数据
                            ohlcvc_data = ohlcvc_list[-1]  # 取最后一条数据
                            # logger.debug(f"最新ohlcvc数据：{ohlcvc_data}")
                            self.cache_ohlcvc[symbol] = ohlcvc_data
                            
                            # 如果有回调函数，调用它
                            if callback:
                                await callback(symbol, timeframe, ohlcvc_data)
                        else:
                            logger.debug(f"没有构建到有效的OHLCVC数据: {symbol}")
                    else:
                        logger.debug(f"没有接收到有效的交易数据: {symbol}")
                        
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

    def _validate_trading_params(self, symbol: str, price: float, amount: float, 
                                side: str) -> Tuple[bool, str]:
        """验证交易参数
        
        Args:
            symbol: 交易对符号
            price: 价格
            amount: 数量
            side: 交易方向
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not symbol or not isinstance(symbol, str):
            return False, "交易对符号不能为空且必须是字符串"
            
        if price <= 0:
            return False, "价格必须大于0"
            
        if amount <= 0:
            return False, "数量必须大于0"
            
        if side not in ['buy', 'sell', 'long', 'short']:
            return False, "交易方向必须是 'buy', 'sell', 'long', 'short' 之一"
            
        return True, ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def convert_contract_coin(self, convert_type: int, symbol: str, amount: Union[str, float], 
                                   price: Union[str, float], op_type: str = "open") -> Optional[str]:
        """币张转换
        
        Args:
            convert_type: 转换类型 (1: 币转张, 2: 张转币)
            symbol: 交易对符号
            amount: 数量
            price: 价格
            op_type: 操作类型 (open/close)
            
        Returns:
            Optional[str]: 转换后的数量，失败返回None
        """
        # 参数验证
        if convert_type not in [1, 2]:
            raise ValueError("转换类型必须是1(币转张)或2(张转币)")
            
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        # 转换为字符串格式
        amount_str = str(amount) if isinstance(amount, (int, float)) else amount
        price_str = str(price) if isinstance(price, (int, float)) else price
        
        # 验证数值
        try:
            float(amount_str)
            float(price_str)
        except ValueError:
            raise ValueError("金额和价格必须是有效数字")
            
        try:
            resp_data = await self.exchange.publicGetPublicConvertContractCoin({
                "type": convert_type,
                "instId": symbol,
                "sz": amount_str,
                "px": price_str,
                'opType': op_type,
            })
            
            if not resp_data or not resp_data.get('data'):
                raise ValueError("币张转换失败: 无有效响应数据")
                
            result = resp_data.get('data')[0].get('sz')
            logger.debug(f"币张转换成功: {amount_str} -> {result}")
            return result
            
        except (ValueError, TypeError) as e:
            logger.error(f"币张转换失败: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def get_max_leverage(self, symbol: str) -> Optional[float]:
        """获取最大杠杆倍数
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[float]: 最大杠杆倍数，失败返回None
        """
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        try:
            await self.exchange.load_markets()
            symbol_market = self.exchange.market(symbol)
            
            if not symbol_market:
                raise ValueError(f"找不到交易对市场信息: {symbol}")
                
            max_leverage = symbol_market.get('limits', {}).get('leverage', {}).get('max')
            
            if max_leverage is None:
                raise ValueError(f"无法获取 {symbol} 的最大杠杆信息")
                
            logger.debug(f"{symbol} 最大杠杆倍数: {max_leverage}x")
            return float(max_leverage)
            
        except ValueError as e:
            logger.error(f"获取最大杠杆失败 {symbol}: {e}")
            return None


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def set_leverage(self, symbol: str, leverage: Optional[float] = None) -> Optional[float]:
        """设置杠杆倍数
        
        Args:
            symbol: 交易对符号
            leverage: 杠杆倍数，None时使用最大杠杆
            
        Returns:
            Optional[float]: 实际设置的杠杆倍数，失败返回None
        """
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        try:
            if leverage is None:
                leverage = await self.get_max_leverage(symbol)
                if leverage is None:
                    raise ValueError("无法获取最大杠杆倍数")
            else:
                # 验证杠杆倍数范围
                max_leverage = await self.get_max_leverage(symbol)
                if max_leverage and leverage > max_leverage:
                    logger.warning(f"请求的杠杆倍数 {leverage}x 超过最大值 {max_leverage}x，使用最大值")
                    leverage = max_leverage
                    
            leverage_resp = await self.exchange.set_leverage(leverage, symbol=symbol)
            logger.info(f"🔧 设置杠杆倍数: {leverage}x, 响应: {leverage_resp}")
            return leverage
                        
        except ValueError as e:
            logger.error(f"设置杠杆失败 {symbol}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def fetch_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """查看持仓信息
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[Dict[str, Any]]: 持仓信息，失败返回None
        """
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        try:
            position_data = await self.exchange.fetch_position(symbol)
            logger.debug(f"获取持仓信息成功: {symbol}")
            return position_data
                        
        except ValueError as e:
            logger.error(f"获取持仓信息失败 {symbol}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def create_limit_order(self, symbol: str, price: float, amount: float, side: str, 
                                stop_loss: float, take_profit: float, leverage: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """创建带止盈止损的限价单
        
        Args:
            symbol: 交易对符号
            price: 限价单价格
            amount: 交易金额(USDT)
            side: 交易方向 ('buy'/'sell')
            stop_loss: 止损价格
            take_profit: 止盈价格
            leverage: 杠杆倍数，None时使用最大杠杆
            
        Returns:
            Optional[Dict[str, Any]]: 订单响应信息，失败返回None
        """
        # 参数验证
        is_valid, error_msg = self._validate_trading_params(symbol, price, amount, side)
        if not is_valid:
            raise ValueError(error_msg)
            
        if stop_loss <= 0 or take_profit <= 0:
            raise ValueError("止损价和止盈价必须大于0")
            
        # 验证止盈止损价格逻辑
        if side == "buy":
            if stop_loss >= price:
                raise ValueError("买入订单的止损价必须小于限价单价格")
            if take_profit <= price:
                raise ValueError("买入订单的止盈价必须大于限价单价格")
        else:  # sell
            if stop_loss <= price:
                raise ValueError("卖出订单的止损价必须大于限价单价格")
            if take_profit >= price:
                raise ValueError("卖出订单的止盈价必须小于限价单价格")
        
        try:
            await self.exchange.load_markets()
            
            # 设置杠杆
            actual_leverage = await self.set_leverage(symbol, leverage)
            if actual_leverage is None:
                raise ValueError("设置杠杆失败")
            
            # 计算购买币数
            symbol_amount = round(actual_leverage * amount / price, 8)
            formatted_price = self.exchange.price_to_precision(symbol, price)
            
            # 币张转换
            contracts = await self.convert_contract_coin(1, symbol, str(symbol_amount), str(formatted_price))
            if contracts is None:
                raise ValueError("币张转换失败")
                
            pos_side = "long" if side == "buy" else "short"
            algo_order_id = f"ATS{int(time.time() * 1000)}"
            # 创建订单参数
            order_params = {
                'tdMode': 'cross',  # 全仓保证金模式
                'posSide': pos_side,  # 持仓方向
                'attachAlgoOrds': [{  # 附加止盈止损算法订单
                    'attachAlgoClOrdId': algo_order_id, # 唯一止盈止损策略ID
                    'tpTriggerPx': self.exchange.price_to_precision(symbol, take_profit),  # 止盈触发价
                    'tpOrdPx': -1,  # 止盈委托价(-1表示市价)
                    'slTriggerPx': self.exchange.price_to_precision(symbol, stop_loss),  # 止损触发价
                    'slOrdPx': -1  # 止损委托价(-1表示市价)
                }]
            }
            
            order_response = await self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=float(contracts),
                price=float(formatted_price),
                params=order_params
            )
            order_id = order_response['id']
            logger.info(f"🔧 创建带止盈止损的限价单成功:")
            logger.info(f"交易对: {symbol}")
            logger.info(f"方向: {side} ({pos_side})")
            logger.info(f"价格: {formatted_price}")
            logger.info(f"合约数量: {contracts}")
            logger.info(f"杠杆: {actual_leverage}x")
            logger.info(f"止损: {stop_loss}")
            logger.info(f"止盈: {take_profit}")
            logger.info(f"订单ID: {order_id}, 止盈止损单ID: {algo_order_id}")
            logger.debug(f"   订单响应: {order_response}")
            
            return order_id, algo_order_id
                        
        except ValueError as e:
            logger.error(f"创建限价单失败 {symbol}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def modify_tp_sl(self, symbol: str, take_profit: float, stop_loss: float,
                          algo_order_id: str = "auto_tp_sl_trigger") -> Optional[Dict[str, Any]]:
        """修改止盈止损价
        
        Args:
            symbol: 交易对符号
            take_profit: 新的止盈价格
            stop_loss: 新的止损价格
            algo_order_id: 算法订单ID
            
        Returns:
            Optional[Dict[str, Any]]: 修改响应信息，失败返回None
        """
        # 参数验证
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        if take_profit <= 0 or stop_loss <= 0:
            raise ValueError("止盈价和止损价必须大于0")
            
        if not algo_order_id:
            raise ValueError("算法订单ID不能为空")
            
        try:
            await self.exchange.load_markets()
            
            modify_params = {
                'instId': symbol,
                'algoClOrdId': algo_order_id,
                'newTpTriggerPx': self.exchange.price_to_precision(symbol, take_profit),  # 新止盈触发价
                'newTpOrdPx': -1,  # 市价止盈
                'newSlTriggerPx': self.exchange.price_to_precision(symbol, stop_loss),  # 新止损触发价
                'newSlOrdPx': -1,  # 市价止损
            }
            
            modify_response = await self.exchange.privatePostTradeAmendAlgos(modify_params)
            
            if modify_response and modify_response.get('code') == '0':
                logger.info(f"🔧 修改止盈止损价成功:")
                logger.info(f"   交易对: {symbol}")
                logger.info(f"   算法订单ID: {algo_order_id}")
                logger.info(f"   新止盈价: {take_profit}")
                logger.info(f"   新止损价: {stop_loss}")
                logger.debug(f"   响应: {modify_response}")
                return modify_response
            else:
                error_msg = modify_response.get('msg', '未知错误') if modify_response else '无响应数据'
                raise ValueError(f"修改失败: {error_msg}")
                        
        except ValueError as e:
            logger.error(f"修改止盈止损价失败 {symbol}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """获取资金费率
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Optional[float]: 资金费率(小数形式)，失败返回None
        """
        if not symbol:
            raise ValueError("交易对符号不能为空")
            
        try:
            rate_response = await self.exchange.fetch_funding_rate(symbol)
            
            if not rate_response:
                raise ValueError("获取资金费率响应为空")
                
            # 尝试多种可能的数据路径
            funding_rate = None
            
            # 路径1: info.fundingRate
            if rate_response.get('info', {}).get('fundingRate'):
                funding_rate = rate_response['info']['fundingRate']
            # 路径2: fundingRate
            elif rate_response.get('fundingRate'):
                funding_rate = rate_response['fundingRate']
            # 路径3: percentage (某些交易所可能返回百分比形式)
            elif rate_response.get('percentage'):
                funding_rate = rate_response['percentage'] / 100
                
            if funding_rate is not None:
                try:
                    funding_rate_float = float(funding_rate)
                    funding_rate_percent = funding_rate_float * 100
                    
                    logger.info(f"📊 {symbol} 资金费率: {funding_rate_percent:.4f}%")
                    return funding_rate_float
                    
                except (ValueError, TypeError):
                    raise ValueError(f"无法转换资金费率为数字: {funding_rate}")
            else:
                raise ValueError("响应中未找到资金费率数据")
                        
        except ValueError as e:
            logger.warning(f"⚠️ 无法获取 {symbol} 的资金费率: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对符号
            
        Returns:
            bool: 是否成功取消
        """
        if not order_id or not symbol:
            raise ValueError("订单ID和交易对符号不能为空")
            
        try:
            cancel_response = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ 成功取消订单: {order_id} ({symbol})")
            return True
                        
        except ValueError as e:
            logger.error(f"取消订单失败 {order_id}: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def get_account_balance(self, currency: str = "USDT") -> Optional[Dict[str, float]]:
        """获取账户余额
        
        Args:
            currency: 货币类型
            
        Returns:
            Optional[Dict[str, float]]: 余额信息 {'free': 可用, 'used': 已用, 'total': 总计}
        """
        try:
            balance = await self.exchange.fetch_balance()
            
            if currency in balance:
                currency_balance = balance[currency]
                result = {
                    'free': float(currency_balance.get('free', 0)),
                    'used': float(currency_balance.get('used', 0)),
                    'total': float(currency_balance.get('total', 0))
                }
                logger.debug(f"账户余额 ({currency}): {result}")
                return result
            else:
                logger.warning(f"未找到 {currency} 的余额信息")
                return None
                        
        except ValueError as e:
            logger.error(f"获取账户余额失败: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout))
    )
    async def get_open_orders(self, symbol: Optional[str] = None) -> Optional[list]:
        """获取未成交订单
        
        Args:
            symbol: 交易对符号，None表示获取所有
            
        Returns:
            Optional[list]: 订单列表
        """
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            logger.debug(f"获取到 {len(orders)} 个未成交订单")
            return orders
                        
        except ValueError as e:
            logger.error(f"获取未成交订单失败: {e}")
            return None

