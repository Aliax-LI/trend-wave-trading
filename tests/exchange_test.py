import asyncio
from loguru import logger
from app.core.exchange_client import ExchangeClient


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
    """优化后的交易示例"""
    exchange_client = ExchangeClient("okx", config={

    })

    try:
        symbol = "BTC-USDT-SWAP"

        # 获取账户余额
        balance = await exchange_client.get_account_balance("USDT")
        if balance:
            logger.info(f"账户USDT余额: {balance}")

        # 获取资金费率
        funding_rate = await exchange_client.get_funding_rate(symbol)
        if funding_rate:
            logger.info(f"当前资金费率: {funding_rate * 100:.4f}%")

        # 获取最大杠杆
        max_leverage = await exchange_client.get_max_leverage(symbol)
        if max_leverage:
            logger.info(f"最大杠杆: {max_leverage}x")

        # 获取当前持仓
        position = await exchange_client.fetch_position(symbol)
        if position:
            logger.info(f"当前持仓: {position}")

        # 获取未成交订单
        open_orders = await exchange_client.get_open_orders(symbol)
        if open_orders:
            logger.info(f"未成交订单数量: {len(open_orders)}")

        # 创建限价单 (注意：这是真实交易，请确保参数正确)
        # 参数说明: symbol, price, amount(USDT), side, stop_loss, take_profit
        logger.info("准备创建限价单...")

        # 示例参数 - 请根据实际市场情况调整
        order_price = 104534.8  # 限价单价格
        order_amount = 15.0  # 交易金额 (USDT)
        stop_loss_price = 95000.0  # 止损价格
        take_profit_price = 130000.0  # 止盈价格

        # 验证价格逻辑
        if stop_loss_price >= order_price or take_profit_price <= order_price:
            logger.error("价格参数不合理，跳过订单创建")
        else:
            resp_data = await exchange_client.create_limit_order(
                symbol=symbol,
                price=order_price,
                amount=order_amount,
                side='buy',
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                leverage=None  # 使用最大杠杆
            )

            if resp_data:
                logger.info("订单创建成功！")
                print(f"订单响应: {resp_data}")
            else:
                logger.error("订单创建失败")

    except Exception as e:
        logger.error(f"交易示例执行失败: {e}")
    finally:
        await exchange_client.close_exchange()


if __name__ == '__main__':
    asyncio.run(main2())