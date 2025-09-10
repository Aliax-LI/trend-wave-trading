"""
配置加载器
用于加载策略数据系统的配置
"""

import yaml
import os
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_file: str = "config/strategy_data_config.yaml"):
        self.config_file = config_file
        self.config = {}
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"配置文件不存在: {self.config_file}, 使用默认配置")
                return self.get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"✅ 配置文件加载成功: {self.config_file}")
            return self.config
            
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            logger.info("使用默认配置")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "exchange": {
                "name": "okx",
                "config": {
                    "api_key": "",
                    "api_secret": "",
                    "passphrase": "",
                    "sandbox": False,
                    "options": {
                        "defaultType": "swap",
                        "OHLCVLimit": 100
                    }
                }
            },
            "symbols": [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT"
            ],
            "timeframes": {
                "trend_filter": "1h",
                "signal_main": "15m",
                "entry_timing": "5m"
            },
            "data_loader": {
                "window_obs": 200,
                "max_cache_length": 500,
                "init_data_timeout": 60
            },
            "monitoring": {
                "funding_rate": {
                    "check_interval": 300,
                    "cache_duration": 300,
                    "high_threshold": 0.001,
                    "extreme_threshold": 0.002
                },
                "price_monitoring": {
                    "breakout_threshold": 0.005,
                    "volume_spike_ratio": 2.0,
                    "price_level_tolerance": 0.002
                },
                "system_health": {
                    "status_check_interval": 30,
                    "cleanup_interval": 3600,
                    "max_price_level_age": 24
                }
            },
            "strategy_callbacks": {
                "enable_trend_callback": True,
                "enable_pattern_callback": True,
                "enable_entry_callback": True
            },
            "logging": {
                "level": "INFO",
                "file_rotation": "1 day",
                "file_retention": "7 days",
                "enable_debug_logs": False
            },
            "risk_management": {
                "atr_period": 14,
                "min_data_requirements": {
                    "trend_analysis": 144,
                    "pattern_analysis": 200,
                    "entry_signal": 50
                }
            },
            "performance": {
                "enable_data_cache": True,
                "cache_cleanup_interval": 3600,
                "max_concurrent_requests": 10,
                "request_timeout": 30,
                "memory_limit_mb": 500,
                "auto_cleanup_enabled": True
            }
        }
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """获取交易所配置"""
        return self.config.get("exchange", {})
    
    def get_symbols(self) -> list:
        """获取监控币种列表"""
        return self.config.get("symbols", ["BTC/USDT:USDT"])
    
    def get_timeframes(self) -> Dict[str, str]:
        """获取时间周期配置"""
        return self.config.get("timeframes", {
            "trend_filter": "1h",
            "signal_main": "15m", 
            "entry_timing": "5m"
        })
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """获取数据加载器配置"""
        return self.config.get("data_loader", {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.config.get("monitoring", {})
    
    def get_strategy_callbacks_config(self) -> Dict[str, bool]:
        """获取策略回调配置"""
        return self.config.get("strategy_callbacks", {})
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """获取风险管理配置"""
        return self.config.get("risk_management", {})
    
    def save_config(self, config: Dict[str, Any] = None):
        """保存配置到文件"""
        try:
            config_to_save = config or self.config
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            logger.info(f"✅ 配置已保存到: {self.config_file}")
            
        except Exception as e:
            logger.error(f"❌ 配置保存失败: {e}")


def load_strategy_config(config_file: str = None) -> Dict[str, Any]:
    """快速加载策略配置"""
    if config_file is None:
        config_file = "config/strategy_data_config.yaml"
    
    loader = ConfigLoader(config_file)
    return loader.load_config()


if __name__ == "__main__":
    # 测试配置加载
    config = load_strategy_config()
    print("配置加载测试:")
    print(f"  交易所: {config['exchange']['name']}")
    print(f"  币种数量: {len(config['symbols'])}")
    print(f"  时间周期: {config['timeframes']}")
    print("✅ 配置加载测试完成")
