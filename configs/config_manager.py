"""
DH-SNN 配置管理器
自动根据实验类型选择合适的配置文件
"""

import importlib
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理器 - 根据实验类型自动加载配置"""
    
    # 实验类型映射
    EXPERIMENT_CONFIGS = {
        'delayed_xor': 'delayed_xor_config',
        'multiscale_xor': 'multiscale_xor_config',
        'speech': 'speech_config',
        'neurovpr': 'neurovpr_config',
        'sequential': 'sequential_config',
        'default': 'config'
    }
    
    def __init__(self):
        self.current_config = None
        
    def load_config(self, experiment_type: str) -> Dict[str, Any]:
        """加载指定实验类型的配置"""
        if experiment_type not in self.EXPERIMENT_CONFIGS:
            print(f"⚠️  未知实验类型: {experiment_type}, 使用默认配置")
            experiment_type = 'default'
            
        config_module_name = self.EXPERIMENT_CONFIGS[experiment_type]
        
        try:
            # 动态导入配置模块
            config_module = importlib.import_module(f'configs.{config_module_name}')
            
            # 提取所有配置
            config = {}
            for attr_name in dir(config_module):
                if not attr_name.startswith('_') and attr_name.isupper():
                    config[attr_name.lower()] = getattr(config_module, attr_name)
            
            self.current_config = config
            print(f"✅ 成功加载 {experiment_type} 实验配置")
            return config
            
        except ImportError as e:
            print(f"❌ 配置文件加载失败: {e}")
            print(f"🔄 回退到默认配置")
            return self.load_config('default')
    
    def get_model_config(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        """获取模型配置"""
        if not self.current_config:
            raise RuntimeError("请先调用 load_config() 加载配置")
            
        # 如果有数据集特定配置，优先使用
        if dataset and f"{dataset}_config" in self.current_config:
            return self.current_config[f"{dataset}_config"]
        elif 'model_config' in self.current_config:
            return self.current_config['model_config']
        else:
            raise KeyError("未找到模型配置")
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        if not self.current_config:
            raise RuntimeError("请先调用 load_config() 加载配置")
        return self.current_config.get('training_config', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        if not self.current_config:
            raise RuntimeError("请先调用 load_config() 加载配置")
        return self.current_config.get('data_config', {})
    
    def get_device(self):
        """获取设备配置"""
        if not self.current_config:
            raise RuntimeError("请先调用 load_config() 加载配置")
        return self.current_config.get('device')
    
    def print_config_summary(self):
        """打印配置摘要"""
        if not self.current_config:
            print("❌ 未加载任何配置")
            return
            
        print("\n" + "="*50)
        print("📋 当前配置摘要")
        print("="*50)
        
        for key, value in self.current_config.items():
            if isinstance(value, dict):
                print(f"\n🔧 {key.upper()}:")
                for sub_key, sub_value in value.items():
                    print(f"   {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        print("="*50)

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config(experiment_type: str) -> Dict[str, Any]:
    """便捷函数：获取指定实验类型的配置"""
    return config_manager.load_config(experiment_type)

def get_available_experiments():
    """获取所有可用的实验类型"""
    return list(ConfigManager.EXPERIMENT_CONFIGS.keys())