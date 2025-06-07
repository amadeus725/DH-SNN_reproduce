# SpikingJelly DH-SNN 配置文件
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from spikingjelly.activation_based import surrogate


@dataclass
class NetworkConfig:
    """网络配置类"""
    # 基本架构参数
    input_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [200])
    output_dim: int = 1
    
    # 树突参数
    use_dendritic: bool = True
    num_branches: int = 4
    test_sparsity: bool = False
    sparsity: float = 0.5
    mask_share: int = 1
    
    # 神经元参数
    v_threshold: float = 0.5
    tau_m_init: Tuple[float, float] = (0.0, 4.0)
    tau_n_init: Tuple[float, float] = (0.0, 4.0)
    tau_initializer: str = 'uniform'  # 'uniform' or 'constant'
    reset_mode: str = 'soft'  # 'soft', 'hard', 'none'
    
    # 替代函数参数
    surrogate_type: str = 'multi_gaussian'  # 'multi_gaussian', 'atan', 'sigmoid'
    surrogate_alpha: float = 0.5
    surrogate_sigma: float = 0.5
    
    # 训练参数
    step_mode: str = 's'  # 's' for single step, 'm' for multi step
    bias: bool = True
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass 
class TrainingConfig:
    """训练配置类"""
    # 优化器参数
    learning_rate: float = 0.001
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    weight_decay: float = 0.0
    
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 32
    
    # 损失函数
    loss_function: str = 'mse'  # 'mse', 'cross_entropy', 'bce'
    
    # 学习率调度
    use_scheduler: bool = False
    scheduler_type: str = 'step'  # 'step', 'cosine', 'plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # 正则化
    gradient_clip: Optional[float] = None
    dropout_rate: float = 0.0
    
    # 验证
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class DataConfig:
    """数据配置类"""
    # 数据生成参数
    num_train_samples: int = 2000
    num_test_samples: int = 500
    sequence_length: int = 100
    delay: int = 50
    
    # 数据预处理
    normalize: bool = False
    add_noise: bool = False
    noise_std: float = 0.1


# 预定义配置
CONFIGS = {
    'dh_snn_default': NetworkConfig(
        input_dim=2,
        hidden_dims=[200],
        output_dim=1,
        use_dendritic=True,
        num_branches=4,
        v_threshold=0.5,
        tau_m_init=(0.0, 4.0),
        tau_n_init=(0.0, 4.0),
        surrogate_type='multi_gaussian'
    ),
    
    'dh_snn_large': NetworkConfig(
        input_dim=2,
        hidden_dims=[512, 256],
        output_dim=1,
        use_dendritic=True,
        num_branches=8,
        v_threshold=0.5,
        tau_m_init=(0.0, 4.0),
        tau_n_init=(0.0, 4.0),
        surrogate_type='multi_gaussian'
    ),
    
    'vanilla_snn': NetworkConfig(
        input_dim=2,
        hidden_dims=[200],
        output_dim=1,
        use_dendritic=False,
        v_threshold=0.5,
        tau_m_init=(0.0, 4.0),
        surrogate_type='multi_gaussian'
    ),
    
    'sparse_dh_snn': NetworkConfig(
        input_dim=2,
        hidden_dims=[200],
        output_dim=1,
        use_dendritic=True,
        num_branches=4,
        test_sparsity=True,
        sparsity=0.3,
        v_threshold=0.5,
        tau_m_init=(0.0, 4.0),
        tau_n_init=(0.0, 4.0),
        surrogate_type='multi_gaussian'
    ),
    
    'multi_branch_snn': NetworkConfig(
        input_dim=2,
        hidden_dims=[200],
        output_dim=1,
        use_dendritic=True,
        num_branches=16,
        v_threshold=0.5,
        tau_m_init=(0.0, 4.0),
        tau_n_init=(0.0, 4.0),
        surrogate_type='multi_gaussian'
    )
}

TRAINING_CONFIGS = {
    'fast_training': TrainingConfig(
        learning_rate=0.01,
        num_epochs=50,
        batch_size=64,
        optimizer='adam'
    ),
    
    'careful_training': TrainingConfig(
        learning_rate=0.001,
        num_epochs=200,
        batch_size=32,
        optimizer='adam',
        use_scheduler=True,
        scheduler_type='step',
        scheduler_params={'step_size': 50, 'gamma': 0.5},
        gradient_clip=1.0,
        early_stopping_patience=20
    ),
    
    'sgd_training': TrainingConfig(
        learning_rate=0.1,
        num_epochs=100,
        batch_size=32,
        optimizer='sgd',
        weight_decay=1e-4,
        use_scheduler=True,
        scheduler_type='cosine',
        gradient_clip=0.5
    )
}

DATA_CONFIGS = {
    'small_dataset': DataConfig(
        num_train_samples=1000,
        num_test_samples=200,
        sequence_length=50,
        delay=25
    ),
    
    'large_dataset': DataConfig(
        num_train_samples=5000,
        num_test_samples=1000,
        sequence_length=100,
        delay=50
    ),
    
    'long_sequence': DataConfig(
        num_train_samples=2000,
        num_test_samples=500,
        sequence_length=200,
        delay=100
    ),
    
    'noisy_dataset': DataConfig(
        num_train_samples=2000,
        num_test_samples=500,
        sequence_length=100,
        delay=50,
        add_noise=True,
        noise_std=0.05
    )
}


def get_surrogate_function(config: NetworkConfig):
    """根据配置获取替代函数"""
    if config.surrogate_type == 'multi_gaussian':
        from SNN_layers.spike_dense_spikingjelly import MultiGaussianSurrogate
        return MultiGaussianSurrogate(
            alpha=config.surrogate_alpha,
            sigma=config.surrogate_sigma
        )
    elif config.surrogate_type == 'atan':
        return surrogate.ATan(alpha=config.surrogate_alpha)
    elif config.surrogate_type == 'sigmoid':
        return surrogate.Sigmoid(alpha=config.surrogate_alpha)
    else:
        raise ValueError(f"Unknown surrogate type: {config.surrogate_type}")


def create_model_from_config(config: NetworkConfig):
    """从配置创建模型"""
    from SNN_layers.spike_dense_spikingjelly import DH_SNN_Network
    
    surrogate_fn = get_surrogate_function(config)
    
    # 这里需要修改DH_SNN_Network来接受更多参数
    model = DH_SNN_Network(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
        num_branches=config.num_branches,
        use_dendritic=config.use_dendritic,
        step_mode=config.step_mode
    )
    
    return model.to(config.device)


def create_optimizer(model, config: TrainingConfig):
    """从配置创建优化器"""
    if config.optimizer.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    elif config.optimizer.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(optimizer, config: TrainingConfig):
    """从配置创建学习率调度器"""
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            **config.scheduler_params
        )
    elif config.scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_params.get('T_max', 100)
        )
    elif config.scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config.scheduler_params
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


def get_loss_function(config: TrainingConfig):
    """从配置获取损失函数"""
    if config.loss_function.lower() == 'mse':
        return torch.nn.MSELoss()
    elif config.loss_function.lower() == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif config.loss_function.lower() == 'bce':
        return torch.nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")


# 使用示例
def example_usage():
    """配置使用示例"""
    # 选择配置
    net_config = CONFIGS['dh_snn_default']
    train_config = TRAINING_CONFIGS['careful_training']
    data_config = DATA_CONFIGS['large_dataset']
    
    print("网络配置:")
    print(f"  架构: {'DH-SNN' if net_config.use_dendritic else 'Vanilla SNN'}")
    print(f"  隐藏层: {net_config.hidden_dims}")
    print(f"  树突分支数: {net_config.num_branches if net_config.use_dendritic else 'N/A'}")
    print(f"  阈值: {net_config.v_threshold}")
    
    print("\n训练配置:")
    print(f"  学习率: {train_config.learning_rate}")
    print(f"  训练轮数: {train_config.num_epochs}")
    print(f"  批次大小: {train_config.batch_size}")
    print(f"  优化器: {train_config.optimizer}")
    
    print("\n数据配置:")
    print(f"  训练样本: {data_config.num_train_samples}")
    print(f"  序列长度: {data_config.sequence_length}")
    print(f"  延迟: {data_config.delay}")
    
    return net_config, train_config, data_config


if __name__ == "__main__":
    example_usage()
