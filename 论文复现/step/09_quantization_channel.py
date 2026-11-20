"""Step09：先实现范围可配置的均匀量化基线。"""

import torch
import torch.nn as nn

import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 复用已有的整包丢失模拟函数。
sys.path.insert(0, str(PROJECT_ROOT / "packet"))

from packet_loss import simulate_packet_loss
# 复用Step08的特征分解模块。
step08 = importlib.import_module("08_feature_decomposition")


class UniformQuantizer(nn.Module):
    """
    固定范围的均匀量化器。

    输入、输出形状不变。

    训练模式：
        裁剪特征后加入均匀噪声，近似量化误差。

    评估模式：
        映射、取整并恢复数值，输出有限个离散级别。

    当前输出仍是浮点张量，供神经网络处理。
    后续再实现码值的实际比特打包。
    """

    def __init__(self, bits=3, min_value=-1.0, max_value=1.0):
        super().__init__()

        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits必须为正整数")

        if min_value >= max_value:
            raise ValueError("min_value必须小于max_value")

        self.bits = bits
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        # 3-bit有8个级别，对应码值0～7。
        self.num_levels = 2 ** bits

        # 相邻重建级别之间的距离。
        self.step_size = (
            (self.max_value - self.min_value)
            / (self.num_levels - 1)
        )

    def forward(self, features):
        if not features.is_floating_point():
            raise TypeError("输入必须是浮点张量")

        # 超出表示范围的数值会饱和到端点。
        clipped = features.clamp(
            min=self.min_value,
            max=self.max_value,
        )

        if self.training:
            # rand_like生成[0,1)范围的随机数。
            # 平移并缩放后，噪声位于[-Δ/2,Δ/2)。
            noise = (
                torch.rand_like(clipped) - 0.5
            ) * self.step_size

            # 不进行取整，使范围内部的输入保留梯度。
            # 加噪后的值可能略超出量化范围，这是本基线的噪声近似。
            return clipped + noise

        # 映射到码值坐标，3-bit时范围为[0,7]。
        code_position = (
            clipped - self.min_value
        ) / self.step_size

        codes = torch.round(code_position)

        # 将码值恢复到特征的数值范围。
        reconstructed = (
            self.min_value + codes * self.step_size
        )

        return reconstructed

def report_quantization(name, features, quantizer):
    """
    统计一个分支的量化情况。

    调用前需要把quantizer切到eval模式，
    这样实际执行取整量化。
    """
    quantized = quantizer(features)

    # 判断哪些原始数值超出了量化范围。
    outside = (
        (features < quantizer.min_value)
        | (features > quantizer.max_value)
    )
    clipping_ratio = outside.float().mean().item()

    clipped = features.clamp(
        quantizer.min_value,
        quantizer.max_value,
    )

    # 只衡量裁剪造成的变化。
    clipping_mse = (
        clipped - features
    ).square().mean().item()

    # 只衡量取整相对于已裁剪输入的变化。
    rounding_mse = (
        quantized - clipped
    ).square().mean().item()

    # 衡量量化输出相对于原始特征的整体变化。
    # 两项MSE存在交叉项，不能简单相加得到总MSE。
    total_mse = (
        quantized - features
    ).square().mean().item()

    print(f"\n{name}")
    print("特征形状：", tuple(features.shape))
    print(
        f"原始范围："
        f"[{features.min().item():.4f}, "
        f"{features.max().item():.4f}]"
    )
    print(f"超范围元素比例：{clipping_ratio:.2%}")
    print(f"裁剪 MSE：{clipping_mse:.6f}")
    print(f"离散化 MSE：{rounding_mse:.6f}")
    print(f"总量化 MSE：{total_mse:.6f}")

    assert quantized.shape == features.shape
    assert torch.isfinite(quantized).all()
    assert torch.unique(quantized).numel() <= quantizer.num_levels

    return quantized

def check_branch_quantization():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    # 先模拟时空编码器输出，独立检查分支接口。
    features = torch.randn(1, 4, 120, 8, 8)

    common_extractor = step08.CommonFeatureExtractor(
        channels=120,
        gop_size=4,
    )
    individual_extractor = step08.IndividualFeatureExtractor()

    # 两分支分别创建量化器，方便以后配置各自的范围。
    common_quantizer = UniformQuantizer(
        bits=3,
        min_value=-1.0,
        max_value=1.0,
    )
    individual_quantizer = UniformQuantizer(
        bits=3,
        min_value=-1.0,
        max_value=1.0,
    )

    for module in (
        common_extractor,
        individual_extractor,
        common_quantizer,
        individual_quantizer,
    ):
        module.eval()

    with torch.no_grad():
        common = common_extractor(features)
        individual = features - common
        compressed = individual_extractor(individual)

        quantized_common = report_quantization(
            "公共分支",
            common,
            common_quantizer,
        )
        quantized_individual = report_quantization(
            "个体分支",
            compressed,
            individual_quantizer,
        )

    #print("\n量化后公共分支：", tuple(quantized_common.shape))
    #print("量化后个体分支：", tuple(quantized_individual.shape))
    print("双分支量化接口检查通过")

class SemanticInterleaver(nn.Module):
    """
    对每个样本内部的特征元素进行固定排列。

    输入、输出形状相同，均要求第一维是batch。
    同一个实例始终使用同一套排列。

    参数：
        elements_per_sample：一个样本内的特征元素总数。
        seed：用于生成固定排列的随机种子。
    """

    def __init__(self, elements_per_sample, seed=42):
        super().__init__()

        if elements_per_sample <= 0:
            raise ValueError("每个样本的元素数必须为正")

        self.elements_per_sample = elements_per_sample

        # 使用独立随机数生成器，避免改变其他实验的随机状态。
        generator = torch.Generator()
        generator.manual_seed(seed)

        permutation = torch.randperm(
            elements_per_sample,
            generator=generator,
        )

        # 如果 y = x[permutation]，
        # 那么 x = y[inverse_permutation]。
        inverse_permutation = torch.argsort(permutation)

        # 排列不需要学习，但应随模型移动设备并保存。
        self.register_buffer("permutation", permutation)
        self.register_buffer(
            "inverse_permutation",
            inverse_permutation,
        )

    def reorder(self, features, index):
        """使用指定索引重新排列每个样本内部的元素。"""
        if features.ndim < 2 or features.shape[0] <= 0:
            raise ValueError("输入需要包含非空batch维和特征维")

        # 保留batch维，其余维度合并为一维。
        flat = features.flatten(start_dim=1)

        if flat.shape[1] != self.elements_per_sample:
            raise ValueError("每个样本的元素数与交织器配置不一致")

        # 索引需要与特征位于同一设备。
        index = index.to(features.device)

        # 对每个样本使用相同排列，不跨样本交换元素。
        reordered = flat.index_select(dim=1, index=index)

        return reordered.reshape(features.shape)

    def forward(self, features):
        """发送端：交织。"""
        return self.reorder(features, self.permutation)

    def deinterleave(self, features):
        """接收端：逆交织，也适用于同布局的缺失mask。"""
        return self.reorder(features, self.inverse_permutation)

def check_interleaving():
    branch_shapes = {
        "公共分支": (2, 1, 120, 8, 8),
        "个体分支": (2, 4, 144, 4, 4),
    }

    for branch_index, (name, shape) in enumerate(branch_shapes.items()):
        # 使用位置编号，便于确认每个元素是否准确还原。
        total_elements = 1
        for size in shape:
            total_elements *= size

        features = torch.arange(
            total_elements,
            dtype=torch.float32,
        ).reshape(shape)

        # shape[0]是batch数，不能算进单个样本的排列长度。
        elements_per_sample = features[0].numel()

        interleaver = SemanticInterleaver(
            elements_per_sample=elements_per_sample,
            seed=42 + branch_index,
        )

        shuffled = interleaver(features)
        restored = interleaver.deinterleave(shuffled)

        #print(f"\n{name}")
        #print("交织前形状：", tuple(features.shape))
        #print("交织后形状：", tuple(shuffled.shape))
        #print(
        #    "交织后第一个样本的前8个元素：",
        #    shuffled[0].reshape(-1)[:8],
        #)

        # 排列和逆排列必须逐元素完全可逆。
        assert torch.equal(restored, features)

        # 第二个样本的编号整体更大。
        # 这两项检查确认交织没有混合不同样本。
        assert shuffled[0].max() < elements_per_sample
        assert shuffled[1].min() >= elements_per_sample

        # 同一交织器重复调用，应得到相同结果。
        assert torch.equal(interleaver(features), shuffled)

    print("\n交织与逆交织检查通过")

def transmit_packets(features, p, packet_size=16):
    """
    对batch中的每个样本分别分包并模拟丢失。

    输入：
        features：[B,T,C,H,W]。
        p：每个包独立丢失的概率。
        packet_size：每包包含的特征元素数。

    返回：
        received：接收特征，与输入同形状，缺失处为0。
        missing_mask：同形状bool张量，True表示缺失。
        dropped_packets：[B,每个样本的包数]。
    """
    if features.ndim != 5 or features.shape[0] <= 0:
        raise ValueError("输入必须为非空 [B,T,C,H,W]")

    received_samples = []
    mask_samples = []
    packet_records = []

    for sample in features:
        # sample形状是[T,C,H,W]。
        # 旧函数会将这个样本展平、按packet_size分组。
        received, mask, dropped = simulate_packet_loss(
            sample,
            p=p,
            packet_size=packet_size,
        )

        received_samples.append(received)
        mask_samples.append(mask)
        packet_records.append(dropped)

    # 堆叠回batch维。
    return (
        torch.stack(received_samples),
        torch.stack(mask_samples),
        torch.stack(packet_records),
    )

def check_interleaving_packet_loss():
    torch.manual_seed(42)

    branch_shapes = {
        "公共分支": (2, 1, 120, 8, 8),
        "个体分支": (2, 4, 144, 4, 4),
    }

    quantizer = UniformQuantizer(
        bits=3,
        min_value=-1.0,
        max_value=1.0,
    )
    quantizer.eval()

    with torch.no_grad():
        for branch_index, (name, shape) in enumerate(
            branch_shapes.items()
        ):
            # 生成范围内的示例特征，再进行评估模式量化。
            features = torch.rand(shape) * 2 - 1
            quantized = quantizer(features)

            interleaver = SemanticInterleaver(
                elements_per_sample=quantized[0].numel(),
                seed=42 + branch_index,
            )

            shuffled = interleaver(quantized)

            for p in (0.0, 0.25, 1.0):
                received_shuffled, mask_shuffled, dropped = (
                    transmit_packets(
                        shuffled,
                        p=p,
                        packet_size=16,
                    )
                )

                # 两者必须采用同一逆排列。
                received = interleaver.deinterleave(
                    received_shuffled
                )
                missing_mask = interleaver.deinterleave(
                    mask_shuffled
                )

                assert received.shape == quantized.shape
                assert missing_mask.shape == quantized.shape
                assert missing_mask.dtype == torch.bool

                # 核心检查：
                # 逆交织后，所有收到的数值必须与原位置完全一致。
                assert torch.equal(
                    received[~missing_mask],
                    quantized[~missing_mask],
                )

                # 所有被标记为缺失的位置都应填0。
                assert (received[missing_mask] == 0).all()

                if p == 0.0:
                    assert not missing_mask.any()
                    assert torch.equal(received, quantized)

                if p == 1.0:
                    assert missing_mask.all()
                    assert (received == 0).all()

                actual_loss = dropped.float().mean().item()

                print(
                    f"{name}：设定丢包概率={p:.0%}，"
                    f"实际丢包率={actual_loss:.1%}，"
                    f"包记录形状={tuple(dropped.shape)}"
                )

    print("交织、整包丢失与掩码逆交织检查通过")


def main():
    torch.manual_seed(42)

    quantizer = UniformQuantizer(
        bits=3,
        min_value=-1.0,
        max_value=1.0,
    )

    # 包含超出范围的数值，用于观察端点饱和。
    values = torch.linspace(-1.5, 1.5, steps=25)

    # ---------- 检查评估模式 ----------

    quantizer.eval()

    with torch.no_grad():
        quantized = quantizer(values)

    #print("量化级别数：", quantizer.num_levels)
    #print("量化间隔：", quantizer.step_size)
    #print("输入数值：", values)
    #print("评估输出：", quantized)
    #print("实际出现的级别：", torch.unique(quantized))

    assert quantized.shape == values.shape
    assert torch.isfinite(quantized).all()
    assert torch.unique(quantized).numel() == 8

    # ---------- 检查训练模式的梯度 ----------

    quantizer.train()

    # 使用范围内部的数值，避开裁剪造成的饱和梯度。
    features = torch.tensor(
        [-0.8, -0.3, 0.2, 0.7],
        requires_grad=True,
    )

    noisy_features = quantizer(features)

    # 检查噪声是否位于约定区间。
    noise = noisy_features.detach() - features.detach()
    tolerance = 1e-6

    assert noise.abs().max() <= (
        quantizer.step_size / 2 + tolerance
    )

    # 构造简单标量，验证加噪路径可反向传播。
    loss = noisy_features.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    #print("训练输入梯度：", features.grad)
    print("量化数值与训练梯度检查通过")
    check_branch_quantization()
    check_interleaving()
    check_interleaving_packet_loss()

if __name__ == "__main__":
    main()