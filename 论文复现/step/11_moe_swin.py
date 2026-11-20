"""Step11：先实现逐位置的top-K专家路由器。"""

import importlib

import torch

# 单向复用Step07的Swin块，Step07不导入本文件，避免循环导入。
step07 = importlib.import_module("07_spatiotemporal_codec")
import torch.nn as nn


class TopKRouter(nn.Module):
    """
    为每个位置选择K个专家，并生成组合权重。

    输入：
        [窗口数, 位置数, dim]

    返回：
        selected_weights：[窗口数, 位置数, top_k]
        expert_indices：  [窗口数, 位置数, top_k]

    专家选择发生在每个位置上，而非整个窗口上。
    """

    def __init__(self, dim=96, num_experts=4, top_k=2):
        super().__init__()

        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k必须位于[1,num_experts]")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 将每个位置的特征映射为各专家的路由分数。
        # 例如96维输入产生4个分数。
        self.gate = nn.Linear(
            dim,
            num_experts,
            bias=False,
        )

    def forward(self, features, return_aux=False):
        if features.ndim != 3 or features.shape[-1] != self.dim:
            raise ValueError("输入必须为 [窗口数,位置数,dim]")

        # [Wn,L,dim] → [Wn,L,num_experts]
        logits = self.gate(features)

        # 在专家维度归一化。
        # 每个位置对全部专家的概率总和为1。
        probabilities = logits.softmax(dim=-1)

        # 对每个位置取概率最高的K个专家。
        selected_weights, expert_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=-1,
        )

        # 被选中的K个专家重新归一化，使组合权重之和为1。
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1,
            keepdim=True,
        )

        if return_aux:
            # 统计每个专家接到多少次top-K分配。
            # expert_indices是整数索引，不参与梯度计算。
            counts = torch.bincount(
                expert_indices.reshape(-1),
                minlength=self.num_experts,
            )

            # 分母是“位置数×top_k”，因此所有负载比例之和为1。
            load_fraction = (
                counts.to(probabilities.dtype)
                / expert_indices.numel()
            )

            # 使用筛选前的完整概率。
            # 输入为[窗口数,位置数,专家数]，
            # 对前两维取平均，得到[专家数]。
            mean_probability = probabilities.mean(dim=(0, 1))

            # load_fraction无梯度，mean_probability保留梯度。
            auxiliary_loss = self.num_experts * (
                load_fraction * mean_probability
            ).sum()

            return selected_weights, expert_indices, auxiliary_loss

        # 保留之前的两项返回，原有检查仍可运行。
        return selected_weights, expert_indices

class FeedForwardExpert(nn.Module):
    """
    一个独立的前馈专家。

    输入、输出：[分配到的位置数, dim]
    内部先扩展通道，再映射回原维度。
    """

    def __init__(self, dim=96, hidden_dim=384):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, features):
        return self.network(features)

class SparseMoE(nn.Module):
    """
    逐位置top-K混合专家。

    输入、输出：[...,dim]，包括窗口序列和Swin的BTHWC布局。
    每个位置只交给选中的K个专家处理。
    """

    def __init__(
        self,
        dim=96,
        num_experts=4,
        top_k=2,
        mlp_ratio=4,
    ):
        super().__init__()

        self.dim = dim
        self.top_k = top_k

        self.router = TopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # ModuleList会注册每个专家的参数，
        # 方便优化器更新、模型保存和设备迁移。
        self.experts = nn.ModuleList([
            FeedForwardExpert(
                dim=dim,
                hidden_dim=int(dim * mlp_ratio),
            )
            for _ in range(num_experts)
        ])

    def forward(self, features, return_aux=False):
        if features.ndim < 2 or features.shape[-1] != self.dim:
            raise ValueError("输入必须至少二维，且最后一维等于dim")
        if features.numel() == 0:
            raise ValueError("MoE输入不能为空")

        # 保存外部布局，最后恢复。路由和专家逐位置计算，
        # 因而可以把窗口、批次、时空位置统一展平。
        original_shape = features.shape
        features = features.reshape(1, -1, self.dim)

        weights, indices, auxiliary_loss = self.router(
            features,
            return_aux=True,
        )

        # 合并窗口数和位置数，方便按专家收集位置。
        # [Wn,L,dim] → [全部位置数,dim]
        flat_features = features.reshape(-1, self.dim)

        # [Wn,L,K] → [全部位置数,K]
        flat_weights = weights.reshape(-1, self.top_k)
        flat_indices = indices.reshape(-1, self.top_k)

        # 保存每个位置最终汇总的结果。
        output = torch.zeros_like(flat_features)

        for expert_id, expert in enumerate(self.experts):
            # 查找哪些位置选择了当前专家。
            #
            # position_ids：这些位置在展平序列中的编号。
            # slot_ids：专家位于该位置的第几个top-K槽位。
            position_ids, slot_ids = torch.where(
                flat_indices == expert_id
            )

            # 某个专家可能在本批次没有接到任何位置。
            if position_ids.numel() == 0:
                continue

            expert_input = flat_features.index_select(
                dim=0,
                index=position_ids,
            )

            expert_output = expert(expert_input)

            # 取出当前专家在各位置对应的组合权重。
            selected_weights = flat_weights[
                position_ids, slot_ids
            ].unsqueeze(-1)

            weighted_output = expert_output * selected_weights

            # 将结果加回原位置。
            # 一个位置会从K个专家收到贡献，因此使用累加。
            output = output.index_add(
                dim=0,
                index=position_ids,
                source=weighted_output,
            )

        output = output.reshape(original_shape)

        if return_aux:
            return output, auxiliary_loss

        return output

def check_sparse_moe():
    torch.manual_seed(42)

    features = torch.randn(
        4, 64, 96,
        requires_grad=True,
    )

    moe = SparseMoE(
        dim=96,
        num_experts=4,
        top_k=2,
    )

    # 单独查看路由分配，用于解释专家梯度检查。
    # 当前路由器没有随机操作，所以与forward内的选择一致。
    with torch.no_grad():
        _, indices = moe.router(features)

        counts = torch.bincount(
            indices.reshape(-1),
            minlength=4,
        )

    output = moe(features)

    print("MoE 输入：", tuple(features.shape))
    print("MoE 输出：", tuple(output.shape))
    print("各专家接收的位置数：", counts)

    assert output.shape == features.shape
    assert torch.isfinite(output).all()

    # 256个位置，每个位置选择2个专家，共512次分配。
    assert counts.sum().item() == 4 * 64 * 2

    loss = output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    router_gradient = moe.router.gate.weight.grad
    assert router_gradient is not None
    assert torch.isfinite(router_gradient).all()

    for expert_id, expert in enumerate(moe.experts):
        # 未被选中的专家没有参与本次计算，
        # 本次任务损失不会给它产生梯度。
        if counts[expert_id].item() == 0:
            continue

        for name, parameter in expert.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(
                    f"专家{expert_id}的{name}没有梯度"
                )

            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"专家{expert_id}的{name}梯度异常"
                )

    print("稀疏MoE分发与梯度检查通过")

def check_moe_auxiliary_loss():
    torch.manual_seed(42)

    features = torch.randn(
        4, 64, 96,
        requires_grad=True,
    )

    moe = SparseMoE(
        dim=96,
        num_experts=4,
        top_k=2,
    )

    output, auxiliary_loss = moe(
        features,
        return_aux=True,
    )

    print("MoE 辅助损失：", auxiliary_loss.item())

    # 辅助损失应为有限的标量。
    assert auxiliary_loss.ndim == 0
    assert torch.isfinite(auxiliary_loss)

    # 单独检查辅助损失到路由器的梯度路径。
    # retain_graph=True，让后面的总损失还能使用这张计算图。
    auxiliary_gradient = torch.autograd.grad(
        auxiliary_loss,
        moe.router.gate.weight,
        retain_graph=True,
    )[0]

    assert torch.isfinite(auxiliary_gradient).all()

    # 检查用任务损失；正式训练时会换成视频重建等损失。
    task_loss = output.square().mean()

    # 0.01只是本轮检查示例，尚未确定论文中的辅助损失权重。
    total_loss = task_loss + 0.01 * auxiliary_loss
    total_loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    router_gradient = moe.router.gate.weight.grad
    assert router_gradient is not None
    assert torch.isfinite(router_gradient).all()

    print(
        "辅助损失对路由器的平均梯度绝对值：",
        auxiliary_gradient.abs().mean().item(),
    )
    print("MoE辅助损失梯度检查通过")

def check_moe_swin_block():
    """检查布局兼容、普通块返回值及移位MoE块的梯度。无需训练。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)

    # 相同的位置数据采用不同布局，结果应保持逐位置一致。
    moe = SparseMoE(dim=96, num_experts=4, top_k=2)
    layout_input = torch.randn(1, 4, 8, 8, 96)
    with torch.no_grad():
        output_5d, aux_5d = moe(layout_input, return_aux=True)
        output_3d, aux_3d = moe(
            layout_input.reshape(4, 64, 96), return_aux=True
        )
    torch.testing.assert_close(output_5d.reshape(4, 64, 96), output_3d)
    torch.testing.assert_close(aux_5d, aux_3d)
    print("MoE三维/五维布局一致性检查通过")

    # 普通MLP块仍然可以只返回特征；可选辅助损失应为零。
    ordinary_block = step07.SwinBlock3D(dim=96)
    features = torch.randn(1, 4, 96, 8, 8, requires_grad=True)
    with torch.no_grad():
        ordinary_output = ordinary_block(features)
        output_with_aux, zero_aux = ordinary_block(features, return_aux=True)
    torch.testing.assert_close(ordinary_output, output_with_aux)
    assert zero_aux.ndim == 0 and zero_aux.item() == 0
    print("普通MLP块返回接口兼容检查通过")

    block = step07.SwinBlock3D(
        dim=96,
        num_heads=3,
        window_size=(4, 4, 4),
        shift_size=(0, 2, 2),
    )
    # 替换模块会自动注册新专家参数。
    # 正式训练必须在完成替换之后创建优化器。
    block.mlp = SparseMoE(dim=96, num_experts=4, top_k=2)

    output, auxiliary_loss = block(features, return_aux=True)
    print("MoE Swin输入：", tuple(features.shape))
    print("MoE Swin输出：", tuple(output.shape))
    print("块辅助损失：", auxiliary_loss.item())
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert auxiliary_loss.ndim == 0 and torch.isfinite(auxiliary_loss)

    # 同时验证任务输出和路由辅助损失的反向传播。
    loss = output.square().mean() + 0.01 * auxiliary_loss
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    # 未接收位置的专家可以没有任务梯度。
    # 其余参数均需具有有限梯度；专家分发的详细检查由前面的函数负责。
    for name, parameter in block.named_parameters():
        if parameter.grad is None:
            if name.startswith("mlp.experts."):
                continue
            raise RuntimeError(f"{name}没有梯度")
        assert torch.isfinite(parameter.grad).all(), name

    print("MoE Swin块联合梯度检查通过")

def replace_mlp_with_moe(model, num_experts=4, top_k=2):
    """
    将指定模型内3D Swin块的普通MLP替换为SparseMoE。

    返回替换的块数。

    必须在创建优化器之前调用，
    才能确保优化器持有新加入的专家参数。
    """
    replaced_count = 0

    # 先取得模块列表，再进行替换，避免遍历时修改模块树。
    for module in list(model.modules()):
        if not isinstance(module, step07.SwinBlock3D):
            continue

        # 跳过已经替换成MoE的块，防止重复初始化。
        if not isinstance(module.mlp, nn.Sequential):
            continue

        # 当前普通MLP的第一层是Linear(dim,hidden_dim)。
        # 读取原来的扩展比例，保持专家的隐藏层规模一致。
        hidden_dim = module.mlp[0].out_features
        mlp_ratio = hidden_dim / module.dim

        module.mlp = SparseMoE(
            dim=module.dim,
            num_experts=num_experts,
            top_k=top_k,
            mlp_ratio=mlp_ratio,
        )

        replaced_count += 1

    return replaced_count

def check_moe_temporal_codec():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    encoder = step07.SpatiotemporalEncoder()
    decoder = step07.SpatiotemporalDecoder()

    encoder_count = replace_mlp_with_moe(encoder)
    decoder_count = replace_mlp_with_moe(decoder)

    print("编码器替换的MoE块数：", encoder_count)
    print("解码器替换的MoE块数：", decoder_count)

    assert encoder_count == 4
    assert decoder_count == 4

    # 再次调用不应替换或重新初始化已有专家。
    assert replace_mlp_with_moe(encoder) == 0

    encoder.eval()
    decoder.eval()

    # 模拟空间编码器输出。
    features = torch.randn(1, 4, 72, 32, 32)

    with torch.no_grad():
        compressed = encoder(features)
        restored = decoder(compressed)

    print("MoE时空编码输入：", tuple(features.shape))
    print("MoE时空编码输出：", tuple(compressed.shape))
    print("MoE时空解码输出：", tuple(restored.shape))

    assert compressed.shape == (1, 4, 120, 8, 8)
    assert restored.shape == features.shape
    assert torch.isfinite(compressed).all()
    assert torch.isfinite(restored).all()

    print("完整MoE时空编解码前向检查通过")

def check_encoder_auxiliary_loss():
    """验证编码器块均值汇总、原接口兼容及四个路由器的梯度。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)

    encoder = step07.SpatiotemporalEncoder()
    features = torch.randn(1, 4, 72, 32, 32, requires_grad=True)

    # 普通MLP编码器也支持return_aux=True，此时辅助损失为零。
    with torch.no_grad():
        original_output = encoder(features)
        output_with_aux, zero_aux = encoder(features, return_aux=True)
    torch.testing.assert_close(original_output, output_with_aux)
    assert zero_aux.ndim == 0 and zero_aux.item() == 0
    print("普通编码器返回接口兼容检查通过")

    count = replace_mlp_with_moe(encoder)
    assert count == 4

    # 验证新接口的特征与旧接口一致，并对照阶段损失计算块均值。
    with torch.no_grad():
        ordinary_output = encoder(features)
        stage4_output, stage4_sum = encoder.stage4(features, return_aux=True)
        stage5_output, stage5_sum = encoder.stage5(stage4_output, return_aux=True)
        expected_mean = (stage4_sum + stage5_sum) / count

    output, auxiliary_loss = encoder(features, return_aux=True)
    torch.testing.assert_close(output, ordinary_output)
    torch.testing.assert_close(output, stage5_output)
    torch.testing.assert_close(auxiliary_loss, expected_mean)
    assert output.shape == (1, 4, 120, 8, 8)
    assert torch.isfinite(output).all()
    assert auxiliary_loss.ndim == 0 and torch.isfinite(auxiliary_loss)

    print("带辅助损失的编码输出：", tuple(output.shape))
    print("编码器平均辅助损失：", auxiliary_loss.item())

    gates = [
        block.mlp.router.gate.weight
        for stage in (encoder.stage4, encoder.stage5)
        for block in stage.blocks
    ]
    # 单独检查汇总辅助项能否传回每个路由器，避免任务损失掩盖断图。
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss, gates, retain_graph=True
    )
    assert all(torch.isfinite(g).all() for g in auxiliary_gradients)

    # 检查用目标；尚未进行optimizer.step，不更新模型参数。
    loss = output.square().mean() + 0.01 * auxiliary_loss
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    for gate in gates:
        assert gate.grad is not None and torch.isfinite(gate.grad).all()

    print("编码器辅助损失块均值检查通过")
    print("编码器辅助损失汇总与梯度检查通过")


def check_decoder_and_joint_auxiliary_loss():
    """检查解码器块均值及编码、解码共8个路由器的联合梯度。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    encoder = step07.SpatiotemporalEncoder()
    decoder = step07.SpatiotemporalDecoder()
    features = torch.randn(1, 4, 72, 32, 32, requires_grad=True)

    # 原始MLP解码器应兼容旧接口，并返回零辅助损失。
    decoder_input = torch.randn(1, 4, 120, 8, 8)
    with torch.no_grad():
        old_output = decoder(decoder_input)
        new_output, zero_aux = decoder(decoder_input, return_aux=True)
    torch.testing.assert_close(old_output, new_output)
    assert zero_aux.ndim == 0 and zero_aux.item() == 0
    print("普通解码器返回接口兼容检查通过")

    encoder_count = replace_mlp_with_moe(encoder)
    decoder_count = replace_mlp_with_moe(decoder)
    assert encoder_count == decoder_count == 4

    encoded, encoder_aux = encoder(features, return_aux=True)
    decoded, decoder_aux = decoder(encoded, return_aux=True)

    # 对照逐阶段的显式汇总，同时检查默认返回的特征没有改变。
    with torch.no_grad():
        middle, stage5_aux = decoder.decode_stage5(encoded, return_aux=True)
        reference_output, stage4_aux = decoder.decode_stage4(middle, return_aux=True)
        reference_aux = (stage5_aux + stage4_aux) / decoder_count
        plain_output = decoder(encoded)
    torch.testing.assert_close(decoded, plain_output)
    torch.testing.assert_close(decoded, reference_output)
    torch.testing.assert_close(decoder_aux, reference_aux)
    assert decoded.shape == features.shape
    assert torch.isfinite(decoded).all()

    # 按块数加权，得到8个块整体的均值。
    # 当前两端各4块，等价于(encoder_aux+decoder_aux)/2。
    total_blocks = encoder_count + decoder_count
    joint_aux = (
        encoder_aux * encoder_count + decoder_aux * decoder_count
    ) / total_blocks
    assert joint_aux.ndim == 0 and torch.isfinite(joint_aux)

    stages = (
        encoder.stage4, encoder.stage5,
        decoder.decode_stage5, decoder.decode_stage4,
    )
    gates = [block.mlp.router.gate.weight for stage in stages for block in stage.blocks]
    auxiliary_gradients = torch.autograd.grad(joint_aux, gates, retain_graph=True)
    assert len(auxiliary_gradients) == 8
    assert all(torch.isfinite(g).all() for g in auxiliary_gradients)

    # 任务项使用特征重建MSE，仅用于联合反向检查。
    # 正式训练会按论文定义视频重建等损失，并核实辅助项系数。
    loss = (decoded - features).square().mean() + 0.01 * joint_aux
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    for gate in gates:
        assert gate.grad is not None and torch.isfinite(gate.grad).all()

    print("带辅助损失的解码输出：", tuple(decoded.shape))
    print("解码器平均辅助损失：", decoder_aux.item())
    print("编解码器整体平均辅助损失：", joint_aux.item())
    print("解码器辅助损失块均值检查通过")
    print("编解码器8个路由器联合梯度检查通过")


def main():
    torch.manual_seed(42)

    # 模拟4个窗口，每个窗口64个位置，每个位置96维。
    features = torch.randn(
        4, 64, 96,
        requires_grad=True,
    )

    router = TopKRouter(
        dim=96,
        num_experts=4,
        top_k=2,
    )

    weights, indices = router(features)

    print("路由输入：", tuple(features.shape))
    print("选中专家编号：", tuple(indices.shape))
    print("选中专家权重：", tuple(weights.shape))

    print("第一个位置的专家编号：", indices[0, 0])
    print("第一个位置的组合权重：", weights[0, 0])

    assert indices.shape == (4, 64, 2)
    assert weights.shape == (4, 64, 2)

    assert indices.min() >= 0
    assert indices.max() < 4
    assert torch.isfinite(weights).all()

    # 每个位置的两个组合权重之和应为1。
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(4, 64),
    )

    # 取第一项权重构造检查目标。
    # 不使用全部权重之和，因为其恒为1。
    loss = weights[..., 0].mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    gradient = router.gate.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    print("路由权重和与梯度检查通过")
    check_sparse_moe()
    check_moe_auxiliary_loss()
    check_moe_swin_block()
    check_moe_temporal_codec()
    check_encoder_auxiliary_loss()
    check_decoder_and_joint_auxiliary_loss()
    

if __name__ == "__main__":
    main()
