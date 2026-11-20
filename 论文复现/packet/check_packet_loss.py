"""
检查丢包函数
"""
import torch
from packet_loss import simulate_packet_loss


def check_packet_loss(device):
    """
    在指定设备上，检查p=0时的情况

    参数
        device:运行设备，如CPU，CUDA
    
    返回：
        无
        检查失败时抛出AssertionError
    """
    features = torch.arange(1,129,dtype=torch.float,device=device).reshape(1,8,4,4)

    original = features.clone()

    received,mask,drop_packet = simulate_packet_loss(features,0.0,16)

    assert torch.equal(received,original),"无丢包时发生错误"
    assert not drop_packet.any().item(), "包级记录不应出现丢失"
    assert not mask.any().item(), "元素掩码不应出现缺失"

    assert torch.equal(features, original), "原始输入被修改"

    print(f"设备={device}，p=0检查通过")

    received, mask, drop_packet = simulate_packet_loss(features,p=1.0,packet_size=16)

    # all()：所有元素都为True时，才返回True
    assert drop_packet.all().item(), "所有包都应标记为丢失"
    assert mask.all().item(), "所有元素都应标记为缺失"

    # received == 0：逐元素判断是否为零
    # torch.all(...)：检查是否全部为零
    assert torch.all(received == 0).item(), "全部丢包时输出应全零"

    # 即使全部丢失，也不能修改原始输入
    assert torch.equal(features, original), "原始输入被修改"

    print(f"设备={device}，p=1检查通过")

    received, mask, drop_packet = simulate_packet_loss(features,p=0.25,packet_size=16)

    # 检查形状和类型
    assert received.shape == features.shape
    assert mask.shape == features.shape
    assert mask.dtype == torch.bool
    assert drop_packet.dtype == torch.bool
    assert drop_packet.shape == (8,)

    # 检查输出都位于输入所在设备
    assert received.device == features.device
    assert mask.device == features.device
    assert drop_packet.device == features.device

    # mask为True的位置必须为零
    assert torch.all(received[mask] == 0).item()

    # ~mask：对bool掩码取反，选出收到的位置
    assert torch.equal(received[~mask], original[~mask])

    # 将包级状态展开成预期的元素级掩码
    # [8] → [8, 1] → [8, 16]
    expected_mask = drop_packet.unsqueeze(1).expand(8, 16)

    # 检查每个包内的16个元素具有相同缺失状态
    assert torch.equal(mask.reshape(8, 16), expected_mask)

    # 缺失元素数量必须等于丢失包数×16
    assert mask.sum().item() == drop_packet.sum().item() * 16

    # 检查输入未被修改
    assert torch.equal(features, original)

    print(f"设备={device}，随机丢包检查通过")

#主流程
def main():
    """依次在CPU和可用的GPU上检查丢包模块。"""

    check_packet_loss("cpu")

    if torch.cuda.is_available():
        check_packet_loss("cuda")

#测试
if __name__ == "__main__":
    main()