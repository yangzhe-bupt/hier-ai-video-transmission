import torch

#丢包函数
def simulate_packet_loss(features, p, packet_size):
    """
    模拟每个包以概率 p 独立丢失。

    参数:
        features: torch.Tensor，形状 [1, 8, 4, 4]，共 128 个元素
        p:        float，每包丢失概率，0 <= p <= 1
        packet_size: int，每包元素数，默认 16（128/16=8 包）

    返回:
        received:    丢包后的张量，形状与 features 相同，缺失处置 0
        mask:        元素级掩码，形状与 features 相同，1=缺失
        drop_packet: 包级丢失记录，形状 [num_packets]，bool，True=该包丢失
    """
    # 检查参数，避免除零或无法正确分组
    if not 0 <= p <= 1:
        raise ValueError("p必须在[0, 1]之间")

    if not isinstance(packet_size, int) or packet_size <= 0:
        raise ValueError("packet_size必须是正整数")

    if features.numel() == 0:
        raise ValueError("features不能为空")

    if features.numel() % packet_size != 0:
        raise ValueError("元素总数必须能被packet_size整除")
    
    numel = features.numel()  # 总数据量
    num_packet = numel // packet_size  # 包数

    # 展平，分包
    flat = features.reshape(-1)
    packets = flat.reshape(num_packet, packet_size)

    #记录输入所在设备
    device = features.device

    # 包级独立
    if p == 0.0:
        drop_packet = torch.zeros(num_packet, dtype=torch.bool,device=device)
    elif p == 1.0:
        drop_packet = torch.ones(num_packet, dtype=torch.bool,device=device)
    else:
        probs = torch.full((num_packet,), p,dtype=torch.float,device=device)  # [0.25,...,0.25]
        drop_packet = torch.bernoulli(probs).bool()  # 伯努利概型以概率赋值1，0

    # 包级掩码
    mask_packets = drop_packet.unsqueeze(1).expand(num_packet, packet_size)
    #[1,8,16]

    # 置零
    # torch.where(condition,input,other):
    # output[i] = input[i]  if condition[i] 为真
    #           = other[i]  if condition[i] 为假
    receive_packet = torch.where(mask_packets, torch.zeros_like(packets), packets)

    # 恢复形状
    received = receive_packet.reshape(features.shape)
    mask = mask_packets.reshape(features.shape)

    # 丢包后的张量  掩码 丢包记录
    return received, mask, drop_packet

#主流程
def main():
    torch.manual_seed(24)

    features = torch.randn(1,8,4,4)

    received, mask, drop_packet = simulate_packet_loss(features, 0.25, 16)

    print("输出形状：", received.shape)
    print("掩码形状：", mask.shape)
    print("包级记录：", drop_packet)
    print("缺失元素数：", mask.sum().item())

    check_packet_loss("cpu")
    
    if torch.cuda.is_available():
        check_packet_loss("cuda")


# 测试
if __name__ == '__main__':
    main()
