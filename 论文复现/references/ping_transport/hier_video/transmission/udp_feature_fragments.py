# ======================================================================
# 文件：udp_feature_fragments.py
# 功能：将较大的应用层特征包拆成受 IPv4 路径 MTU 约束的 UDP 负载。
# 函数汇总：
#   - fragment_feature_packet：生成带 GOP、原包及分片索引的负载。
#   - reassemble_feature_packets：按索引合并收到的分片，只交付完整原包。
# 约定：公共副本复用原包分片标识。任一副本的同索引分片均可补齐原包。
# ======================================================================

from struct import Struct


IPV4_UDP_BYTES = 28
FRAGMENT_HEADER = Struct("!2sIHHHH")
_MAGIC = b"F1"


def _capacity(path_mtu):
    """计算扣除 IPv4/UDP 头和应用层分片头后，每个分片可装的字节。"""
    if type(path_mtu) is not int or not IPV4_UDP_BYTES + FRAGMENT_HEADER.size < path_mtu <= 65535:
        raise ValueError("IPv4 路径 MTU 无效")
    return path_mtu - IPV4_UDP_BYTES - FRAGMENT_HEADER.size


def fragment_feature_packet(raw, *, gop_id, packet_id, path_mtu):
    """
    参数：
        raw：已序列化的应用层特征包。
        gop_id、packet_id：接收端重组使用的原包身份。
        path_mtu：包括 IPv4 和 UDP 包头的最大 IP 包长度。
    返回：每个 UDP 数据报的应用层负载，均包含分片头。
    """
    capacity = _capacity(path_mtu)
    if (type(gop_id) is not int or not 0 <= gop_id <= 0xFFFFFFFF
            or type(packet_id) is not int or not 0 <= packet_id <= 0xFFFF
            or not 0 < len(raw) <= 0xFFFF):
        raise ValueError("分片标识或原包长度无效")
    count = (len(raw) + capacity - 1) // capacity
    return tuple(
        FRAGMENT_HEADER.pack(_MAGIC, gop_id, packet_id, index, count, len(raw))
        + raw[index * capacity:(index + 1) * capacity]
        for index in range(count)
    )


def reassemble_feature_packets(datagrams, *, gop_id, path_mtu):
    """
    恢复当前 GOP 收齐的原包，返回原包标识到完整字节的字典。
    允许乱序和内容一致的副本，丢失任一索引时不交付对应原包。
    包头冲突、错误分片长度和跨 GOP 混入会报错。
    """
    capacity = _capacity(path_mtu)
    packet_info = {}
    chunks = {}
    for datagram in datagrams:
        if not FRAGMENT_HEADER.size < len(datagram) <= path_mtu - IPV4_UDP_BYTES:
            raise ValueError("UDP 负载长度无效")
        magic, source_gop, packet_id, index, count, total = FRAGMENT_HEADER.unpack_from(datagram)
        if (magic != _MAGIC or source_gop != gop_id or not 0 <= index < count
                or count != (total + capacity - 1) // capacity):
            raise ValueError("分片头无效或 GOP 不匹配")
        payload = datagram[FRAGMENT_HEADER.size:]
        if len(payload) != min(capacity, total - index * capacity):
            raise ValueError("分片有效载荷与包头不一致")
        if packet_info.setdefault(packet_id, (count, total)) != (count, total):
            raise ValueError("同一原包的分片头冲突")
        group = chunks.setdefault(packet_id, {})
        if group.setdefault(index, payload) != payload:
            raise ValueError("同一位置的两个分片内容不一致")
    return {
        packet_id: b"".join(group[index] for index in range(packet_info[packet_id][0]))
        for packet_id, group in chunks.items() if len(group) == packet_info[packet_id][0]
    }

