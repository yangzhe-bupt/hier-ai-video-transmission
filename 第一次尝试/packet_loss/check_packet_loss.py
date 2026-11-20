"""兼容入口：从仓库根目录运行python -m packet_loss.check_packet_loss。"""
from checks.check_packet_loss import check_packet_loss, main

if __name__ == '__main__':
    main()
