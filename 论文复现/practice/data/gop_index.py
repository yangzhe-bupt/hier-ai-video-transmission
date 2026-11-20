# ======================================================================
# 文件：gop_index.py
# 位置：practice/data
# 功能：统计实际可读取帧数，生成完整GOP的起始帧编号并检查分组规则。
# 类与函数汇总：
#   - build_gop_starts：将视频划分为多个完整的GOP,返回各组的起始帧编号
#   - check_gop_starts：检查不同帧数和间隔下的分组结果
#   - count_video_frames：顺序读取视频，统计实际可读取的帧数
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py data/gop_index.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import cv2 #引入真实视频


#1.生成GOP起点
def build_gop_starts(total_frames,gop_size=4,stride=4):
    """
    将视频划分为多个完整的GOP,返回各组的起始帧编号

    参数：
        total_frames:视频总帧数
        gop_size:每组包含帧数
        stride:下一组起点向后移动帧数

    返回：
        list[int]:起始帧编号列表，尾部不足一组会被丢弃
    """
    #----------------检查数据-------------------

    #检查参数是否为整数
    for name,value in [
        ("total_frames", total_frames),
        ("gop_size", gop_size),
        ("stride", stride)
    ]:
        if type(value) is not int:
            raise ValueError(f"{name}必须是整数")

    if total_frames < 0:
        raise ValueError("总帧数不能为负数")

    if gop_size <= 0 or stride <= 0:
        raise ValueError("组长和移动间隔必须大于0")

    #--------------开始划分----------------
    
    last_start = total_frames - gop_size
    if last_start<0:
        return []#视频太短，无法组成完整GOP

    # range(起点, 终点, 间隔)：包含起点，不包含终点
    starts = list(range(0,last_start+1,stride))

    return starts

#---------------检查结果----------------------
def check_gop_starts():
    """
    检查不同帧数和间隔下的分组结果

    参数：
        无
    
    返回：
        无
        检查错误时抛出AssertionError
    """
    # 每组4帧，不重叠：0～3、4～7、8～11
    assert build_gop_starts(12) == [0, 4, 8]

    # 尾部第12、13号帧不足一组，丢弃
    assert build_gop_starts(14) == [0, 4, 8]

    # 视频不足4帧，返回空列表
    assert build_gop_starts(3) == []
    assert build_gop_starts(0) == []

    # 每组4帧，起点每次移动2帧：0～3、2～5、4～7
    assert build_gop_starts(8, gop_size=4, stride=2) == [0, 2, 4]

    print("不重叠分组：", build_gop_starts(12))
    print("尾部不足：", build_gop_starts(14))
    print("重叠分组：", build_gop_starts(8, 4, 2))
    print("GOP起始帧清单检查通过")

#--------------统计视频帧数-----------------
def count_video_frames(video_path):
    """
    顺序读取视频，统计实际可读取的帧数

    参数：
        video_path:视频文件路径

    返回：
        total_frames:从视频开头连续成功读取的帧数
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("视频打开失败，请检查路径")

    total_frames = 0

    try:
        while True:
            #每次读取下一帧，_表示不保留图像内容
            ok,_ = cap.read()

            if not ok:
                break

            total_frames +=1

    finally:
        cap.release()

    return total_frames

#主程序
def main():
    #---------检查分组模块-----------
    check_gop_starts()

    #---------读取真实视频帧数-------
    video_path = r"D:\learn\pytorch\data\毕洋洋.mp4"
    total_frames = count_video_frames(video_path)

    #---------分组--------------
    gop_size = 4
    starts = build_gop_starts(total_frames,gop_size,4)
    print("实际可读取帧数：", total_frames)
    print("完整GOP数量：", len(starts))
    print("前5个GOP起点：", starts[:5])

    if starts:
        print("最后一个GOP起点：", starts[-1])

    # 不重叠分组下，计算尾部未使用的帧数
    unused_frames = total_frames - len(starts) * gop_size
    print("尾部未使用帧数：", unused_frames)

#测试
if __name__=='__main__':
    main()