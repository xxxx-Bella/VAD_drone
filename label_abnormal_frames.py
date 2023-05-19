import numpy as np
import os

dir = 'D:\\Desktop\\postG\\VAD\\dataset\\Campus\\playground\\frames\\abnormal\\'

# 手工标注异常帧
abnormal_dict = {
    '01_batterycar_1': (400, 516),
    '01_batterycar_2': (53, 127),
    '01_bike': (123, 206),
    '01_bike2': (280, 377),
    '01_tiny_batterycar': (349, 568)
}

for k, v in abnormal_dict.items():
    start, end = v
    frame_path = os.path.join(dir, k)
    frames = [f for f in os.listdir(frame_path) if f.endswith(".jpg")]
    print(len(frames))
    label = np.zeros(len(frames), dtype=int)
    label[start:end] = 1
    # print('abnormal frame of {}: {}~{}'.format(k, start, end))
    # print('label:', label)
    # 保存为npy文件
    np.save(frame_path+'.npy', label)
