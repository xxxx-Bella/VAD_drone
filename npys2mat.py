import os
import numpy as np
from scipy import io

npy_path = 'D:\\Desktop\\postG\\VAD\\dataset\\ShanghaiTech_Campus\\Testing\\test_frame_mask'  # '/home/featurize/work/pred_CVPR2018/Data/Data/Bike_Roundabout/sequence1'
# 定义文件夹路径和合并后的npy文件保存路径
merged_path = '/home/featurize/work/pred_CVPR2018/Data/Data/Bike_Roundabout/sequence1/all.npy'
# mat_path = '/home/featurize/work/pred_CVPR2018/Data/Data/ped2/ped2.mat'
mat_path = '/home/featurize/work/pred_CVPR2018/Data/Data/avenue/avenue.mat'  # '/home/featurize/work/pred_CVPR2018/Data/Data/Bike_Roundabout/sequence1/Bike_Roundabout.mat'
np.set_printoptions(threshold=np.inf)

# 1. concat several npys into one npy

# 遍历文件夹，将npy文件读入到一个列表中
file_list = []
for file_name in os.listdir(npy_path):
    if file_name.endswith(".npy"):
        # print('file_name:', file_name)  # 07.npy
        file_list.append(os.path.join(npy_path, file_name))
print('file_list:', file_list)

tmp = np.load(file_list[1], allow_pickle=True)
print(file_list[1])
# print('01_0015.npy')
print(tmp)
print('npy len:', len(tmp))

# # load each .npy，将多个数组合并成一个数组，每个数组作为新数组中的一个元素
# # arrays = [np.load(file, allow_pickle=True) for file in file_list]
# # merged_array = np.array(arrays)
# # np.save(merged_path, merged_array)


# # 1.2 查看合并后的 all.npy
# gt = np.load(merged_path, allow_pickle=True)
# np.set_printoptions(threshold=np.inf)  # 打印时显示全部内容
# for i in range(len(gt)):
#     print('gt {}: num_gt = {}'.format(i, len(gt[i])))


# 2. transform all.npy into Bike_Roundabout.mat

# gt = np.load(merged_path, allow_pickle=True)
# io.savemat(mat_path, {'gt': gt})


# # 2.2 查看 mat文件
# mat = io.loadmat(mat_path)

# # 打印所有变量名
# print("Variable names:")
# for name in mat:
#     if not name.startswith('__'):
#         print(name)

# # 打印某个特定变量的值
# var_name = 'gt'
# if var_name in mat:
#     var_value = mat[var_name]
#     np.set_printoptions(threshold=np.inf)
#     print(f"{var_name} = {var_value}")
#     # print('length =', len(var_value))
#     print('shape =', var_value.shape)

