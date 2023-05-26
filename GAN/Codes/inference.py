# import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
'''
推断
'''
import os
import time
import numpy as np
import pickle  # 实现 Python 对象的存储及恢复
from models import generator
from utils import DataLoader, load, psnr_error  # save
from constant import const
import evaluate

# slim = tf.contrib.slim
# import tf_slim as slim


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER  # ../Bike_Roundabout/sequence1/test

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR  # the directory to save models OR model.ckpt-xxx
psnr_dir = const.PSNR_DIR  # /psnrs/..
evaluate_name = const.EVALUATE
evaluate.set_data_dir("/p300/datasets")
print(evaluate.DATA_DIR)
print(const)


# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.compat.v1.placeholder(shape=[1, height, width, 3*(num_his+1)], dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))  # (1, 256, 256, 12)
    print('test prediction gt = {}'.format(test_gt))  # (1, 256, 256, 3)

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver and loader
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)  # tf.train.Saver()保存和加载模型; tf.global_variables()获取程序中的变量,返回的值是变量的一个列表

    restore_var = [v for v in tf.global_variables()]  # 储存变量
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)
        # print('\ninference_func', '\nckpt:', ckpt, '\ndataset_name:', dataset_name, '\nevaluate_name:', evaluate_name)
        psnr_records = []
        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())
        total = 0  # 数据集(如ped2)的总帧数
        timestamp = time.time()
        # print('videos_info.items():', videos_info.items())

        # 依次处理每个视频片段
        for video_name, video in videos_info.items():
            if not video_name.endswith('.npy') and not video_name.endswith('.mat'):  # and video_name == '01':
                length = video['length']
                total += length
                psnrs = np.empty(shape=(length,), dtype=np.float32)  # 长度为length的一维数组
                print('video = {} / {}, length = {}'.format(video_name, num_videos, length))

                # 依次处理每 t+1 个帧
                for i in range(num_his, length):  # length; num_his = t = 4  (4, 0)
                    video_clip = data_loader.get_video_clips(video_name, i-num_his, i+1)  # video, start, end (clip长度为 t+1： 0~t+1; 1~t+2; 2~t+3...)
                    psnr = sess.run(test_psnr_error, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                    psnrs[i] = psnr  # 计算每个帧的 psnr
                    # print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
                    #     video_name, num_videos, i, length, psnr))

                psnrs[0:num_his] = psnrs[num_his]  # 前 t 帧 (0,1,2,3) 的 psnr 都赋值为第四帧的 psnr (t=3)
                psnr_records.append(psnrs)
        # 预测结果
        result_dict = {'dataset': dataset_name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}

        used_time = time.time() - timestamp
        print('total time = {}, fps = {}'.format(used_time, total / used_time))  # fps: frame pre second

        # save result_dict
        # TODO specify what's the actual name of ckpt.
        pickle_path = os.path.join(psnr_dir, os.path.split(ckpt)[-1])  # psnrs/model_name/
        with open(pickle_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)  # dump()将 obj 保存到 psnr file 中

        # 重新 inference 之前要将 psnrs/model_name/ 中的 model 删除
        results = evaluate.evaluate(evaluate_name, pickle_path)  # 指定评价指标 对模型进行评估
        print(results)  # HERE

    # print(snapshot_dir)
    # print(os.path.isdir(snapshot_dir), '\n')  # True
    if os.path.isdir(snapshot_dir):  # 保存 model的目录： checkpoints/model_name/

        def check_ckpt_valid(ckpt_name):  # 保证 checkpoints/model_name 下每个文件名都valid（3），还有一个checkpoint文件保存all model checkpoint paths
            is_valid = False
            ckpt = ''
            if ckpt_name.startswith('model.ckpt-'):
                ckpt_name_splits = ckpt_name.split('.')
                ckpt = str(ckpt_name_splits[0]) + '.' + str(ckpt_name_splits[1])
                ckpt_path = os.path.join(snapshot_dir, ckpt)
                if os.path.exists(ckpt_path + '.index') and os.path.exists(ckpt_path + '.meta') and \
                        os.path.exists(ckpt_path + '.data-00000-of-00001'):
                    is_valid = True

            return is_valid, ckpt

        def scan_psnr_folder():
            tested_ckpt_in_psnr_sets = set()
            for test_psnr in os.listdir(psnr_dir):  # psnr_dir: the directory to save psnrs results in testing.
                tested_ckpt_in_psnr_sets.add(test_psnr)
            return tested_ckpt_in_psnr_sets  # a set

        def scan_model_folder():
            saved_models = set()
            for ckpt_name in os.listdir(snapshot_dir):  # snapshot_dir = checkpoints/model_name
                # print('ckpt_name: ', ckpt_name)  # model.ckpt-1000.data-00000-of-0000; .index; .meta; checkpoint
                is_valid, ckpt = check_ckpt_valid(ckpt_name)
                if is_valid:
                    saved_models.add(ckpt)
            return saved_models

        tested_ckpt_sets = scan_psnr_folder()  # psnr文件夹下:已经测试过的model psnr results

        # ######## INFERENCE
        while True:
            all_model_ckpts = scan_model_folder()  # checkpoints文件夹下:train保存的所有model
            print('saved_models:', all_model_ckpts)  # saved_models: {'model.ckpt-1000'}
            new_model_ckpts = all_model_ckpts - tested_ckpt_sets
            # print('new_model_ckpts:', new_model_ckpts)

            for ckpt_name in new_model_ckpts:
                ckpt = os.path.join(snapshot_dir, ckpt_name)
                inference_func(ckpt, dataset_name, evaluate_name)

                tested_ckpt_sets.add(ckpt_name)

            print('waiting for models...')
            evaluate.evaluate(evaluate_name, psnr_dir)  # psnr_dir就是evaluate函数的save_file，也就是计算 auc 时用到的loss_file
            time.sleep(60)
    else:  # snapshot_dir 不是目录，是 model.ckpt-xxx，直接inference
        inference_func(snapshot_dir, dataset_name, evaluate_name)
