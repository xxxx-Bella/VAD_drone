import numpy as np
import scipy.io as scio
import os
import argparse
import pickle
from sklearn import metrics
import json
'''
模型效果评估
'''

# data folder contain all datasets, such as ped1, ped2, avenue, shanghaitech, etc
DATA_DIR = '../Data/Data'
# hostname = socket.gethostname()
# if hostname == 'dl-T8520-G10':  # 119
#     DATA_DIR = '/home/liuwen/ssd/datasets'
# elif hostname == 'admin' or hostname == 'compute101' or hostname == 'compute113' or hostname == 'compute106' \
#         or hostname == 'compute107' or hostname == 'compute114':   # node02
#     DATA_DIR = '/home/luowx/liuwen/datasets'
# elif hostname == 'gpu13' or 'gpu14':
#     DATA_DIR = '/public/home/gaoshenghua/liuwen/datasets'
# else:
#     # raise NotImplementedError('Not found this machine {}!'.format(hostname))
#     DATA_DIR = '../Data'


# normalize scores in each sub video
NORMALIZE = True

# number of history frames, since in prediction based method, the first 4 frames can not be predicted, so that
# the first 4 frames are undecidable, we just ignore the first 4 frames(0,1,2,3)
DECIDABLE_IDX = 4


def set_data_dir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir


def parser_args():
    parser = argparse.ArgumentParser(description='evaluating the model, computing the roc/auc.')

    parser.add_argument('-f', '--file', type=str, help='the path of loss file.')
    parser.add_argument('-t', '--type', type=str, default='compute_auc',
                        help='the type of evaluation, choosing type is: plot_roc, compute_auc, '
                             'test_func\n, the default type is compute_auc')
    return parser.parse_args()


class RecordResult(object):
    def __init__(self, dataset=None, loss_file=None, auc=-np.inf, precious=-np.inf, recall=-np.inf, f1=-np.inf):  # fpr=None, tpr=None,
        # self.fpr = fpr
        # self.tpr = tpr
        self.auc = auc
        self.precious = precious
        self.recall = recall
        self.f1 = f1
        self.dataset = dataset
        self.loss_file = loss_file  # psnrs/..

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}, precious = {}, recall = {}, f1 = {}'  \
            .format(self.dataset, self.loss_file, self.auc, self.precious, self.recall, self.f1)
        # loss file = psnrs/ped2_l2_alpha1_lp1.0_/model.ckpt-1000


class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED1 = 'ped1'
    PED1_PIXEL_SUBSET = 'ped1_pixel_subset'
    PED2 = 'ped2'
    DRONE_ANOMALY = 'Drone_Anomaly'
    CAMPUS = 'Campus'
    CAMPUS_DRONE = 'Campus_Drone'
    SHANGHAITECH_CAMPUS = 'shanghaitech_Campus'
    ENTRANCE = 'enter'
    EXIT = 'exit'
    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/testing/test_frame_mask')
    TOY_DATA = 'toydata'
    TOY_DATA_LABEL_PATH = os.path.join(DATA_DIR, TOY_DATA, 'toydata.json')

    NAME_MAT_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
        PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat'),
        DRONE_ANOMALY: os.path.join(DATA_DIR, 'Bike_Roundabout/sequence1/Bike_Roundabout.mat'),
        CAMPUS: os.path.join(DATA_DIR, 'Campus/playground_compact/playground.mat'),
        CAMPUS_DRONE: os.path.join(DATA_DIR, 'Campus_Drone/playground_drone_compact/playground_drone.mat'),
        SHANGHAITECH_CAMPUS: os.path.join(DATA_DIR, 'shanghaitech/shanghaitech_Campus.mat'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/enter.mat'),
        EXIT: os.path.join(DATA_DIR, 'exit/exit.mat')
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/testing/frames'),
        PED1: os.path.join(DATA_DIR, 'ped1/testing/frames'),
        PED2: os.path.join(DATA_DIR, 'ped2/testing/frames'),
        DRONE_ANOMALY: os.path.join(DATA_DIR, 'Bike_Roundabout/sequence1/test'),
        CAMPUS: os.path.join(DATA_DIR, 'Campus/playground_compact/test'),
        CAMPUS_DRONE: os.path.join(DATA_DIR, 'Campus_Drone/playground_drone_compact/test'),
        SHANGHAITECH_CAMPUS: os.path.join(DATA_DIR, 'shanghaitech/testing/frames'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/testing/frames'),
        EXIT: os.path.join(DATA_DIR, 'exit/testing/frames')
    }

    def __init__(self, mapping_json=None):
        """
        Initial a ground truth loader, which loads the ground truth with given dataset name.

        :param mapping_json: the mapping from dataset name to the path of ground truth.
        """

        if mapping_json is not None:
            with open(mapping_json, 'rb') as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING  # a dict, paths of different '.mat'

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        elif dataset == GroundTruthLoader.TOY_DATA:
            gt = self.__load_toydata_gt()
        else:
            gt = self.__load_ucsd_avenue_DA_C_gt(dataset)
        # print('__call__ ### gt:', gt)
        return gt

    def __load_ucsd_avenue_DA_C_gt(self, dataset):  # ucsd ped1, ucsd ped2
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        # 加载.mat文件
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']  # squeeze_me=True: 在加载时将所有维数为1的维度去掉
        # np.set_printoptions(threshold=np.inf)
        # print('abnormal_events: ', abnormal_events)

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]  # /Bike_Roundabout/sequence1/test
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()
        # print('video_list:', video_list)
        # print('ground true of {}, num_video = {}'.format(dataset, num_video), '\n')  # i.e. DRONE_ANOMALY: Bike_Roundabout, num_video=7
        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):  # for each video's label
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            # np.set_printoptions(threshold=np.inf)
            # print('i={}, sub_abnormal_events.ndim={}, \n {}'.format(i, sub_abnormal_events.ndim, sub_abnormal_events))
            if sub_abnormal_events.ndim == 1:  # reshape 使其具有两个维度，以便可以正确索引数组
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            # print('sub_abnormal_events.shape={}'.format(sub_abnormal_events.shape))
            _, num_abnormal = sub_abnormal_events.shape  # (1200, 1)

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt  # load (DA or Campus) gt: abnormal_events; original gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video)))

        return gt

    @staticmethod
    def __load_toydata_gt():
        with open(GroundTruthLoader.TOY_DATA_LABEL_PATH, 'r') as gt_file:
            gt_dict = json.load(gt_file)

        gt = []
        for video, video_info in gt_dict.items():
            length = video_info['length']
            video_gt = np.zeros((length,), dtype=np.uint8)
            sub_gt = np.array(np.matrix(video_info['gt']))

            for anomaly in sub_gt:
                start = anomaly[0]
                end = anomaly[1] + 1
                video_gt[start: end] = 1
            gt.append(video_gt)
        return gt

    @staticmethod
    def get_pixel_masks_file_list(dataset):
        # pixel mask folder
        pixel_mask_folder = os.path.join(DATA_DIR, dataset, 'pixel_masks')
        pixel_mask_file_list = os.listdir(pixel_mask_folder)
        pixel_mask_file_list.sort()

        # get all testing videos
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        # get all testing video names with pixel masks
        pixel_video_ids = []
        ids = 0
        for pixel_mask_name in pixel_mask_file_list:
            while ids < len(video_list):
                if video_list[ids] + '.npy' == pixel_mask_name:
                    pixel_video_ids.append(ids)
                    ids += 1
                    break
                else:
                    ids += 1

        assert len(pixel_video_ids) == len(pixel_mask_file_list)

        for i in range(len(pixel_mask_file_list)):
            pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])

        return pixel_mask_file_list, pixel_video_ids


def load_psnr_gt(loss_file):
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)

    dataset = results['dataset']
    psnr_records = results['psnr']
    # print('predict \n', 'dataset: {}, psnr_records: {}, length = {}'.format(dataset, psnr_records, len(psnr_records)))

    num_videos = len(psnr_records)
    # print('num_videos =', num_videos)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)
    np.set_printoptions(threshold=np.inf)
    # print('gt \n', 'gt: {}, length = {}'.format(gt, len(gt)))

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnr_records, gt


def load_psnr_gt_flow(loss_file):
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)

    dataset = results['dataset']
    psnrs = results['psnr']
    flows = results['flow']

    num_videos = len(psnrs)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnrs, flows, gt


def load_psnr(loss_file):
    """
    load image psnr or optical flow psnr.
    :param loss_file: loss file path
    :return:
    """
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)
    psnrs = results['psnr']
    return psnrs


def get_scores_labels(loss_file):
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file)
    # print('dataset:{}, psnr_records:{}, gt:{}'.format(dataset, len(psnr_records), len(gt)))
    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance

        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:]), axis=0)
    # print('scores:', scores)
    # print('labels:', labels)
    return dataset, scores, labels


def precision_recall_f1_auc(loss_file):
    print('--- compute precision_recall_f1_auc')
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)  # 返回指定文件夹 包含的 文件或文件夹的名字的列表
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file)
        # P=TP/(TP+FP); R=TP/(TP+FN)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        avr_p = np.average(precision)
        avr_r = np.average(recall)
        # F1=(2∗P∗R)/(P+R) 模型准确率和召回率的加权平均，最大值1，最小值0，值越大模型越好
        # f1 = metrics.f1_score(labels, scores, pos_label=0)
        # f1 = (2*precision*recall) / (precision+recall)

        avr_f1 = (2*avr_p*avr_r) / (avr_p+avr_r)
        # avr_f1 = np.average(f1)
        # print('len =', len(precision), len(recall), len(f1))
        # auc = metrics.auc(recall, precision)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)  # pos_label指定正例标签
        auc = metrics.auc(fpr, tpr)

        # update results
        results = RecordResult(dataset, sub_loss_file, auc, avr_p, avr_r, avr_f1)  # sub_loss_file: / -200 or -1000

        if optimal_results < results:
            optimal_results = results

        # if os.path.isdir(loss_file):
        #     print(results)
    print('##### `precision_recall_f1_auc` optimal result and model : {} \n'.format(optimal_results))
    return results  # optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


# def compute_eer(loss_file):
#     print('--- compute eer')
#     if not os.path.isdir(loss_file):
#         loss_file_list = [loss_file]
#     else:
#         loss_file_list = os.listdir(loss_file)
#         loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

#     optimal_results = RecordResult(auc=np.inf)
#     for sub_loss_file in loss_file_list:
#         dataset, scores, labels = get_scores_labels(sub_loss_file)
#         fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
#         eer = cal_eer(fpr, tpr)

#         results = RecordResult(fpr, tpr, eer, dataset, sub_loss_file)

#         if optimal_results > results:
#             optimal_results = results

#         if os.path.isdir(loss_file):
#             print(results)
#     print('##### `eer` optimal result and model : {} \n'.format(optimal_results))
#     return optimal_results


def compute_auc(loss_file):
    # print('--- compute auc')
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:  # psnrs/Drone_Anomaly_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0/model.ckpt-30
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)  # 7

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]
            # print('i = {}, distance: {}, num = {}'.format(i, distance, len(distance)))

            # has_nan = np.isnan(distance).any()  # 检查是否包含 NaN
            # has_inf = np.isinf(distance).any()  # 检查是否包含 Inf
            # print('has_nan:', has_nan, '\n')

            if NORMALIZE:  # distances = (distance - min) / (max - min)
                distance -= distance.min()
                distance /= distance.max()
                # distance = 1 - distance

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)  # y_score
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)  # y_true
        # print('y_score:{} , length = {}'.format(scores, len(scores)))
        # print('y_true:{} , length = {}'.format(labels, len(labels)))

        # tpr = TP/(TP+FN), fpr = FP/(FP+TN)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)  # pos_label指定正例标签
        auc = metrics.auc(fpr, tpr)
        # auc = metrics.roc_auc_score(labels, scores)

        results = RecordResult(dataset, sub_loss_file, auc)  # fpr, tpr

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    # print('##### `auc` optimal result and model : {} \n'.format(optimal_results))
    return optimal_results


def average_psnr(loss_file):
    print('--- compute average_psnr')
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    max_avg_psnr = -np.inf
    max_file = ''
    for file in loss_file_list:
        psnr_records = load_psnr(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        avg_psnr = np.mean(psnr_records)
        if max_avg_psnr < avg_psnr:
            max_avg_psnr = avg_psnr
            max_file = file
        print('{}, average psnr = {}'.format(file, avg_psnr))

    print('max average psnr file: {}, psnr = {} \n'.format(max_file, max_avg_psnr))


def calculate_psnr(loss_file):
    print('--- compute psnr')
    # optical_result = compute_auc(loss_file)
    # print('##### optimal result and model : {}'.format(optical_result))

    mean_psnr = []
    for file in os.listdir(loss_file):
        file = os.path.join(loss_file, file)
        dataset, psnr_records, gt = load_psnr_gt(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        gt = np.concatenate(gt, axis=0)

        mean_normal_psnr = np.mean(psnr_records[gt == 0])
        mean_abnormal_psnr = np.mean(psnr_records[gt == 1])
        mean = np.mean(psnr_records)
        print('mean normal psrn = {}, mean abnormal psrn = {}, mean = {}'.format(
            mean_normal_psnr,
            mean_abnormal_psnr,
            mean)
        )
        mean_psnr.append(mean)
    print('max mean psnr = {} \n'.format(np.max(mean_psnr)))


def calculate_score(loss_file):
    print('--- compute score')
    if not os.path.isdir(loss_file):
        loss_file_path = loss_file
    else:
        optical_result = compute_auc(loss_file)
        loss_file_path = optical_result.loss_file
        # print('##### optimal result and model : {}'.format(optical_result))
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file_path)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        distance = (distance - distance.min()) / (distance.max() - distance.min())

        scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)

    mean_normal_scores = np.mean(scores[labels == 0])
    mean_abnormal_scores = np.mean(scores[labels == 1])
    print('mean normal scores = {}, mean abnormal scores = {}, '
          'delta = {} \n'.format(mean_normal_scores, mean_abnormal_scores, mean_normal_scores - mean_abnormal_scores))


def test_func(*args):
    # simulate testing on CUHK AVENUE dataset
    dataset = GroundTruthLoader.AVENUE

    # load the ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    num_videos = len(gt)

    simulated_results = {
        'dataset': dataset,
        'psnr': []
    }

    simulated_psnr = []
    for i in range(num_videos):
        sub_video_length = gt[i].shape[0]
        simulated_psnr.append(np.random.random(size=sub_video_length))

    simulated_results['psnr'] = simulated_psnr

    # writing to file, 'generated_loss.bin'
    with open('generated_loss.bin', 'wb') as writer:
        pickle.dump(simulated_results, writer, pickle.HIGHEST_PROTOCOL)

    print(file_path.name)
    result = compute_auc(file_path.name)

    print('optimal = {}'.format(result))


eval_type_function = {
    'compute_auc': compute_auc,
    # 'compute_eer': compute_eer,
    'precision_recall_f1_auc': precision_recall_f1_auc,
    'calculate_psnr': calculate_psnr,
    'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr
}


def evaluate(eval_type, save_file):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())
    eval_func = eval_type_function[eval_type]  # eval_type='compute_auc', eval_func=compute_auc
    optimal_results = eval_func(save_file)  # 调用 compute_auc 函数，save_file就是loss_file: psnrs/..
    return optimal_results


if __name__ == '__main__':
    # 已保存 inference 时计算得到的 psnrs，直接计算 AUC
    # 'python evaluate.py'
    eval_type = ['precision_recall_f1_auc', 'calculate_psnr', 'calculate_score']  # ['precision_recall_f1_auc', 'average_psnr']
    file_path = 'psnrs/Campus_Drone_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0'  # Campus_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0' // Campus_Drone_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0' ped2_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0' // Drone_Anomaly_l2_alpha1_lp1.0_adv0.05_gdl1.0_flow2.0' // 
    for et in eval_type:
        evaluate(et, file_path)

#     args = parser_args()

#     eval_type = args.type
#     file_path = args.file

#     print('Evaluate type = {}'.format(eval_type))
#     print('File path = {}'.format(file_path))

#     if eval_type == 'test_func':
#         test_func()
#     else:
#         evaluate(eval_type, file_path)
