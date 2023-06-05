import os
import argparse
import configparser
'''
设置常量
'''

def get_dir(directory):
    """
    get the directory, if no such directory, then make it.

    @param directory: The new directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory

'''argparse 模块是 Python 内置的一个用于命令行选项options与参数arguments解析的模块'''
def parser_args():
    # 1.创建一个解析器
    parser = argparse.ArgumentParser(description='Options to run the network.')
    # 2.调用add_argument()添加参数
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the device id of gpu.') # type:命令行参数应该被转换成的类型
    parser.add_argument('-i', '--iters', type=int, default=1,
                        help='set the number of iterations, default is 1')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='set the batch size, default is 4.')
    parser.add_argument('--num_his', type=int, default=4,
                        help='set the time steps, default is 4.')  # 将数据reshape成[samples, time_steps, features], num_his即文中的t（用连续的 t 帧来预测第 t+1 帧）

    parser.add_argument('-d', '--dataset', type=str,
                        help='the name of dataset.')
    parser.add_argument('--train_folder', type=str, default='',
                        help='set the training folder path.')
    parser.add_argument('--test_folder', type=str, default='',
                        help='set the testing folder path.')

    parser.add_argument('--config', type=str, default='training_hyper_params/hyper_params.ini',
                        help='the path of training_hyper_params, default is training_hyper_params/hyper_params.ini')

    parser.add_argument('--snapshot_dir', type=str, default='',
                        help='if it is folder, then it is the directory to save models, '
                             'if it is a specific model.ckpt-xxx, then the system will load it for testing.')
    parser.add_argument('--summary_dir', type=str, default='', help='the directory to save summaries.')
    parser.add_argument('--psnr_dir', type=str, default='', help='the directory to save psnrs results in testing.')

    parser.add_argument('--evaluate', type=str, default='compute_auc',
                        help='the evaluation metric, default is compute_auc')
    # 3.调用parse_args()解析添加的参数：检查命令行，把每个参数转换为适当的类型然后调用相应的操作
    return parser.parse_args()


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError('const name {} is not all uppercase'.format(name))  # 常量名称不全是大写的

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str


args = parser_args()
const = Const()  # 创建一个Const()实例/对象

# 将从命令行解析得到的参数 赋值给常量对象const
# inputs constants
const.DATASET = args.dataset
const.TRAIN_FOLDER = args.train_folder
const.TEST_FOLDER = args.test_folder

const.GPU = args.gpu

const.BATCH_SIZE = args.batch
const.NUM_HIS = args.num_his
const.ITERATIONS = args.iters

const.EVALUATE = args.evaluate

# network constants
const.HEIGHT = 256
const.WIDTH = 256
const.FLOWNET_CHECKPOINT = 'checkpoints/pretrains/flownet-SD.ckpt-0'  # flownet-SD.ckpt-0
const.FLOW_HEIGHT = 384
const.FLOW_WIDTH = 512

# set training hyper-parameters of different datasets
config = configparser.ConfigParser()  # configparser:读取配置文件的包
assert config.read(args.config)  # 读取配置文件

# for lp loss. (e.g, 1 or 2 for l1 and l2 loss, respectively)
const.L_NUM = config.getint(const.DATASET, 'L_NUM')
# the power to which each gradient term is raised in GDL loss
const.ALPHA_NUM = config.getint(const.DATASET, 'ALPHA_NUM')
# the percentage of the adversarial loss to use in the combined loss
const.LAM_ADV = config.getfloat(const.DATASET, 'LAM_ADV')
# the percentage of the lp loss to use in the combined loss
const.LAM_LP = config.getfloat(const.DATASET, 'LAM_LP')
# the percentage of the GDL loss to use in the combined loss
const.LAM_GDL = config.getfloat(const.DATASET, 'LAM_GDL')
# the percentage of the different frame loss
const.LAM_FLOW = config.getfloat(const.DATASET, 'LAM_FLOW')

# Learning rate of generator
const.LRATE_G = eval(config.get(const.DATASET, 'LRATE_G'))
const.LRATE_G_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_G_BOUNDARIES'))

# Learning rate of discriminator
const.LRATE_D = eval(config.get(const.DATASET, 'LRATE_D'))
const.LRATE_D_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_D_BOUNDARIES'))

# 从保存的每个模型的名称 可以知道其参数
const.SAVE_DIR = '{dataset}_l{L_NUM}_alpha{ALPHA_NUM}_lp{LAM_LP}_' \
                 'adv{LAM_ADV}_gdl{LAM_GDL}_flow{LAM_FLOW}'.format(dataset=const.DATASET, L_NUM=const.L_NUM, ALPHA_NUM=const.ALPHA_NUM,
                                                                      LAM_LP=const.LAM_LP,
                                                                      LAM_ADV=const.LAM_ADV,
                                                                      LAM_GDL=const.LAM_GDL,
                                                                      LAM_FLOW=const.LAM_FLOW)
# const.SAVE_DIR = '{dataset}___l{L_NUM}_alpha{ALPHA_NUM}_lp{LAM_LP}_adv{LAM_ADV}_'.format(dataset=const.DATASET, L_NUM=const.L_NUM, ALPHA_NUM=const.ALPHA_NUM,
#                                                                       LAM_LP=const.LAM_LP,
#                                                                       # LAM_GDL=const.LAM_GDL
#                                                                       LAM_ADV=const.LAM_ADV
#                                                                       # LAM_FLOW=const.LAM_FLOW
#                                                                     )

# ### checkpoints
# -- testing 若命令行有指定checkpoints directory，说明当前是在测试，需要用到 trained model
if args.snapshot_dir:
    # if the snapshot_dir is model.ckpt-xxx, which means it is the single model for testing.
    if os.path.exists(args.snapshot_dir + '.meta') or os.path.exists(args.snapshot_dir + '.data-00000-of-00001') or \
            os.path.exists(args.snapshot_dir + '.index'):
        # checkpoint文件列出了所有保存的模型，由三个文件组成:
        # .meta保存了 Tensorflow 计算图的结构，可以简单理解为神经网络的网络结构
        # .index 和 .data-*****-of-***** 保存了所有变量的取值
        const.SNAPSHOT_DIR = args.snapshot_dir
        print(const.SNAPSHOT_DIR)
    else:  # 是一个目录(checkpoint文件夹)，保存了一个目录下所有的模型文件列表
        const.SNAPSHOT_DIR = get_dir(os.path.join('checkpoints', const.SAVE_DIR + '_' + args.snapshot_dir))

# -- training 命令行没有指定checkpoints directory，表示当前正在训练模型，模型需要被保存下来
else:
    const.SNAPSHOT_DIR = get_dir(os.path.join('checkpoints', const.SAVE_DIR))

# ### summary
if args.summary_dir:
    const.SUMMARY_DIR = get_dir(os.path.join('summary', const.SAVE_DIR + '_' + args.summary_dir))
else:
    const.SUMMARY_DIR = get_dir(os.path.join('summary', const.SAVE_DIR))

# ### PSNR
if args.psnr_dir:
    const.PSNR_DIR = get_dir(os.path.join('psnrs', const.SAVE_DIR + '_' + args.psnr_dir))
else:
    const.PSNR_DIR = get_dir(os.path.join('psnrs', const.SAVE_DIR))


