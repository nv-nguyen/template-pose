# Adapted from https://github.com/monniert/dti-clustering/blob/b57a77d4c248b16b4b15d6509b6ec493c53257ef/src/utils/logger.py
import logging
import time
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import os
import shutil
from lib.utils import coerce_to_path_and_check_exist


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_info(s):
    print(TerminalColors.OKBLUE + "[" + get_time() + "] " + str(s) + TerminalColors.ENDC)


def print_warning(s):
    print(TerminalColors.WARNING + "[" + get_time() + "] WARN " + str(s) + TerminalColors.ENDC)


def print_error(s):
    print(TerminalColors.FAIL + "[" + get_time() + "] ERROR " + str(s) + TerminalColors.ENDC)


def get_logger(log_dir, name):
    log_dir = coerce_to_path_and_check_exist(log_dir)
    logger = logging.getLogger(name)
    file_path = log_dir / "{}.log".format(name)
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def print_info(s):
    print(TerminalColors.OKBLUE + "[" + get_time() + "] " + str(s) + TerminalColors.ENDC)


def print_and_log_info(logger, string):
    logger.info(string)


def create_tensorboard_logger(logdir=None):
    # assert os.path.exists(logdir), 'Log file dir is not existed.'
    ensure_dir(logdir)

    log_path = os.path.join(logdir, 'tensorboard',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class TensorboardLogger(object):

    def __init__(self, logdir=None):
        super(TensorboardLogger, self).__init__()
        self.logger = create_tensorboard_logger(logdir)

    def add_scalar_dict(self, tag, scalar_dict, it):
        """
        :param tag: str, train, eval or test.
        :param scalar_dict:
                type: dict
                {'scalar name', scalar value, ...}
        :param it: global step
        """

        assert isinstance(scalar_dict, dict), 'scalar dict must be dict type.'
        for k, v in scalar_dict.items():
            self.logger.add_scalar(tag + '/' + k, v, it)

    def add_scalar_dict_list(self, tag, scalar_dict_list, it):
        """
        :param scalar_dict_list:
        :param tag: str, it generally is 'trainval'
        :param scalars_list:
                type: list
                [{'scalar name': scalar value, ...}, ]
        :param it: global step
        """

        assert isinstance(scalar_dict_list, list), 'scalars list must be list type.'
        for k, v in enumerate(scalar_dict_list):
            self.logger.add_scalars(tag, v, it)

    def add_img(self, tag, img, it):
        self.logger.add_image(tag, img, it)  # ,, dataformats='HWC'

    def add_imgs(self, tag, img, it):
        self.logger.add_images(tag, img, it)

    def close(self):
        self.logger.close()

    def add_hist(self, tag, model, it):
        # https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
        model_dict = model.state_dict()  # current model dict
        for k, v in model_dict.items():
            if "img_encoder" in k and "conv" in k:
                if k == "img_encoder.conv1.weight":
                    self.logger.add_histogram(os.path.join(tag, k), v, it)
                else:
                    layer_name = k.split('.')[-4]
                    conv_name = k.split('.')[-2]
                    self.logger.add_histogram(os.path.join(tag, layer_name, conv_name), v, it)
            elif ("classifier" in k or "pose" in k) and ("weight" in k or "bias" in k):
                self.logger.add_histogram(os.path.join(tag, k), v, it)

    def add_grad_ratio(self, tag, model, it):
        '''
            for debug
            :return:
            '''
        gradsum = 0
        gradmax = 0
        datasum = 0
        layercnt = 0
        for param in model.parameters():
            if param.grad is not None:
                if param.grad.abs().max() > gradmax:
                    gradmax = param.grad.abs().max()
                grad = param.grad.abs().mean()
                data = param.data.abs().mean()
                # print(grad)
                gradsum += grad
                datasum += data
                layercnt += 1
        gradsum /= layercnt
        datasum /= layercnt
        gradmax, gradsum, datasum = float(gradmax), float(gradsum), float(datasum)
        self.add_scalar_dict_list(tag, [{'gradmax': gradmax}], it)
        self.add_scalar_dict_list(tag, [{'gradsum': gradsum}], it)
        self.add_scalar_dict_list(tag, [{'datasum': datasum}], it)


if __name__ == '__main__':
    logger = TensorboardLogger(logdir="./logger/")
