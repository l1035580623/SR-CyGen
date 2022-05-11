import os
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5"])


def tensor2img(tensor):
    tensor = tensor.cpu().numpy().squeeze(0)
    tensor = tensor.transpose((1, 2, 0))
    tensor = tensor * 255.0
    tensor = tensor.astype("int")
    return tensor


def bicubic(img, upscale=4.0):
    img = img.astype(np.float32)
    img = cv2.resize(img, (int(img.shape[0] * upscale), int(img.shape[1] * upscale)), interpolation=cv2.INTER_CUBIC)
    img = img.astype("int")
    return img


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calc_psnr_ssim(img1, img2):
    psnr = psnr_calc(img1, img2, data_range=255.0)
    ssim = ssim_calc(img1, img2, data_range=255.0, multichannel=True)
    return psnr, ssim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.meter_now = None
        self.meter_list = []

    def create(self):
        self.meter_now = AverageMeter()
        self.meter_list.append(self.meter_now)

    def update(self, val, n=1):
        self.meter_now.update(val, n)

    def get_avg_now(self):
        return self.meter_now.avg

    def get_avg_all(self):
        sum = 0
        num = 0
        for meter in self.meter_list:
            sum += meter.sum
            num += meter.count
        return sum / num


class MessageRecoder(object):
    def __init__(self, recoder_dict):
        self.recoder_dict = recoder_dict
        for k in recoder_dict.keys():
            recoder_dict[k] = RunningAverageMeter()

    def reset(self):
        for meter in self.recoder_dict.values():
            meter.reset()

    def create(self):
        for meter in self.recoder_dict.values():
            meter.create()

    def update(self, value_dict, n=1):
        for k, v in value_dict.items():
            self.recoder_dict[k].update(v, n)

    def get_avg_now(self):
        avg_dict = {}
        for k, v in self.recoder_dict.items():
            avg_dict.update({k: v.get_avg_now()})
        return avg_dict

    def get_avg_all(self):
        avg_dict = {}
        for k, v in self.recoder_dict.items():
            avg_dict.update({k: v.get_avg_all()})
        return avg_dict


