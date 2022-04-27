import os
import h5py
import torch.utils.data as data
import numpy as np
import random
import cv2
from torchvision import transforms as tfs


class DataSet_HDF5(data.Dataset):
    def __init__(self, file_path, use_hflip=False, use_vflip=False):
        super(DataSet_HDF5, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.SR = hf.get("SR")
        self.LR = hf.get("LR")
        self.use_hflip = use_hflip
        self.use_vflip = use_vflip
        # hf.close()

    def __getitem__(self, index):
        LR_img = self.LR[index, :, :, :]
        SR_img = self.SR[index, :, :, :]
        LR_img = LR_img.astype(np.float32) / 255
        SR_img = SR_img.astype(np.float32) / 255
        LR_img = np.clip(LR_img, 0, 1)
        SR_img = np.clip(SR_img, 0, 1)

        # 随机翻转
        if self.use_hflip and random.randint(0, 1) == 1:
            LR_img = np.flip(LR_img, 0)
            SR_img = np.flip(SR_img, 0)
        if self.use_vflip and random.randint(0, 1) == 1:
            LR_img = np.flip(LR_img, 1)
            SR_img = np.flip(SR_img, 1)

        LR_img = cv2.resize(LR_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        LR_img = LR_img.transpose((2, 0, 1))
        SR_img = SR_img.transpose((2, 0, 1))

        return LR_img.copy(), SR_img.copy()

    def __len__(self):
        return self.SR.shape[0]
