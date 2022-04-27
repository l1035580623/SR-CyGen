import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from dataset_hdf5 import DataSet_HDF5
from SR_cygen_2 import SR_CyGen
from os.path import join

import random
import time
import datetime
import os
from torch.utils.data import DataLoader

import utils.utils as utils
from skimage.metrics import structural_similarity as ssim_calc
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from math import log10


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5"])


parser = argparse.ArgumentParser(description="SR-CyGen Train")

parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--cmtype', type=str, default="jacnorm_x")
parser.add_argument('--pxtype', type=str, default="nllhmarg")
parser.add_argument('--w_cm', type=float, default=0.1)
parser.add_argument('--w_px', type=float, default=1.)
parser.add_argument('--n_mc_cm', type=int, default=1)
parser.add_argument('--n_mc_px', type=int, default=1)
parser.add_argument('--n_mc_eval', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dataset', default="C:/MyDocuments/Dataset_SR/testset", type=str, help='Path of the training dataset(.hdf5)')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dim_z', type=int, default=128)
parser.add_argument('--save_path', type=str, default="C:/MyCode/code for BS/SR-CyGen\expm")
parser.add_argument('--model_path', type=str, default="C:/MyCode/code for BS/SR-CyGen/models/SR_CyGen_2_iter_073.pkl")
parser.add_argument('--is_load_model', type=bool, default=True)
parser.add_argument('--random_seed', type=int, default=99)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:' + str(args.gpu) if args.gpu >= 0 and args.cuda else 'cpu')
    torch.cuda.set_device(device)

    args.save = args.save_path + "/expm-" + str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_')
    utils.makedirs(args.save)
    utils.makedirs(args.save + "/generate")
    logger = utils.get_logger(args.save + "/logs")
    logger.info(args)

    # 加载模型
    model = SR_CyGen(args).to(device)
    # 设置optim
    if args.is_load_model:
        print("load Model：" + args.model_path)
        model_checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(model_checkpoint['model'])
        args.start_iters = model_checkpoint['epoch'] + 1
        print("start epoch is:" + str(args.start_iters))

    test_sets = [x for x in sorted(os.listdir(args.dataset)) if is_hdf5_file(x)]
    print(test_sets)

    psnr_meter = utils.RunningAverageMeter()
    ssim_meter = utils.RunningAverageMeter()

    model.eval()
    for j in range(len(test_sets)):
        # 读取第j个h5文件
        print("Testing folder is {}".format(join(args.dataset, test_sets[j])))
        test_set = DataSet_HDF5(join(args.dataset, test_sets[j]))
        testloader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(len(testloader))

        psnr_meter.create()
        ssim_meter.create()

        for k, (LR, SR) in enumerate(testloader):
            LR = LR.to(device)
            SR = SR.to(device)

            with torch.no_grad():
                # SR_pred = model.generate("gibbs", LR, LR_resize)
                SR_pred = model.generate("gibbs", LR, 1, 1)

            for i in range(args.batch_size):
                LR_index = LR[i:i + 1, :, :, :]
                SR_index = SR[i:i + 1, :, :, :]
                SR_pred_index = SR_pred[-1][i:i + 1, :, :, :]

                LR_img = tensor2img(LR_index)
                SR_img = tensor2img(SR_index)
                SR_pred_img = tensor2img(SR_pred_index)

                psnr = psnr_calc(SR_pred_img, SR_img, data_range=255.0)
                ssim = ssim_calc(SR_pred_img, SR_img, data_range=255.0, multichannel=True)
                psnr_LR = psnr_calc(LR_img, SR_img, data_range=255.0)
                ssim_LR = ssim_calc(LR_img, SR_img, data_range=255.0, multichannel=True)
                psnr_meter.update(psnr)
                ssim_meter.update(ssim)
                result = np.concatenate((SR_img, SR_pred_img, LR_img), 1)

                # img_list = []
                # for i in range(len(SR_pred)):
                #     SR_pred_img = tensor2img(SR_pred[i])
                #     psnr = psnr_calc(SR_pred_img, SR_img, data_range=255.0)
                #     ssim = ssim_calc(SR_pred_img, SR_img, data_range=255.0, multichannel=True)
                #     print(psnr, ssim)
                #     img_list.append(tensor2img(SR_pred[i]))
                # result = np.concatenate(img_list, 1)
                #
                # img_file = args.save + "/generate/{:04d}_{:02d}.jpg".format(k, i)
                # cv2.imwrite(img_file, result)

                log_message = ("img_file:{:04d}_{:02d}.jpg || PSNR:{:.4f}({:.4f}) || SSIM:{:.4f}({:.4f})"
                               .format(k, i, psnr, psnr_LR, ssim, ssim_LR))
                print(log_message)
                logger.info(log_message)

        log_message = ("Avg_PSNR:{:.4f} || Avg_SSIM:{:.4f}".format(psnr_meter.get_avg_now(), ssim_meter.get_avg_now()))
        print(log_message)
        logger.info(log_message)
    log_message = ("Avg_PSNR:{:.4f} || Avg_SSIM:{:.4f}".format(psnr_meter.get_avg_all(), ssim_meter.get_avg_all()))
    print(log_message)
    logger.info(log_message)







