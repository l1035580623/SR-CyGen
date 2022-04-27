import argparse
import os

import wandb
import yaml
import random

import torch
import cv2
import datetime
import math

from utils.logs import MessageLogger

parser = argparse.ArgumentParser(description="SR-CyGen Train")


parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--is_load_model', type=bool, default=False)
parser.add_argument('--opt', type=str, default="options/train_SR_CyGen.yml")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parser.parse_args()
    # get opt from yaml file
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    # set random seed
    if opt['random_seed'] is None:
        seed = random.randint(1, 10000)
    else:
        seed = opt['random_seed']
    print("Random Seed is:", seed)
    setup_seed(seed)
    # set device
    args.cuda = torch.cuda.is_available() and opt['gpu'] >= 0
    device = torch.device('cuda:' + str(opt['gpu']) if args.cuda else 'cpu')
    torch.cuda.set_device(device)

    # mkdir for experiment
    args.save = opt['save_path'] + "/" + opt['name']
    if os.path.exists(args.save):
        args.save += str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_')
    os.makedirs(args.save)

    logger = MessageLogger(opt, args)
    logger.print_log("This is a demo test for tensorboard and wandb")

    for i in range(100):
        loss_mse = random.random() * math.exp(-0.1 * i)
        loss_cm_1 = random.random() * math.exp(-0.1 * i)
        loss_cm_2 = random.random() * math.exp(-0.1 * i)
        loss = loss_mse + loss_cm_1 + loss_cm_2
        log_dict = {'epoch': i,
                    'loss': loss,
                    'loss_mse': loss_mse,
                    'loss_cm_1': loss_cm_1,
                    'loss_cm_2': loss_cm_2}
        log_message = "Epoch {:03d} | Loss {:.4f}\n " \
                      "Loss Item {:.4f} / {:.4f} / {:.4f}".format(i, loss, loss_mse, loss_cm_1, loss_cm_2)
        logger.print_log(log_message)
        logger.train_logger(log_dict, i + 1)
        img1 = cv2.imread("0.jpg", cv2.IMREAD_COLOR)
        img2 = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
        logger.img_logger([img1, img2], i + 1)




