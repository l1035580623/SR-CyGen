import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from os.path import join
from shutil import copyfile

import random
import yaml
import time
import datetime
import cv2

from datasets.dataset_hdf5 import DataSet_HDF5
from models.SR_CyGen_DBPN import SR_CyGen_DBPN
from models.SR_CyGen_RRDB import SR_CyGen_RRDB
# from models.SR_CyGen_RRDB import SR_CyGen_ablation as SR_CyGen
# from models.SR_CyGen_DBPN import SR_CyGen_ablation as SR_CyGen_DBPN
from utils.utils import *
from utils.logs import MessageLogger

parser = argparse.ArgumentParser(description="SR-CyGen Test")

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--opt', type=str, default="options/test_SR_CyGen.yml")


def test_pipeline(test_sets, model, test_recoder, logger, args, opt):
    model.eval()
    test_recoder.reset()
    use_resize = True if opt['network']['type'] == 'DBPN' else False
    save_img_n = 0
    save_img_list = []

    for j in range(len(test_sets)):
        # get dataloader
        test_set_path = join(opt['datasets']['dataroot'], test_sets[j])
        test_set = DataSet_HDF5(test_set_path, use_hflip=False, use_vflip=False, use_resize=use_resize)
        testloader = DataLoader(dataset=test_set, batch_size=opt['batch_size'], shuffle=False, num_workers=0)

        test_recoder.create()  # create new recoder for this dataloader
        for k, (LR, SR) in enumerate(testloader):
            # generate
            start_time = time.time()
            LR = LR.to(device)
            SR = SR.to(device)
            with torch.no_grad():
                SR_pred = model.generate(LR, n_iter1=opt['test']['generate_n_1'], n_iter2=opt['test']['generate_n_2'])
            end_time = time.time()

            # calc psnr and ssim
            psnr_avg, ssim_avg = 0.0, 0.0
            n_img = LR.shape[0]
            for i in range(n_img):
                LR_index = LR[i:i + 1, :, :, :]
                SR_index = SR[i:i + 1, :, :, :]
                SR_pred_index = SR_pred[-1][i:i + 1, :, :, :]

                LR_img = tensor2img(LR_index)
                if not use_resize:
                    LR_img = bicubic(LR_img, upscale=4.0)
                SR_img = tensor2img(SR_index)
                SR_pred_img = tensor2img(SR_pred_index)

                psnr, ssim = calc_psnr_ssim(SR_pred_img, SR_img)
                psnr_avg += psnr
                ssim_avg += ssim
                log_message = ("Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
                    (end_time - start_time) / n_img, psnr, ssim))
                logger.print_log(log_message)
                # save images
                if save_img_n < opt['test']['save_num']:
                    # save img as : HR | SR | LR
                    if opt['test']['save_type'] == "contrast":
                        save_img = np.concatenate((SR_img, SR_pred_img, LR_img), 1)
                    # save img as : SR[0] | SR[1] |...| SR[n1-1]
                    elif opt['test']['save_type'] == "loop":
                        img_list = []
                        for l in range(len(SR_pred)):
                            SR_pred_img = tensor2img(SR_pred[l])
                            psnr, ssim = calc_psnr_ssim(SR_pred_img, SR_img)
                            print(psnr, ssim)
                            img_list.append(SR_pred_img)
                        save_img = np.concatenate(img_list, 1)
                    else:
                        save_img = SR_pred_img
                    cv2.imwrite(args.save + "/generate" + "/img_{:04d}.jpg".format(save_img_n), save_img)
                    save_img_list.append(save_img)
                    save_img_n += 1
            # update recoder
            test_recoder.update({'time': (end_time - start_time) / n_img,
                                  'psnr': psnr_avg / n_img,
                                  'ssim': ssim_avg / n_img}, n_img)
        # print log message for this dataloader
        log_msg_dict = test_recoder.get_avg_now()
        log_message = ("Dataset {} | Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
            test_sets[j], log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
        logger.print_log(log_message)

    # print log message for this epoch
    log_msg_dict = test_recoder.get_avg_all()
    log_message = ("Testset | Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
        log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
    logger.print_log(log_message)
    logger.img_logger(save_img_list, 1)


def generate_HR_LR(test_sets, test_recoder, logger):
    test_recoder.reset()
    save_img_n = 0
    for j in range(len(test_sets)):
        # get dataloader
        test_set_path = join(opt['datasets']['dataroot'], test_sets[j])
        test_set = DataSet_HDF5(test_set_path, use_hflip=False, use_vflip=False)
        testloader = DataLoader(dataset=test_set, batch_size=opt['batch_size'], shuffle=False, num_workers=0)

        test_recoder.create()  # create new recoder for this dataloader
        for k, (LR, SR) in enumerate(testloader):
            # generate
            start_time = time.time()
            LR = LR.to(device)
            SR = SR.to(device)
            end_time = time.time()

            # calc psnr and ssim
            psnr_avg, ssim_avg = 0.0, 0.0
            n_img = LR.shape[0]
            for i in range(n_img):
                LR_index = LR[i:i + 1, :, :, :]
                SR_index = SR[i:i + 1, :, :, :]

                LR_img = tensor2img(LR_index)
                # LR_img = bicubic(LR_img, 4.0)
                SR_img = tensor2img(SR_index)

                # psnr, ssim = calc_psnr_ssim(LR_img, SR_img)
                psnr, ssim = 0, 0
                psnr_avg += psnr
                ssim_avg += ssim
                log_message = ("Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
                    (end_time - start_time) / n_img, psnr, ssim))
                logger.print_log(log_message)
                # save images
                cv2.imwrite(args.save + "/generate" + "/img_{:04d}.jpg".format(save_img_n), SR_img)
                # cv2.imwrite(args.save + "/generate" + "/img_{:04d}_SR.jpg".format(save_img_n), SR_img)
                save_img_n += 1
            # update recoder
            test_recoder.update({'time': (end_time - start_time) / n_img,
                                  'psnr': psnr_avg / n_img,
                                  'ssim': ssim_avg / n_img}, n_img)
        # print log message for this dataloader
        log_msg_dict = test_recoder.get_avg_now()
        log_message = ("Dataset {} | Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
            test_sets[j], log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
        logger.print_log(log_message)

    # print log message for this epoch
    log_msg_dict = test_recoder.get_avg_all()
    log_message = ("Testset | Time {:.4f} | PSNR {:.4f} | SSIM {:.4f}".format(
        log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
    logger.print_log(log_message)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    print("BEGIN!")
    # set device
    args.cuda = torch.cuda.is_available() and opt['gpu'] >= 0
    device = torch.device('cuda:' + str(opt['gpu']) if args.cuda else 'cpu')
    opt['device'] = device
    torch.cuda.set_device(device)

    # mkdir for experiment
    args.save = opt['save_path'] + "/" + opt['name']
    if os.path.exists(args.save):
        args.save += str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_')
    makedirs(args.save)
    makedirs(join(args.save, "generate"))

    # load checkpoint if exists
    if opt['network']['type'] == "RRDB":
        model = SR_CyGen_RRDB(opt).to(device)
    else:
        model = SR_CyGen_DBPN(opt).to(device)
    if opt['load_path'] is not None:
        print("load Model：" + opt['load_path'])
        model_checkpoint = torch.load(opt['load_path'], map_location=device)
        model.load_state_dict(model_checkpoint['model'])

    # set logger
    logger = MessageLogger(opt, args)
    logger.print_log("Log File successfully create in:" + args.save)
    logger.print_log(opt)
    logger.print_log(args)
    copyfile(args.opt, args.save + "/options.yml")

    test_sets = [x for x in sorted(os.listdir(opt['datasets']['dataroot'])) if is_hdf5_file(x)]
    test_recoder = MessageRecoder(dict().fromkeys(('time', 'psnr', 'ssim')))

    test_pipeline(test_sets=test_sets, model=model, test_recoder=test_recoder, logger=logger, args=args, opt=opt)
    # generate_HR_LR(test_sets=test_sets, test_recoder=test_recoder, logger=logger)

