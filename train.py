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
from models.SR_CyGen_RRDB import SR_CyGen_RRDB
from models.SR_CyGen_DBPN import SR_CyGen_DBPN
# from models.SR_CyGen_RRDB import SR_CyGen_ablation as SR_CyGen
# from models.SR_CyGen_DBPN import SR_CyGen_ablation as SR_CyGen_DBPN
from utils.utils import *
from utils.logs import MessageLogger


parser = argparse.ArgumentParser(description="SR-CyGen Train")


parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--is_load_model', type=bool, default=False)
parser.add_argument('--opt', type=str, default="options/train_SR_CyGen.yml")


def train_pipeline(epoch, train_sets, model, train_recoder, logger, optimizer, args, opt):
    model.train()
    train_recoder.reset()
    use_resize = True if opt['network']['type'] == 'DBPN' else False
    iter = 0
    for j in range(len(train_sets)):
        # get dataloader
        train_set_path = join(opt['datasets']['train']['dataroot'], train_sets[j])
        train_set = DataSet_HDF5(train_set_path, use_hflip=opt['datasets']['train']['use_hflip'],
                                 use_vflip=opt['datasets']['train']['use_vflip'],
                                 use_resize=use_resize)
        trainloader = DataLoader(dataset=train_set, batch_size=opt['batch_size'], shuffle=True, num_workers=0)

        train_recoder.create()  # create new recoder from this trainloader
        for k, (LR, HR) in enumerate(trainloader):
            # forward
            start_time = time.time()
            LR = LR.to(device)
            HR = HR.to(device)

            loss = model.getlosses(LR, HR)

            # backward
            optimizer.zero_grad()
            if epoch < opt['train']['warm_up']:
                loss[1].backward()
            else:
                loss[0].backward()
            optimizer.step()

            # update recoder
            train_recoder.update({'time': time.time() - start_time,
                                  'loss': loss[0].item(),
                                  'loss_mse': loss[1].item(),
                                  'loss_cm_1': loss[2].item(),
                                  'loss_cm_2': loss[3].item()}, LR.shape[0])
            iter += 1
            if iter % opt['logger']['print_freq'] == 0:
                print("{:.4f} | {:.4f} | {:.6f} | {:.6f}".format(loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()))
        # print log message for this trainloader
        log_msg_dict = train_recoder.get_avg_now()
        log_message = ("Epoch {:03d} | Dataset {} | Time {:.4f}\n" \
                       "Loss {:.4f} | Loss Item {:.4f} / {:.6f} / {:.6f}".format(
            epoch, train_sets[j], log_msg_dict['time'], log_msg_dict['loss'], log_msg_dict['loss_mse'],
            log_msg_dict['loss_cm_1'], log_msg_dict['loss_cm_2']))
        logger.print_log(log_message)

    # print log message for this epoch
    log_msg_dict = train_recoder.get_avg_all()
    log_message = ("Epoch {:03d} | Trainset | Time {:.4f}\n" \
                   "Loss {:.4f} | Loss Item {:.4f} / {:.6f} / {:.6f}".format(
        epoch, log_msg_dict['time'], log_msg_dict['loss'], log_msg_dict['loss_mse'],
        log_msg_dict['loss_cm_1'], log_msg_dict['loss_cm_2']))
    logger.print_log(log_message)
    logger.train_logger(log_msg_dict, epoch)


def valid_pipeline(epoch, valid_sets, model, valid_recoder, logger, args, opt):
    model.eval()
    valid_recoder.reset()
    use_resize = True if opt['network']['type'] == 'DBPN' else False
    save_img_n = 0
    save_img_list = []
    for j in range(len(valid_sets)):
        # get dataloader
        valid_set_path = join(opt['datasets']['valid']['dataroot'], valid_sets[j])
        valid_set = DataSet_HDF5(valid_set_path, use_hflip=False, use_vflip=False,
                                 use_resize=use_resize)
        validloader = DataLoader(dataset=valid_set, batch_size=opt['batch_size'], shuffle=True, num_workers=0)

        valid_recoder.create()  # create new recoder for this dataloader
        for k, (LR, SR) in enumerate(validloader):
            # generate
            start_time = time.time()
            LR = LR.to(device)
            SR = SR.to(device)
            with torch.no_grad():
                SR_pred = model.generate(LR, 6, 5)
            end_time = time.time()

            # calc psnr and ssim
            psnr_avg, ssim_avg = 0.0, 0.0
            n_img = LR.shape[0]
            for i in range(n_img):
                LR_index = LR[i:i + 1, :, :, :]
                SR_index = SR[i:i + 1, :, :, :]
                SR_pred_index = SR_pred[-1][i:i + 1, :, :, :]

                LR_img = tensor2img(LR_index)
                SR_img = tensor2img(SR_index)
                SR_pred_img = tensor2img(SR_pred_index)

                psnr, ssim = calc_psnr_ssim(SR_pred_img, SR_img)
                psnr_avg += psnr
                ssim_avg += ssim
                # save images
                if save_img_n < opt['logger']['save_img_per_val']:
                    if use_resize:
                        LR_img_resize = LR_img
                    else:
                        LR_img_resize = bicubic(LR_img, upscale=4.0)
                    save_img = np.concatenate((SR_img, SR_pred_img, LR_img_resize), 1)
                    save_img_list.append(save_img)
                    save_img_n += 1
            # update recoder
            valid_recoder.update({'time': (end_time - start_time) / n_img,
                                  'psnr': psnr_avg / n_img,
                                  'ssim': ssim_avg / n_img}, n_img)
        # print log message for this dataloader
        log_msg_dict = valid_recoder.get_avg_now()
        log_message = ("Epoch {:03d} | Dataset {} | Time {:.4f}\n" \
                       "PSNR {:.4f} | SSIM {:.4f}".format(
            epoch, train_sets[j], log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
        logger.print_log(log_message)

    # print log message for this epoch
    log_msg_dict = valid_recoder.get_avg_all()
    log_message = ("Epoch {:03d} | Validset | Time {:.4f}\n" \
                   "PSNR {:.4f} | SSIM {:.4f}".format(
        epoch, log_msg_dict['time'], log_msg_dict['psnr'], log_msg_dict['ssim']))
    logger.print_log(log_message)
    logger.valid_logger(log_msg_dict, epoch)
    logger.img_logger(save_img_list, epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    # set random seed
    seed = random.randint(1, 10000) if opt['random_seed'] is None else opt['random_seed']
    setup_seed(seed)

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

    # set optimizer
    if opt['network']['type'] == "RRDB":
        model = SR_CyGen_RRDB(opt).to(device)
        resDecoder_params = list(map(id, model.resDecoder.parameters()))
        others_params = filter(lambda p: id(p) not in resDecoder_params, model.parameters())
        optimizer = optim.Adamax([
            {"params": model.resDecoder.parameters(), "lr": opt['train']['lr_resDecoder']},
            {"params": others_params, "lr": opt['train']['lr']}
        ], eps=1e-7, weight_decay=opt['train']['weight_decay'])
    else:
        model = SR_CyGen_DBPN(opt).to(device)
        optimizer = optim.Adamax(model.parameters(), lr=opt['train']['lr'], eps=1e-7,
                                 weight_decay=opt['train']['weight_decay'])
    print("Total number of paramerters is {}  ".format(sum(x.numel() for x in model.parameters())))
    lrsched = optim.lr_scheduler.StepLR(optimizer, step_size=opt['train']['scheduler']['step_size'],
                                        gamma=opt['train']['scheduler']['gamma'])

    # load checkpoint if exists
    start_epoch = 1
    if args.is_load_model and opt['load_path'] is not None:
        print("load Model：" + opt['load_path'])
        model_checkpoint = torch.load(opt['load_path'])
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['epoch'] + 1
        print("start epoch is:" + str(start_epoch))

    # set logger
    logger = MessageLogger(opt, args)
    logger.print_log("Log File successfully create in:" + args.save)
    logger.print_log(opt)
    logger.print_log(args)
    copyfile(args.opt, args.save + "/options.yml")

    # get trainset file path
    train_sets = [x for x in sorted(os.listdir(opt['datasets']['train']['dataroot'])) if is_hdf5_file(x)]
    valid_sets = [x for x in sorted(os.listdir(opt['datasets']['valid']['dataroot'])) if is_hdf5_file(x)]

    # init message recoder
    train_recoder = MessageRecoder(dict().fromkeys(('time', 'loss', 'loss_mse', 'loss_cm_1', 'loss_cm_2')))
    valid_recoder = MessageRecoder(dict().fromkeys(('time', 'psnr', 'ssim')))

    for epoch in range(start_epoch, opt['train']['total_epoch']):
        train_pipeline(epoch=epoch, train_sets=train_sets, model=model, train_recoder=train_recoder,
                       logger=logger, optimizer=optimizer, args=args, opt=opt)
        valid_pipeline(epoch=epoch, valid_sets=valid_sets, model=model, valid_recoder=valid_recoder,
                       logger=logger, args=args, opt=opt)
        logger.print_log("Epoch {:03d} is over!".format(epoch))
        logger.print_log("-------------------------------------")
        # save checkpoints
        model_path = join(args.save, "SR_CyGen_epoch_{:03d}.pkl".format(epoch))
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer,
                    'epoch': epoch}, model_path)
        lrsched.step()      # update lr


