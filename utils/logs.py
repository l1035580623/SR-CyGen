import logging
import os
from os.path import join
import wandb


def get_logger(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    info_file_handler = logging.FileHandler(logpath, mode="a")
    logger.addHandler(info_file_handler)
    screen_file_handler = logging.StreamHandler()
    logger.addHandler(screen_file_handler)

    return logger


def init_tb_logger(log_dir):
    if log_dir == None:
        return None
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


def init_wandb_logger(opt):
    if opt['logger']['wandb']['project'] is None:
        return None
    if opt['logger']['wandb']['resume_id'] is None:
        id = wandb.util.generate_id()
        opt['logger']['wandb']['resume_id'] = id
        resume = "never"
    else:
        id = opt['logger']['wandb']['resume_id']
        resume = "allow"
    wandb.init(project=opt['logger']['wandb']['project'], id=id, resume=resume, config=opt, name=opt['name'])
    return id


class MessageLogger(object):
    def __init__(self, opt, arg):
        self.logger = get_logger(join(arg.save, "logs"))
        if opt['logger']['use_tb_logger']:
            os.mkdir(join(arg.save, "tb_logs"))
            self.tb_logger = init_tb_logger(join(arg.save, "tb_logs"))
        else:
            self.tb_logger = None
        id = init_wandb_logger(opt)
        if id is not None:
            self.print_log(id)
        self.use_wandb = False if opt['logger']['wandb']['project'] is None else True
        self.interval = opt['logger']['print_freq']


    def print_log(self, msg):
        self.logger.info(msg)

    def train_logger(self, log_dict, epoch):
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('Train/Loss', log_dict['loss'], epoch)
            self.tb_logger.add_scalar('Train/Loss_MSE', log_dict['loss_mse'], epoch)
            self.tb_logger.add_scalar('Train/Loss_cm_1', log_dict['loss_cm_1'], epoch)
            self.tb_logger.add_scalar('Train/Loss_cm_2', log_dict['loss_cm_2'], epoch)
        if self.use_wandb:
            wandb.log(log_dict, step=epoch)

    def valid_logger(self, log_dict, epoch):
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('Valid/PSNR', log_dict['psnr'], epoch)
            self.tb_logger.add_scalar('Valid/SSIM', log_dict['ssim'], epoch)
        if self.use_wandb:
            wandb.log(log_dict, step=epoch)

    def img_logger(self, images, epoch):
        if self.use_wandb:
            wandb.log({'examples': [wandb.Image(img) for img in images]}, step=epoch)



