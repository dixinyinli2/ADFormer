import os
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import random
from torch.optim.lr_scheduler import _LRScheduler

def get_log_dir(model_name, dataset, exp_id):
    log_dirs = './log'
    
    log_dir = os.path.join(log_dirs, model_name, dataset, exp_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_logger(log_dir, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    log_file = os.path.join(log_dir, 'run.log')
    print('Creat Log File in: ', log_file)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        t_data =  (data - self.mean) / self.std
        return t_data

    def inverse_transform(self, data):
        it_data =  (data * self.std) + self.mean
        return it_data
    

def create_dataloader(x_data, y_data, batch_size=32, shuffle=True):
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_t=0, warmup_lr_init=1e-5, eta_min=0, last_epoch=-1):
        """
        Warmup + Cosine Annealing learning rate scheduler.

        :param optimizer: The optimizer for which the learning rate is adjusted
        :param T_max: The number of epochs for one cycle of cosine annealing
        :param warmup_t: The number of epochs for the warmup phase
        :param warmup_lr_init: The initial learning rate during warmup
        :param eta_min: The minimum learning rate after cosine annealing
        :param last_epoch: The index of the last epoch (default: -1)
        """
        self.T_max = T_max
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_t:
            # Warmup phase
            warmup_lr = self.warmup_lr_init + (self.base_lrs[0] - self.warmup_lr_init) * self.last_epoch / self.warmup_t
            return [warmup_lr for _ in self.base_lrs]
        else:
            # Cosine annealing phase
            epoch_in_cosine_phase = self.last_epoch - self.warmup_t
            if epoch_in_cosine_phase >= self.T_max:
                # After T_max, keep the learning rate at eta_min
                return [self.eta_min for _ in self.base_lrs]
            else:
                cosine_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * epoch_in_cosine_phase / self.T_max)) / 2
                return [cosine_lr for _ in self.base_lrs]
    

        


def mae(y_true, y_pred, mask=0.5):
    if mask != -1:
        mask = y_true > mask
    else:
        mask = np.ones_like(y_true, dtype=bool)
    loss = np.abs(y_true[mask] - y_pred[mask])
    return np.mean(loss)

def rmse(y_true, y_pred, mask=0.5):
    if mask != -1:
        mask = y_true > mask
    else:
        mask = np.ones_like(y_true, dtype=bool)
    loss = (y_true[mask] - y_pred[mask]) ** 2 
    return np.sqrt(np.mean(loss))

def mape(y_true, y_pred, mask=5.0):
    mask = y_true > mask
    loss = np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask]))
    return np.mean(loss) * 100 



def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)