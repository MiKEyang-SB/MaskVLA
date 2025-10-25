import math
import time
import torch
import random
import numpy as np
from utils.misc import NoOp, set_random_seed
import wandb
from utils.logger import LOGGER
from utils.distributed import set_cuda
import os
import uuid

def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)
    # now = time.time()
    # message += '%s'%(as_minutes(now - start_time))


    for k, v in losses.items():
        message += ' %s: %.5f ' % (k, v)
    # message += ' sl_length:%2d tf_ratio:%.2f'%(sl_steps, tf_ratio)
    print(message)

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setting(config):
    if config.wandb_enable:
        time_id = f"{time.strftime('%m%d-%H')}"
        wandb.init(project='mini-diff', name=config.wandb_name + f"{time_id}_{str(uuid.uuid4())[:8]}", config = config)
    default_gpu, n_gpu, device = set_cuda(config)

    if default_gpu:
        LOGGER.info(
            'device: {}, n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )
    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)
    return default_gpu, n_gpu, device

def load_checkpoint(config, len_train_dataloader, model):
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.resume_training: #检查是否恢复训练
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']#设置step
        restart_epoch = global_step // len_train_dataloader#设置epoch
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len_train_dataloader #训练重新开始
    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                # TODO: mae_encoder.encoder.first_conv.0.weight
                if k == 'mae_encoder.encoder.first_conv.0.weight':
                    if v.size(1) != state_dict[k].size(1):
                        new_checkpoint[k] = torch.zeros_like(state_dict[k])
                        min_v_size = min(v.size(1), state_dict[k].size(1))
                        new_checkpoint[k][:, :min_v_size] = v[:, :min_v_size] #有通道不匹配就用0填充，复制可用的部分
                if v.size() == state_dict[k].size():
                    if config.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                        continue
                    new_checkpoint[k] = v # 正常匹配就加载
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)
    return global_step, restart_epoch