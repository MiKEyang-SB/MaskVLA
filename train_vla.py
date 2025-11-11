import gc
import os
# from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import wandb
import draccus
import torch
import torch.optim as optim
from tqdm import tqdm
# from accelerate import PartialState
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
# from transformers.modeling_outputs import CausalLMOutputWithPast
from models.vla_vq.action_vqvae_wrapper import ActionVQVAELossWrapper
# from models.vla.action_tokenizer import VQVAEActionTokenizer
from models.vla.dataset import RLDSDataset, RLDSBatchTransform
from models.mask_transformer.transformer import Mask_VLA_Agent
# from PIL import Image
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp
from utils.logger import LOGGER
from utils.scheduler import update_lr_warm_up
from utils.utils import setting, load_checkpoint
from datetime import datetime
from contextlib import nullcontext

@dataclass
class Config:
    # Model & Device Configuration
    image_sizes: tuple = (224, 224)

    # Directory Paths
    data_root_dir: str = "./datasets/LIBERO_RLDS"
    dataset_name: str = "libero_10_no_noops"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)

    # Model & Device Configuration
    vqvae_config_path: str = "models/vla_vq/action_vqvae_config" 
    device: str = "cuda"

    # VQVAE Parameters
    window_size: int = 10                                            # Action sequence window size for VQVAE
    image_window_size: int = 1                                      # Number of image frames to use as input
    vq_layer_group: int = 4                                         # Number of VQ layer groups in VQVAE
    n_latent_dims: int = 128                                        # Latent dimension size for VQVAE encoding
    checkpoint_path: str = "./checkpoints/action_tokenizer_weight/all_data_vq.pth"  
    use_action_type_pe: bool = True  
    use_time_pe: bool = True

    #traing Parameters
    SEED: int = 1
    wandb_enable: bool = True
    wandb_name: str = "MaskVLA"
    world_size: int = 0
    local_rank: int = -1
    cuda_device: int = 0

    output_dir = f"run/experiments/{datetime.now():%H-%M-%S}/"
    batch_size: int = 256
    max_epochs: int = 50
    shuffle_buffer_size: int = 100_000 
    image_aug: bool = True
    learning_rate: float = 2e-4
    gamma: float = 0.1
    gradient_accumulation_steps: int = 1
    grad_norm: int = 200
    bar_steps: int = 1
    log_steps: int = 10
    save_steps: int = 10000

    #checkpoint
    checkpoint: Optional[str] = None
    checkpoint_strict_load: bool = False
    resume_training: bool = False
    resume_encoder_only: bool = False

    #VLA Parameters
    code_dim: int = 512
    latent_dim: int = 512
    img_latent_dim: int = 512
    ff_size: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.2
    cond_drop_prob: float = 0.1
    # clip_version : str = 'ViT-L/14@336px'
    clip_version: str = 'ViT-B/32'
    num_tokens: int = 256 #logits的置信度的大小softmax,vq的codesize
    mask_type: str = '1D'
    step_unroll: int = 1


@draccus.wrap()
def train(config: Config) -> None:
    default_gpu, n_gpu, device = setting(config)
    # device = torch.device(config.device)
    vla_vqvae_model = ActionVQVAELossWrapper(
        config.vqvae_config_path,
        model_dtype="bf16",  # For training, we used mixed training
        interpolate=False,
        checkpoint_path=config.checkpoint_path,
        use_action_type_pe=config.use_action_type_pe,
        use_time_pe=config.use_time_pe,
        freeze=True,
        eval=True,
    ).to(device)

    vla_model = Mask_VLA_Agent(
        code_dim = config.code_dim,
        cond_mode='text',
        latent_dim = config.latent_dim,
        ff_size = config.ff_size,
        num_layers = config.num_layers,
        num_heads = config.num_heads,
        dropout = config.dropout,
        clip_dim = 512,
        cond_drop_prob = config.cond_drop_prob,
        lang_clip_version = config.clip_version,
        num_tokens = config.num_tokens,
        device = config.device,
        opt = config,
    ).to(device)

    if torch.distributed.is_initialized():
        vla_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vla_model)
    if torch.distributed.is_initialized():
        vla_model = DDP(
            vla_model,
            device_ids=[config.local_rank] if torch.cuda.is_available() else None,
            output_device=config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    
    torch.cuda.empty_cache()
    gc.collect() #clean rubbish
    # vla_vqvae_model.to(device)
    # vla_model.to(device)

    batch_transform = RLDSBatchTransform(
        vqvae_model = vla_vqvae_model
    )
    #code_idx1, _ = self.vq_model.encode(motion1) 其实这样子就行了
    vla_dataset = RLDSDataset(
        config.data_root_dir,
        config.dataset_name,
        batch_transform, #get_iter调用
        resize_resolution=config.image_sizes,
        shuffle_buffer_size=config.shuffle_buffer_size, #100_000
        image_aug=config.image_aug,
        window_size=config.window_size,
    )
    # collator = collate_fn()
    # print(vla_dataset.device)
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            vla_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle_flag = False
        pre_epoch = sampler.set_epoch

    else:
        sampler = None
        shuffle_flag = False  # RLDS/TFDS 自己有随机性时建议 False
        pre_epoch = lambda e: None

    train_dataloader = DataLoader(
        vla_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        # collate_fn=collate_fn(),
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        # pin_memory=True, 
    )
    len_train_dataloader = len(train_dataloader)
    total_iters = len_train_dataloader * config.max_epochs
    LOGGER.info('-----dataset lens-------:', len_train_dataloader)
    LOGGER.info("Model: nweights %d nparams %d" % (vla_model.num_parameters))#17,649,632个参数
    LOGGER.info("Model: trainable nweights %d nparams %d" % (vla_model.num_trainable_parameters))

    _model_for_ckpt = vla_model.module if isinstance(vla_model, DDP) else vla_model
    global_step, restart_epoch = load_checkpoint(config, len_train_dataloader, _model_for_ckpt)

    if torch.distributed.is_initialized():
        obj = [global_step, restart_epoch]
        torch.distributed.broadcast_object_list(obj, src=0)
        global_step, restart_epoch = obj
        torch.distributed.barrier()

    if default_gpu:
        save_training_meta(config)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        pbar = tqdm(initial=global_step, total=total_iters)
        # add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()
        pbar = NoOp()
    
    #------------------save----------------
        

    optimizer = optim.AdamW(vla_model.parameters(), 
                            betas=(0.9, 0.99), 
                            lr=config.learning_rate, 
                            weight_decay=1e-5)
    
    milestones = [int(total_iters * 0.5), int(total_iters * 0.7), int(total_iters * 0.85)]
    warm_up_iter = len_train_dataloader // 4
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=milestones,
                                                gamma=config.gamma)
    

    #分布式训练要看vqae的内容
    if config.wandb_enable and default_gpu:
        wandb_dict = {}

    scaler = None
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    vla_model.train()
    vla_vqvae_model.eval()

    optimizer.zero_grad()
    running_metrics = {}
    for epoch_id in range(restart_epoch, config.max_epochs):
        pre_epoch(epoch_id)
        for step, batch in enumerate(train_dataloader):
            need_sync = ((step + 1) % config.gradient_accumulation_steps == 0)
            ddp_ctx = (vla_model.no_sync() if isinstance(vla_model, DDP) and not need_sync else nullcontext())
            with ddp_ctx:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    losses, acc, _, _, _ = vla_model(batch)
                if config.gradient_accumulation_steps > 1:  # average loss
                    losses= losses / config.gradient_accumulation_steps
                losses.backward() #to wandb

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                # lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)#学习率衰减
                if step < warm_up_iter:
                    current_lr = update_lr_warm_up(step, 
                                                   optimizer, 
                                                   warm_up_iter, 
                                                   config.learning_rate)
                else:
                    current_lr = optimizer.param_groups[0]["lr"]
                # for kp, param_group in enumerate(optimizer.param_groups):
                #     param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-5)
                if config.wandb_enable and default_gpu:
                    wandb_dict.update({
                        'loss': losses.item(), 
                        'acc': acc.item(),
                        'lr': current_lr, 
                        'global_step': global_step})

                if config.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        vla_model.parameters(), config.grad_norm
                    )
                    if config.wandb_enable and default_gpu:
                        wandb_dict.update({'grad_norm': grad_norm})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if global_step % config.bar_steps == 0:
                pbar.update(config.bar_steps)#更新进度条

            if global_step % config.log_steps == 0 and config.wandb_enable and default_gpu:
                wandb.log(wandb_dict) 

            if global_step % config.save_steps == 0 and default_gpu:
                _to_save = vla_model.module if isinstance(vla_model, DDP) else vla_model
                model_saver.save(_to_save, global_step, optimizer=optimizer, rewrite_optimizer=True)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

    if global_step % config.save_steps != 0 and default_gpu:
        _to_save = vla_model.module if isinstance(vla_model, DDP) else vla_model
        model_saver.save(_to_save, global_step, optimizer=optimizer, rewrite_optimizer=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if config.wandb_enable and default_gpu:
        wandb.finish()


if __name__ == '__main__':
    train()
    # CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 train.py
