import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.transformer_modules import *
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from PIL import Image
from torchvision.transforms import ToPILImage
from models.mask_transformer.networks.clip_lang_encoder import LangClip
from models.mask_transformer.networks.resnets import BesoResNetEncoder
# import hydra
from utils.logger import LOGGER
import einops 
from timm.models.layers import trunc_normal_
class BaseModel(nn.Module):
    @property
    def num_parameters(self):
        nweights = sum(p.numel() for p in self.parameters())
        nparams  = sum(1 for _ in self.parameters())
        return nweights, nparams


    @property
    def num_trainable_parameters(self):
        nweights = 0
        nparams = 0
        for p in self.parameters():          # 不用名字就不必 named_parameters
            if p.requires_grad:
                nweights += p.numel()        # 等价于 np.prod(p.size())
                nparams  += 1
        return int(nweights), int(nparams)


    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def _init_weights(self, m):#初始化权重参数
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class SpaTempPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SpaTempPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = PositionalEncoding2D(d_model)
    
    def forward(self, x, window_size=10):
        seqlen, bs, input_feats = x.shape
        t = window_size//5
        # x1, sep, x2  = x.split([seqlen//2, 1, seqlen//2])
        #这里的时空注意力出问题了，到底该怎么办啊
        def add_positional_encoding(x):
            x = x.permute(1,0,2) # [seqen, bs, input_feats] -> [bs, seqen, input_feats]
            x = x.reshape(x.shape[0], t, x.shape[1]//t, x.shape[2])

            x = x + self.positional_encoding(x)

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            x = x.permute(1,0,2)

            return x

        x = add_positional_encoding(x)
        # x2 = add_positional_encoding(x2)
        
        # x = torch.cat([x1, sep, x2], dim=0)
        return self.dropout(x)

class OutputProcess_adaLN(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        
        self.LayerNorm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = AdaLNModulation(latent_dim, nchunks=2)
        
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        
        shift, scale = self.adaLN_mod(cond)
        hidden_states = modulate(self.LayerNorm(hidden_states), shift, scale)

        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, 
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.nbp = 5
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = SpaTempPositionalEncoding(self.latent_dim, self.dropout)

        self.Transformer = InterMTransformer(d_model=self.latent_dim,
                                        nhead=num_heads,
                                        dim_feedforward=ff_size,
                                        dropout=dropout,
                                        num_layers=num_layers,
                                        nbp=self.nbp)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 3  # three dummy tokens, one for masking, one for padding, one for separating
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.sep_id = opt.num_tokens + 2

        self.output_process = OutputProcess_adaLN(out_feats=opt.num_tokens, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.initialize_weights()
        
        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule
        self.noise_schedule_backward = cosine_schedule_backward

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def initialize_weights(self):
        def __init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)
        
        self.apply(__init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.Transformer.blocks:
            nn.init.constant_(block.adaLN_mod_combined.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_combined.model[-1].bias, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].bias, 0)
            
        
        # nn.init.normal_(self.position_enc.pe, mean=0.0, std=0.02)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].weight, 0)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].bias, 0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_')]

    def load_and_freeze_clip(self, clip_version): #ViT-B/32

        ##From InterGen
        clip_model, preprocess = clip.load(clip_version, device="cpu", jit=False)

        self.clip_token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.clip_positional_embedding = clip_model.positional_embedding
        self.clip_ln_final = clip_model.ln_final
        self.clip_dtype = clip_model.dtype
        self.preprocess = preprocess

        for p in self.clip_transformer.parameters():
            p.requires_grad = False
        for p in self.clip_token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip_ln_final.parameters():
            p.requires_grad = False
        
        clipTransLayer = nn.TransformerEncoderLayer(d_model=768,
                                                    nhead=8,
                                                    dim_feedforward=2048,
                                                    dropout=0.1,
                                                    activation="gelu",
                                                    batch_first=True)
        self.clipTrans = nn.TransformerEncoder(clipTransLayer, num_layers=2)
        self.clipln = nn.LayerNorm(768)

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
            
        # From InterGen
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.clip_token_embedding(text).type(self.clip_dtype)
            pe_tokens = x + self.clip_positional_embedding.type(self.clip_dtype)
            x = pe_tokens.permute(1,0,2)
            x = self.clip_transformer(x)
            x = x.permute(1,0,2)
            clip_out = self.clip_ln_final(x).type(self.clip_dtype)
        
        out = self.clipTrans(clip_out)
        out = self.clipln(out)
        feat_clip_text = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE 哪些地方是有效的
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        bs, n_tokens = motion_ids.shape
        
        cond = self.mask_cond(cond, force_mask=force_mask)
        
        motion_ids = torch.cat((motion_ids[:, :n_tokens//2], 
                                self.sep_id * torch.ones((bs, 1), device=motion_ids.device, dtype=torch.long),
                                motion_ids[:, n_tokens//2:]), dim=-1)
        #中间加一个分隔符(8, 751)
        x = self.token_emb(motion_ids) #(b, 751, 512)
        
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)
        cond = self.cond_emb(cond) #(1, b, latent_dim)

        x = self.position_enc(x) #(751, b, latent_dim)
        
        padding_mask = torch.cat([padding_mask[:, :n_tokens//2],
                                torch.zeros_like(padding_mask[:, 0:1]), 
                                padding_mask[:, n_tokens//2:]], dim=1) #(b, seqlen+1)

        output = self.Transformer(x, cond, src_key_padding_mask=padding_mask) #(seqlen, b, e)
        logits = self.output_process(output, cond) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens):#motion, cond, m_lens
        '''
        :param ids: (b, n)  #(8, 750)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action 
        :m_lens: (b,)  #(8,)
        :return:
        '''

        bs, ntokens = ids.shape
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        max_len = ntokens/2  #375
        max_len = max_len/self.nbp #这是一个人的 75

        non_pad_mask = lengths_to_mask(m_lens, max_len) #(b, max_len * 2)
        non_pad_mask = non_pad_mask.repeat(1, self.nbp) #(b, max_len * 2 * nbp)
        # print(f">>> Pad mask: {non_pad_mask.reshape(2,2,-1).reshape(2,2,5,-1)}")

        ids = torch.where(non_pad_mask, ids, self.pad_id) #(8, 750) 将后面的填充为pad_id
        # print(f">>> Padded ids: {ids.reshape(2,2,-1).reshape(2,2,5,-1)}")

        force_mask = False
        if self.cond_mode == 'text':
            cond_vector = self.encode_text(y)  #(8, 768)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        interaction_mask = torch.bernoulli(torch.tensor(self.opt.interaction_mask_prob)).bool().item() #指定interaction_mask_prob概率生成一个bool值
        # print(f"Interaction: {interaction_mask}")
        if interaction_mask: #lianggeren 
            mask_person_one = torch.bernoulli(torch.tensor(0.5)).bool().item()
            # print(f"zero first: {zeros_first}")
            if mask_person_one: #是否将第一个人放在half_batch_mask的前半部分
                half_batch_mask = torch.cat((torch.zeros((bs//2,ntokens//2)), torch.ones((bs//2,ntokens//2))), dim=-1).bool()
            else:
                half_batch_mask = torch.cat((torch.ones((bs//2,ntokens//2)), torch.zeros((bs//2,ntokens//2))), dim=-1).bool()
            #要么前半部分是0,后半部分是1,要么反过来
            mask = torch.cat((half_batch_mask, ~half_batch_mask), dim=0).to(device) #(8, 750)
            rand_mask_probs = torch.ones((bs,), device=device) * 0.5
            rand_time = self.noise_schedule_backward(rand_mask_probs)

        else:#随机掩码序列的一部分
            rand_time = uniform((bs,), device=device)#生成一个均匀分布的随机值
            rand_mask_probs = self.noise_schedule(rand_time)#计算每个位置的掩码概率
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)#序列掩码数量

            batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1) #位置的排序索引，排序目的是为了掩码选择哪些位置
            # Positions to be MASKED are ALL TRUE
            mask = batch_randperm < num_token_masked.unsqueeze(-1)#随机掩码一部分

        # Positions to be MASKED must also be NON-PADDED
        #mask:(8, 750)  non_pad_mask:(8, 750)
        mask &= non_pad_mask  #哪些掩码 + 哪些是有效的

        # Note this is our training target, not input
        labels = torch.where(mask, ids, self.mask_id) #mask的值变成为mask_id  #(8, 750)
        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)#百分之10替换为不正确的token
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)#
        x_ids = torch.where(mask_rid, rand_id, x_ids)#百分之10的概率替换为随机值
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token

        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)#被选中mask的部分有百分之88被替换为mask

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)#(8, 1024, 750)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        logits = logits.permute(0,2,1) # B,ntokens,T -> B,T,ntokens

        if self.opt.step_unroll:
            su_ce_loss, su_pred_id, su_acc = self.step_unroll_forward(x_ids, mask_mid, labels, logits,
                                                                cond_vector, non_pad_mask, force_mask)
        
            return ce_loss + (self.opt.step_unroll * su_ce_loss), (acc + self.opt.step_unroll*su_acc)/2, pred_id, su_pred_id, logits
        else:
            return ce_loss, acc, pred_id, None, logits

    def step_unroll_forward(self, prev_masked_ids, prev_mask, prev_labels, logits, cond_vector, non_pad_mask, force_mask):
        # print(f">>>>>>>>>>>> Step unroll >>>>>>>>>>>>>>>")
        total_timesteps = 20
        prev_rand_mask_probs = prev_mask.count_nonzero(dim = -1).float() / prev_mask.shape[-1] #计算上一个时间步中被掩码的比例
        prev_rand_time = self.noise_schedule_backward(prev_rand_mask_probs) #计算反向噪声调度的时间 #(b,)

        rand_time = (prev_rand_time + (1/(total_timesteps+1))).clamp(max=1) #(b,)
        rand_mask_probs = self.noise_schedule(rand_time) #计算当前时间步的随机掩码概率 #(b,)

        probs = logits.softmax(dim=-1) #(b, 750, 1024)
        scores, pred_ids = probs.max(dim=-1)
        scores = scores.masked_fill(~prev_mask, 1e5)#被掩码的地方设置一个很大的数

        sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        num_token_masked = torch.round(rand_mask_probs * (scores.shape[-1])).clamp(min=1)#需要掩码的token数量

        mask = (ranks < num_token_masked.unsqueeze(-1))#哪些位置需要掩码

        retained_preds = torch.logical_and(prev_mask == True,  mask == False)#哪些位置在prev_mask是有效的，被保留的
        labels = torch.where(retained_preds, self.mask_id, prev_labels)#保留的位置替换为mask_id

        x_ids = torch.where(retained_preds, pred_ids, prev_masked_ids)#根据保留的位置，更新当前时间步的输入

        step_unroll_logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)
        return cal_performance(step_unroll_logits, labels, ignore_index=self.mask_id)



    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale  #Classifier-free
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        token_lengths = m_lens*2

        if self.cond_mode == 'text':
            cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        padding_mask = padding_mask.repeat(1, self.nbp)
        token_lengths = token_lengths*self.nbp

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            # temperature is annealed, gradually reducing temperature hence randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def generate_reaction(self,
                            conds,
                            motion1_ids,
                            m_lens,
                            timesteps: int,
                            cond_scale: int,
                            temperature=1,
                            topk_filter_thres=0.9,
                            gsample=False,
                            force_mask=False
                            ):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        token_lengths = m_lens

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")
        
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        padding_mask = padding_mask.repeat(1, self.nbp)
        token_lengths = token_lengths*self.nbp

        # Start from all tokens being masked
        ids1 = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, motion1_ids)
        ids = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, self.mask_id)
        scores = torch.where(ids == self.mask_id, 0., 1e5)
        
        starting_temperature = temperature
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps//2, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(torch.cat((ids1,ids), dim=-1), cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)
            logits = logits[:, :, padding_mask.shape[1]//2:]
            
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.cat((ids1, ids), dim=-1)
        ids = torch.where(padding_mask, -1, ids)

        return ids

class Mask_VLA_Agent(BaseModel):
    def __init__(self, 
                 code_dim, #512
                 cond_mode, 
                 latent_dim=256, 
                 img_latent_dim=512,
                 ff_size=1024, 
                 num_encoder_layers=2,
                 num_decoder_layers=8,
                 num_heads=4, 
                 dropout=0.1, 
                 clip_dim=512, 
                 cond_drop_prob=0.1, 
                 lang_clip_version=None, 
                 num_tokens = 1024, #vq里面的值的大小 256
                 device = 'cuda',
                 opt=None, 
                #  mask_type='1D',
                #  window_size=5,
                 **kargs):
        super().__init__()
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.img_latent_dim = img_latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.mask_type = opt.mask_type
        self.window_size = opt.window_size
        self.nbp = self.window_size//5
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.device = device

        self.language_clip = LangClip(model_name=lang_clip_version)

        self.img_resnet = BesoResNetEncoder(self.img_latent_dim)

        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = SpaTempPositionalEncoding(self.latent_dim, self.dropout)

        self.Transformer = VLATransformer(d_model=self.latent_dim,
                                        nhead=num_heads,
                                        dim_feedforward=ff_size,
                                        dropout=dropout,
                                        num_layers=num_decoder_layers,
                                        nbp=self.nbp)
        
        self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)

        cond_TransLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                    nhead=8,
                                                    dim_feedforward=4*self.latent_dim,
                                                    dropout=0.1,
                                                    activation="gelu",
                                                    batch_first=True)
        self.cond_Trans = nn.TransformerEncoder(cond_TransLayer, num_layers=num_encoder_layers)

        self.cond_ln = nn.LayerNorm(self.latent_dim, bias = True)
        _num_tokens = num_tokens + 1
        self.mask_id = num_tokens
        # self.pad_id = num_tokens + 1
        # self.sep_id = num_tokens + 2

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)#256+1,512
        #看一下vqvae输出的这个是不是1024啊
        self.output_process = OutputProcess_adaLN(out_feats=_num_tokens, latent_dim=latent_dim)

        # self.clip_version = lang_clip_version
        # self.load_and_freeze_clip(lang_clip_version)
        
        self.noise_schedule = cosine_schedule
        self.noise_schedule_backward = cosine_schedule_backward

    # def load_and_freeze_clip(self, clip_version): #ViT-B/32

    #     ##From InterGen
    #     clip_model, preprocess = clip.load(clip_version, device="cpu", jit=False)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     clip_model = clip_model.to(device) 
    #     clip_model.eval() 
    #     self.clip_model = clip_model
    #     self.clip_encode_image = clip_model.encode_image
    #     self.clip_token_embedding = clip_model.token_embedding
    #     self.clip_transformer = clip_model.transformer
    #     self.clip_positional_embedding = clip_model.positional_embedding
    #     self.clip_ln_final = clip_model.ln_final
    #     self.clip_dtype = clip_model.dtype
    #     self.preprocess = preprocess

    #     for p in self.clip_transformer.parameters():
    #         p.requires_grad = False
    #     for p in self.clip_token_embedding.parameters():
    #         p.requires_grad = False
    #     for p in self.clip_ln_final.parameters():
    #         p.requires_grad = False
        
    #     clipTransLayer = nn.TransformerEncoderLayer(d_model=768,
    #                                                 nhead=8,
    #                                                 dim_feedforward=2048,
    #                                                 dropout=0.1,
    #                                                 activation="gelu",
    #                                                 batch_first=True)
    #     self.clipTrans = nn.TransformerEncoder(clipTransLayer, num_layers=2)
    #     self.clipln = nn.LayerNorm(768)

    def encode_text_origin(self, raw_text):
        device = next(self.parameters()).device
        # for raw_text in raw_text_list:
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)#(B, 77)
            x = self.clip_token_embedding(text).type(self.clip_dtype)#(B, 77, 768)
            pe_tokens = x + self.clip_positional_embedding.type(self.clip_dtype)#(B, 77, 768)
            x = pe_tokens.permute(1,0,2) #torch.Size([77, B, 768])
            x = self.clip_transformer(x)
            x = x.permute(1,0,2)
            clip_out = self.clip_ln_final(x).type(self.clip_dtype)#torch.Size([B, 77, 768])
        out = self.clipTrans(clip_out)
        out = self.clipln(out)#torch.Size([B, 77, 768])
        feat_clip_text = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]#(B, 768)
        return feat_clip_text
    
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        language_clip_model = self.language_clip
        text_vector = language_clip_model(raw_text).to(device)
        return text_vector#(B, 1, 512)
    
    def encode_img_origin(self, pil_image_list):
        device = next(self.parameters()).device
        image_input = torch.stack([self.preprocess(img) for img in pil_image_list]).to(device)
        #torch.Size([8, 3, 336, 336])
        #这里报错，提示模型在cpu上，但是权重在GPU上，应该保持一致看一下，intermask是怎么做的
        with torch.no_grad():
            img_out = self.clip_encode_image(image_input).type(self.clip_dtype)
        return img_out
    
    def encode_img(self, img): #img:(B, 1, 3, 224, 224)
        #先做转换
        img = img.to(self.device)
        if img.ndim == 5:
            img = einops.rearrange(img, 'b t c h w -> (b t) c h w')
        img_tokens = self.img_resnet(img).unsqueeze(1)
        return img_tokens
    
    def mask_cond(self, cond, force_mask=False):#条件掩码
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:#1
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
        
    def trans_forward(self, discretized_action_ids, cond, force_mask=False):
        # # bs, n_tokens = discretized_action_ids.shape
        # cond = self.mask_cond(cond, force_mask=force_mask)
        # x = self.token_emb(cond) #(b,len,dim)
        # x = self.input_process(x)
        # cond = self.cond_emb(cond) #(b, latent_dim)
        # x = self.position_enc(x)
        # output = self.Transformer(x, cond, src_key_padding_mask=padding_mask)
        # logits = self.output_process(output, cond) 
        # return logits
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(discretized_action_ids) #(b,ntokens) -> (b,ntokens,dim)
        x = self.input_process(x) #(ntokens,b,dim)
        x = self.position_enc(x, window_size = self.window_size)
        output = self.Transformer(x, cond, src_key_padding_mask=None)#这里的src_key_padding_mask要注意是否要给出定义
        logits = self.output_process(output, cond) #(B, _num_tokens, x*y)
        return logits

    def cond_encoder(self, cond, custom_attn_mask=None):#(B, 2, 512)
        #deal condition shapes

        x = self.cond_Trans(cond, src_key_padding_mask=custom_attn_mask)
        x = self.cond_ln(x)
        return x

    def forward(self, batch):
        '''
        :param ids: (b, n)  #(b, vqvae_output_dimension)
        :batch:
        {
            batch['discretized_action']: (B, w/5, 4),
            batch['dataset_name']:[[b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops', b'libero_10_no_noops']]
            batch['action']:(B, 5, 7),
            batch['lang']: [txt1,txt2,..]
            batch['img']:torch.Size([B, 224, 224, 3])
            batch['img_tensor']:torch.Size([8, 1, 3, 224, 224])
        }
        '''
        batch = self.prepare_batch(batch) #to GPU
        discretized_action = batch['discretized_action']
        lang = batch['lang']
        img = batch['img']
        img_tensor = batch['img_tensor']
        # pil_image_list = []
        # for i in range(bs):
        #     single_image_array = image[i] #(224,224,3)
        #     single_image_array = single_image_array.permute(2, 0, 1)
        #     to_pil = ToPILImage()
        #     pil_image = to_pil(single_image_array)
        #     pil_image_list.append(pil_image)
        #研究img和text的数量大小，和在哪个设备上过模型，有没有需要加速的地方
        force_mask = False
        img_vector = self.encode_img(img_tensor)#(B, 1, 512)
        text_vector = self.encode_text(lang) #(B, 1, 512)
        cond = torch.cat([img_vector, text_vector], dim = 1)#(B, 2, 512)
        #这里cond要过几层encoder
        cond = self.cond_emb(cond)
        cond = self.cond_encoder(cond)#(B, 2, 512)
        cond = cond.mean(dim = 1)#(B, 512)
        #(clip_dim→latent_dim)
        #这里要提防绝对值大小不一样的问题，看一下MDT怎么解决的，先放着
        #这里的融合有讲究，要么cross-attention,要么直接在第二个维度拼接,这个后面还可以改
        
        discretized_action_ids = discretized_action.view(discretized_action.shape[0], -1) #(B, w/5 * 4)
        #暂时B, 4，之后B, 2, 4
        bs, xtokens, ytokens = discretized_action.shape
        ntokens = xtokens * ytokens
        #1D掩码
        if self.mask_type == '1D':
            rand_time = uniform((bs,), device=self.device)#(bs,)
            rand_mask_probs = self.noise_schedule(rand_time)#每个样本的掩码比例
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)#每个样本的掩码数量
            batch_randperm = torch.rand((bs, ntokens), device=self.device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)#(bs, ntokens),遮蔽的位置
            labels = torch.where(mask, discretized_action_ids, self.mask_id)#(bs, ntokens)
            #mask or keep origin
            x_ids = discretized_action_ids.clone()
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid, rand_id, x_ids)#10%替换为随机的值
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)#(b, x*y)

        elif self.mask_type == '2D':
            rand_time = uniform((bs,), device=self.device)
            rand_mask_probs = self.noise_schedule(rand_time)

            # ========== temporal mask
            #因为太少了，min就设置为0吧
            num_token_masked = (xtokens * rand_mask_probs).round().clamp(min=0)
            batch_randperm = torch.rand((bs, xtokens), device=self.device).argsort(dim=-1)
            # Positions to be MASKED are ALL TRUE
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            # Positions to be MASKED must also be NON-PADDED
            # mask = mask & non_pad_mask[..., 0]
            # Note this is our training target, not input
            labels = torch.where(mask[..., None].repeat(1, 1, ytokens), discretized_action, self.mask_id)
            x_ids = discretized_action.clone()
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid[..., None].repeat(1, 1, ytokens), rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid[..., None].repeat(1, 1, ytokens), self.mask_id, x_ids)
            mask_time = mask
            mask_time = mask_time[..., None].repeat(1, 1, ytokens)       # keep temperal mask still masked
            # print((x_ids==512).sum(), mask_time.sum(), mask.sum(), (labels!=512).sum(), mask_rid.sum())

            # ========== spatial mask
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=0)
            batch_randperm = torch.rand((bs, ntokens), device=self.device).argsort(dim=-1)
            # Positions to be MASKED are ALL TRUE
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            # Positions to be MASKED must also be NON-PADDED
            # mask = mask & non_pad_mask.reshape(bs, -1)
            mask = mask & ~mask_time.reshape(bs, -1)
            # Note this is our training target, not input
            labels = torch.where(mask, x_ids.reshape(bs, -1), labels.reshape(bs, -1))
            x_ids = x_ids.reshape(bs, -1)
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask, 0.1)

            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            # mask_mid = mask
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)#(b, x*y)
        
        #trans_forward
        logits = self.trans_forward(x_ids, cond, force_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
        logits = logits.permute(0,2,1)

        if self.opt.step_unroll:
            su_ce_loss, su_pred_id, su_acc = self.step_unroll_forward(x_ids, mask_mid, labels, logits, cond, force_mask)
            loss = ce_loss + (self.opt.step_unroll * su_ce_loss)
            acc = (acc + self.opt.step_unroll*su_acc)/2,
            return loss, acc, pred_id, su_pred_id, logits
        else:
            return ce_loss, acc, pred_id, None, logits
    
    def step_unroll_forward(self, prev_masked_ids, prev_mask, prev_labels, logits, cond_vector, force_mask):
        # print(f">>>>>>>>>>>> Step unroll >>>>>>>>>>>>>>>")
        #纠错式再预测
        total_timesteps = 20
        prev_rand_mask_probs = prev_mask.count_nonzero(dim = -1).float() / prev_mask.shape[-1]
        prev_rand_time = self.noise_schedule_backward(prev_rand_mask_probs)
        #反推上一步的时间刻度
        rand_time = (prev_rand_time + (1/(total_timesteps+1))).clamp(max=1)
        rand_mask_probs = self.noise_schedule(rand_time)

        probs = logits.softmax(dim=-1) #用上一步的logits
        scores, pred_ids = probs.max(dim=-1) 
        scores = scores.masked_fill(~prev_mask, 1e5)#上一轮对不该动的位置设置为1e5,避免再次挑中

        sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        num_token_masked = torch.round(rand_mask_probs * (scores.shape[-1])).clamp(min=1)

        mask = (ranks < num_token_masked.unsqueeze(-1))#挑出前num_token_masked个位置mask重新预测

        retained_preds = torch.logical_and(prev_mask == True,  mask == False)
        labels = torch.where(retained_preds, self.mask_id, prev_labels)

        x_ids = torch.where(retained_preds, pred_ids, prev_masked_ids)

        step_unroll_logits = self.trans_forward(x_ids, cond_vector, force_mask)
        return cal_performance(step_unroll_logits, labels, ignore_index=self.mask_id)

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits
    
    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 img_tensor,
                 lang,
                #  m_lens,
                 timesteps: int,
                 cond_scale: int, #classifire-free
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):

        # device = next(self.parameters()).device
        # seq_len = max(m_lens)
        # batch_size = len(m_lens)
        # token_lengths = m_lens*2

        # if self.cond_mode == 'text':
        #     cond_vector = self.encode_text(conds)
        # elif self.cond_mode == 'action':
        #     cond_vector = self.enc_action(conds).to(device)
        # elif self.cond_mode == 'uncond':
        #     cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        # else:
        #     raise NotImplementedError("Unsupported condition mode!!!")
        img_vector = self.encode_img(img_tensor)#(B, 1, 512)
        text_vector = self.encode_text(lang) #(B, 1, 512)
        cond = torch.cat([img_vector, text_vector], dim = 1)#(B, 2, 512)
        #这里cond要过几层encoder
        cond = self.cond_emb(cond)
        cond = self.cond_encoder(cond)#(B, 2, 512)
        cond_vector = cond.mean(dim = 1)#(B, 512)

        # padding_mask = ~lengths_to_mask(m_lens, seq_len) #这两个直接设
        # padding_mask = padding_mask.repeat(1, self.nbp)
        token_lengths = token_lengths*self.nbp

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)#全部mask的token
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            #1D
            num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k)每个位置的排名
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids) #不确定的位置进行重采样

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                #   padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)    
            #  logits = logits_cond + s*(logits_cond - logits_uncond)
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            # temperature is annealed, gradually reducing temperature hence randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
                #近似argmax采样
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)
                #多项式采样
            ids = torch.where(is_mask, pred_ids, ids)#刚才标记的mask位置
            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            #取概率作为分数
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids