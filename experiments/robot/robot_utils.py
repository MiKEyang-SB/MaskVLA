"""Utils for evaluating robot policies in various environments."""

import os
import random
import time

import numpy as np
import torch
from models.mask_transformer.transformer import Mask_VLA_Agent
# from experiments.robot.openvla_utils import (
#     get_vla,
#     get_vla_action,
#     get_vqvla,
# )

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# def get_model(cfg, wrap_diffusion_policy_for_droid=False):
#     """Load model for evaluation."""
#     if cfg.model_family == "openvla":
#         model = get_vla(cfg)
#     elif cfg.model_family == "vqvla":
#         model = get_vqvla(cfg)
#     else:
#         raise ValueError("Unexpected `model_family` found in config.")
#     print(f"Loaded model: {type(model)}")
#     return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla" or cfg.model_family == "vqvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, vqvae_model, vla_model, obs):
    """
    Queries the model to get an action.

    Args:
        cfg: Configuration object with model parameters
        vqvae_model: ActionVQVAELossWrapper model for decoding
        vla_model: Mask_VLA_Agent model for generating action tokens
        obs: Observation dictionary containing:
            - img_tensor: (1, 3, 224, 224) processed image tensor
            - lang: str, language instruction (lowercase)

    Returns:
        action: (window_size, 7) decoded action sequence
    """
    # Extract observation components
    img_tensor = obs['img_tensor']  # (1, 3, 224, 224)
    lang = [obs['lang']]  # Convert to list for batch processing

    # Generate action token IDs using MaskVLA model
    # ids shape: (batch_size, vq_action_dim * nbp) = (1, 4*2) = (1, 8)
    ids = vla_model.generate(
        img_tensor=img_tensor,
        lang=lang,
        timesteps=getattr(cfg, 'timesteps', 20),  # Number of denoising steps
        cond_scale=getattr(cfg, 'cond_scale', 3),  # Classifier-free guidance scale
        temperature=getattr(cfg, 'temperature', 1.0),
        topk_filter_thres=getattr(cfg, 'topk_filter_thres', 0.9),
        gsample=getattr(cfg, 'gsample', False),
        force_mask=False
    ) #(1,8)

    # Reshape ids for VQVAE decoding
    # ids: (1, 8) -> (1, 2, 4) where 2 = nbp, 4 = vq_action_dim
    batch_size = ids.shape[0]
    vq_action_dim = cfg.vq_action_dim  # 4
    nbp = cfg.window_size // 5  # 10 // 5 = 2
    ids = ids.view(batch_size, nbp, vq_action_dim)  # (1, 2, 4)

    # Get latent embeddings from VQ codebook
    # z_embed: (1, 2, 128) where 128 = n_latent_dims
    z_embed = vqvae_model.draw_code_forward(ids)
    n_latent_dims = z_embed.shape[-1]
    # # Flatten latent for decoding
    # # z_embed: (1, 2, 128) -> (1, 256)
    # z_embed_flat = z_embed.view(batch_size, -1) #(1,2,128)
    z_embed_flat = z_embed.reshape(-1, n_latent_dims) #(b*nbp, vq_action_dim)

    # Decode latent to actions
    # action: (1, window_size, 7) = (1, 10, 7)
    action = vqvae_model.get_action_from_latent(
        latent=z_embed_flat,
        robot_type=None,
        control_frequency=None
    )#(b*nbp,5,7)
    action = action.reshape(batch_size, -1, 7)

    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def get_maskvla(config):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
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
    )
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    #载入模型
    return vla_model