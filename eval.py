import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import torch
import torchvision.transforms as T
from LIBERO.libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf
import hydra
from PIL import Image

sys.path.append("../..")

import wandb
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    # get_model,
    get_maskvla,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from models.vla_vq.action_vqvae_wrapper import ActionVQVAELossWrapper
from models.mask_transformer.transformer import Mask_VLA_Agent
from utils.transforms import RandomShiftsAug, ScaleImageTensor
# @dataclass
# class GenerateConfig:
#     #################################################################################################################
#     # Model-specific parameters
#     #################################################################################################################



#     #################################################################################################################
#     # LIBERO environment-specific parameters
#     #################################################################################################################
#     task_suite_name: str = "libero_10_no_noops"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#     num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
#     num_trials_per_task: int = 50                    # Number of rollouts per task
    
#     #################################################################################################################
#     # Utils
#     #################################################################################################################
#     run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
#     local_log_dir: str = "./experiments/logs" 

#     wandb_enable: bool = False                          # Whether to also log results in Weights & Biases
#     wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
#     wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

#     seed: int = 7                                    # Random Seed (for reproducibility)

#     window_size: int = 4
#     vqvae_ckpt: str = ""

#     image_history_size: int = 1

#     image_sizes: int = 224


# @draccus.wrap()
@hydra.main(version_base=None, config_path=".", config_name="config")
def eval_libero(config: DictConfig) -> None:
    # Initialize LIBERO task suite
    set_seed_everywhere(config.seed)

    # config.unnorm_key = config.task_suite_name

    # Load device
    device = torch.device(config.device)

    # Load VQVAE model
    print("[*] Loading VQVAE model...")
    vla_vqvae_model = ActionVQVAELossWrapper(
        config.vqvae_config_path,
        model_dtype="bf16",
        interpolate=False,
        checkpoint_path=config.checkpoint_path,
        use_action_type_pe=config.use_action_type_pe,
        use_time_pe=config.use_time_pe,
        freeze=True,
        eval=True,
    ).to(device)
    vla_vqvae_model.eval()
    print("[*] VQVAE model loaded successfully")

    # Load MaskVLA model
    print("[*] Loading MaskVLA model...")
    vla_model = Mask_VLA_Agent(
        code_dim=config.code_dim,
        cond_mode='text',
        latent_dim=config.latent_dim,
        ff_size=config.ff_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        clip_dim=512,
        cond_drop_prob=config.cond_drop_prob,
        lang_clip_version=config.clip_version,
        num_tokens=config.num_tokens,
        device=config.device,
        opt=config,
        eval=True,
    ).to(device)

    vla_model.eval()
    print("[*] MaskVLA model ready for inference")

    # Initialize image transform (same as training)
    image_transform = T.Compose([
        RandomShiftsAug(pad=10),
        ScaleImageTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    run_id = f"EVAL-{config.task_suite_name}-{DATE_TIME}"
    if config.run_id_note is not None:
        run_id += f"--{config.run_id_note}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(config.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    log_file.write("Task suite:----------------------")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if config.wandb_enable:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_id,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {config.task_suite_name}")
    log_file.write(f"Task suite: {config.task_suite_name}\n")

    resize_size = config.image_sizes #224
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, vla_model, resolution=256)
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(config.num_trials_per_task)):#50
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            done = False
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []
            if config.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif config.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif config.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif config.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif config.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + config.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < config.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(config.model_family))
                    t += 1
                    continue
                img = get_libero_image(obs, 224)
                replay_images.append(img)
                img_history = replay_images[-config.image_history_size :]
                if len(img_history) < config.image_history_size:
                    img_history.extend([replay_images[-1]] * (config.image_history_size - len(img_history)))

                # Process image following training pipeline
                # Convert image to tensor and apply transforms
                img_np = img_history[0]  # Get the most recent image (224, 224, 3)
                img_tensor = torch.from_numpy(np.array(img_np, copy=True)).permute(2, 0, 1).float()  # (3, 224, 224)
                img_tensor = image_transform(img_tensor.unsqueeze(0))  # (1, 3, 224, 224)
                img_tensor = img_tensor.to(device)

                # Prepare observations dict (following training format)
                observation = {
                    "img_tensor": img_tensor,  # (1, 3, 224, 224)
                    "lang": task_description.lower(),  # Language instruction
                }

                # Debug: Save observation image and print language instruction
                # if t == config.num_steps_wait:  # Save first frame after wait period
                # debug_dir = os.path.join(config.local_log_dir, "debug_observations")
                # os.makedirs(debug_dir, exist_ok=True)

                # Save original image (before normalization)
                # img_pil = Image.fromarray(img_np.astype(np.uint8))
                # img_save_path = os.path.join(debug_dir, f"episode_{total_episodes+1}_step_{t}_task_{task_id}.png")
                # img_pil.save(img_save_path)

                # Print and log language instruction
                # print(f"[DEBUG] Saved observation image to: {img_save_path}")
                # print(f"[DEBUG] Language instruction: {observation['lang']}")
                # log_file.write(f"[DEBUG] Episode {total_episodes+1}, Step {t}\n")
                # log_file.write(f"[DEBUG] Image saved: {img_save_path}\n")
                # log_file.write(f"[DEBUG] Language: {observation['lang']}\n")
                # log_file.flush()

                #action
                # if t == 150:
                #     pass
                actions = get_action( #这一步是怎么回归的
                    config,
                    vla_vqvae_model,
                    vla_model,
                    observation,
                )#t到了150就报错

                actions = normalize_gripper_action(actions, binarize=True)  # 1,10,7
                actions = invert_gripper_action(actions)
                # breakpoint()
                # Execute action in environment
                for action in actions[0].tolist():
                    # breakpoint()
                    action = np.array(action)
                    obs, reward, done, info = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                if done:
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if config.wandb_enable:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if config.wandb_enable:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == '__main__':
    eval_libero()