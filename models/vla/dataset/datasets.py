"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

# from prismatic.models.backbones.llm.prompting import PromptBuilder
# from prismatic.models.backbones.vision import ImageTransform
# from prismatic.util.data_utils import tree_map
# from prismatic.vla.action_tokenizer import ActionTokenizer, VQVAEActionTokenizer
from models.vla.dataset.rlds import make_interleaved_action_dataset, make_interleaved_dataset, make_single_dataset
from models.vla.dataset.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from models.vla.dataset.rlds.utils.data_utils import NormalizationType
from models.vla.action_tokenizer import ActionTokenizer, VQVAEActionTokenizer
from models.vla_vq.action_vqvae_wrapper import ActionVQVAELossWrapper
from utils.transforms import *
import torchvision.transforms as T
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

'''
@dataclass
class RLDSVQBatchTransform:
    action_tokenizer: VQVAEActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    image_window_size: int = 1

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # breakpoint()
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][self.image_window_size - 1 :]
        img = [Image.fromarray(rlds_batch["observation"]["image_primary"][t]) for t in range(self.image_window_size)]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # lang = "pick up the toy snake"
        robot_type, frequency = None, None  # OXE_ROBOT[rlds_batch["dataset_name"]]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(self.action_tokenizer.vqvae_model.token_num + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            robot_type=robot_type,
            control_frequency=frequency,
            actions=action,
        )


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        robot_type, frequency = None, None  # OXE_ROBOT[rlds_batch["dataset_name"]]
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            robot_type=robot_type,
            control_frequency=frequency,
            actions=action,
        )


@dataclass
class FastRLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        robot_type, frequency = None, None  # OXE_ROBOT[rlds_batch["dataset_name"]]
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        action_tokens, tokens_length = self.action_tokenizer(action)
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(tokens_length + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            robot_type=robot_type,
            control_frequency=frequency,
        )
'''
@dataclass
class RLDSBatchTransform:
    
    # action_tokenizer: VQVAEActionTokenizer
    vqvae_model: ActionVQVAELossWrapper
    image_window_size: int = 1
    image_transform: Any = None
    # image_transform: ImageTransform #这个要去大模型里面找

    def __post_init__(self):
        if self.image_transform is None:
            ops = []
            ops.append(RandomShiftsAug(pad=10))
            ops.append(ScaleImageTensor())
            ops.append(T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ))
            self.image_transform = T.Compose(ops)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        就做几个事
        1、把action在这里过vqvae,先取出5个,后面还要改成2*4的
        2、img_tranform,这个之后再说
        要注意此时的batch在GPU还是在CPU上
        """
        dataset_name = rlds_batch["dataset_name"]
        action = rlds_batch["action"][self.image_window_size - 1 :] #array(5, 7)
        # img = [rlds_batch["observation"]["image_primary"][t] for t in range(self.image_window_size)] #[array(224,224,3)]
        img = rlds_batch["observation"]["image_primary"][0]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = self.image_transform(img_tensor.unsqueeze(0))#注意这里是不是/255的

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        action1 = action[:5, :]
        action2 = action[5:, :]
        discretized_action1 = self.vqvae_model.get_code(action1)#(1, 4)
        discretized_action2 = self.vqvae_model.get_code(action2)#(1, 4)
        discretized_action = torch.cat([discretized_action1, discretized_action2], dim = 0)
        # discretized_action = self.vqvae_model.get_code(action) #(1,4)
        # discretized_action = discretized_action.to('cpu') #for pin_memory
        return dict(
            discretized_action = discretized_action, #(2, 4)
            dataset_name = dataset_name,
            action = action,
            lang = lang,
            img = img,
            img_tensor = img_tensor,
        )
    

@dataclass
class RLDSActionBatchTransform:
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        actions = rlds_batch["action"]
        actions_tensor = torch.tensor(actions.copy())
        return dict(actions=actions_tensor)


@dataclass
class RLDSNoiseActionBatchTransform:
    noise_level: float = 0.05
    dtype: torch.dtype = torch.bfloat16

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        actions = rlds_batch["action"]
        actions_copy = actions.copy()

        noise = np.random.normal(0, self.noise_level, actions_copy.shape)
        noise[..., -1] = 0

        actions_copy = actions_copy + noise
        actions_tensor = torch.tensor(actions_copy, dtype=self.dtype)
        return dict(actions=actions_tensor)


@dataclass
class RLDSLableBatchTransform:
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        actions = rlds_batch["action"]
        dataset_name = rlds_batch["dataset_name"]
        actions_tensor = torch.tensor(actions.copy())
        return dict(actions=actions_tensor, dataset_name=dataset_name)


class VqVAERLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSActionBatchTransform,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        window_size=10,
        image_window_size: int = 1,
        only_action: bool = False,
        sample_ratio=1.0,
    ) -> None:
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.sample_ratio = sample_ratio
        mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",), # TODO
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=image_window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=window_size-1,                        # For action chunking
                skip_unlabeled=False,                                # Skip trajectories without language labels
                goal_relabeling_strategy=None,                 # Goals are currently unused
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            only_action=only_action,
        )
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
        # if self.sample_ratio < 1.0:
        #     import tensorflow as tf
        #     def random_sample(x):
        #         return tf.random.uniform(()) < self.sample_ratio
        #     self.dataset = self.dataset.filter(random_sample)

    def make_dataset(self, rlds_config):
        return make_interleaved_action_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        rank = dist.get_rank()  # Get the current process rank
        world_size = dist.get_world_size()  # Get total number of processes
        for i, rlds_batch in enumerate(self.dataset.as_numpy_iterator()):
            if i % world_size == rank:
                yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int], 
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
        image_window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # # Configure RLDS Dataset(s)
        # if self.data_mix in OXE_NAMED_MIXTURES:
        #     mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        # else:
        #     # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
        mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        '''
        {
            'name': 'libero_10_no_noops',
            'data_dir': ...,
            'image_obs_keys':{'primary': 'image'}
            'absolute_action_mask': [False, False, False, False, False, False, True]
            'absolute_normalization_mask': [True, True, True, True, True, True, False]
            'action_proprio_normalization_type': '<NormalizationType.BOUNDS_Q99: 'bounds_q99'>
            'language_key': 'language_instruction'
            'standardize_fn': <function libero_dataset_transform at 0x735cc8189240> 
        }
        '''
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=image_window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=window_size-1,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )
        '''
        rlds_config:
        {
            'traj_transform_kwargs': 
                {
                    'window_size': 1, 
                    'future_action_window_size': 4, 
                    'skip_unlabeled': True, 
                    'goal_relabeling_strategy': 'uniform'
                }, 
            'frame_transform_kwargs': 
                {
                    'resize_size': (224, 224), 
                    'num_parallel_calls': 16 #数据处理线程
                    'image_augment_kwargs':
                    {
                        'random_resized_crop':
                        'random_brightness':
                        'random_contrast':
                        'random_saturation':
                        'random_hue':
                        'augment_order':['random_resized_crop', 'random_brightness', 'random_contrast', 'random_saturation', 'random_hue']
                    }
                }, 
            'dataset_kwargs_list': 
                [
                    {
                    'name': 'libero_10_no_noops', 
                    'data_dir': '/home/mike/InterMask/datasets/LIBERO_RLDS', 
                    'image_obs_keys': {'primary': 'image'}, 
                    'absolute_action_mask': [False, False, False, False, False, False, True], #只有最后一个动作使用有效的
                    'action_normalization_mask': [True, True, True, True, True, True, False], #前6个动作会进行标准化,最后一个动作不会
                    'action_proprio_normalization_type': <NormalizationType.BOUNDS_Q99: 'bounds_q99'>, 
                    'language_key': 'language_instruction', #加载语言指令的键
                    'standardize_fn': <function libero_dataset_transform at 0x707a3dd8d360>
                    }
                ] 
            'shuffle_buffer_size': 100000, 
            'sample_weights': [1.0], 
            'balance_weights': True, 
            'traj_transform_threads': 1, 
            'traj_read_threads': 1, 
            'train': True
        }
        '''
        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)
        # breakpoint()
        '''
        dataset_statistics:
            {
                'action':
                {
                    'mean': (7,)
                    'std': (7,)
                    'max': (7,)
                    'min': (7,)
                    'q01': (7,)
                    'q99': (7,)
                    'mask': (7,) 前6true,最后一个false
                }
                'proprio':
                {
                    'mean': (7,)
                    'std': (7,)
                    'max': (7,)
                    'min': (7,)
                    'q01': (7,)
                    'q99': (7,)  #全0
                }
                'num_tramsitions': array(101469)
                'num_trajectories': array(379)
            }
        dataset: element_spec:
            {
                'observation':
                {
                    'image_primary': TensorSpec(shape=(1, 224, 224, None), dtype=tf.uint8, name=None)
                    'timestep': TensorSpec(shape=(1,), dtype=tf.int32, name=None)
                    'pad_mask_dict':
                    'pad_mask':
                }
                'task':
                {
                    'language_instruction':TensorSpec(shape=(), dtype=tf.string, name=None)
                    'pad_mask_dict':
                    'image_primary': TensorSpec(shape=(224, 224, None), dtype=tf.uint8, name=None)
                    'timestep': TensorSpec(shape=(), dtype=tf.int32, name=None)
                }
                'action': TensorSpec(shape=(5, 7), dtype=tf.float32, name=None)
                'dataset_name': TensorSpec(shape=(), dtype=tf.string, name=None)
                'absolute_action_mask': TensorSpec(shape=(7,), dtype=tf.bool, name=None)
            }
        dataset_length: 
            dataset_statistics.num_tramsitions
        '''
    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        '''
        不过batch_transform的rlds_batch:
        {
            'observation':
            {
                'image_primary': array(1, 224, 224, 3)
                'timestep': array(1,)
                'pad_mask_dict':
                {
                    'image_primary': array([ True]),
                    'timestep': array([ True])
                }
                'pad_mask': array([ True])
            }
            'task':
            {
                'language_instruction': '...'
                'pad_mask_dict':
                {
                    language_instruction: True
                    image_primary: True
                    timestep: True
                }
                'image_primary': array(224, 224, 3)
                'timestep': int
            }
            'action':array(5, 7)
            'dataset_name': b'libero_10_no_noops'
            'absolute_action_mask': array([False, False, False, False, False, False,  True])
        }

        '''
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)
            # yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


# class DummyDataset(Dataset):
#     def __init__(
#         self,
#         action_tokenizer: ActionTokenizer,
#         base_tokenizer: PreTrainedTokenizerBase,
#         image_transform: ImageTransform,
#         prompt_builder_fn: Type[PromptBuilder],
#     ) -> None:
#         self.action_tokenizer = action_tokenizer
#         self.base_tokenizer = base_tokenizer
#         self.image_transform = image_transform
#         self.prompt_builder_fn = prompt_builder_fn

#         # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
#         # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
#         self.dataset_statistics = {
#             "dummy_dataset": {
#                 "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
#             }
#         }

#     def __len__(self):
#         # TODO =>> Replace with number of elements in your dataset!
#         return 10000

#     def __getitem__(self, idx):
#         # TODO =>> Load image, action and instruction from disk -- we use dummy values
#         image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
#         action = np.asarray(np.random.rand(7), dtype=np.float32)
#         instruction = "do something spectacular"

#         # Add instruction to VLA prompt
#         prompt_builder = self.prompt_builder_fn("openvla")
#         conversation = [
#             {"from": "human", "value": f"What action should the robot take to {instruction}?"},
#             {"from": "gpt", "value": self.action_tokenizer(action)},
#         ]
#         for turn in conversation:
#             prompt_builder.add_turn(turn["from"], turn["value"])

#         # Tokenize (w/ `base_tokenizer`)
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         labels = list(input_ids)

#         # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
#         #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
#         input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
#         pixel_values = self.image_transform(image)

#         # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
#         labels[: -(len(action) + 1)] = IGNORE_INDEX

#         return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
