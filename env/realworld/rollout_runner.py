# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
 
import fnmatch
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

RESOLUTION = (480, 480)

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    dataset_dir = os.path.expanduser(dataset_dir)
    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} does not exist.")
        return []

    hdf5_files = []
    for root, _, files in os.walk(dataset_dir, followlinks=True):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def convert_dataset_image(dataset_dir, task_name='toy_collection', action_name=['actions'], observation_name=['qpos'], camera_names=['main_left', 'main_right', 'wrist'], chunk_size=60):
    hdf5_files = sorted(find_all_hdf5(dataset_dir, skip_mirrored_data=True))
    for file_idx, hdf5_path in tqdm(enumerate(hdf5_files)):
        with h5py.File(hdf5_path, 'r') as root:
            action_data = []
            action_mask_data = []
            for name in action_name:
                action_data.append(root[f'/{name}'][()])
                item_name = name.split('/')[-1]
                action_mask_data.append(root[f'/masks/act_{item_name}'][()])
            action = np.concatenate(action_data, axis=-1)
            action_mask = np.concatenate(action_mask_data, axis=-1)
            original_action_shape = action.shape
            episode_len = original_action_shape[0]
            # get observation at start_ts only
            if len(observation_name) > 0:
                observation_data = []
                for name in observation_name:
                    try:
                        observation_data.append(root[f'/observations/{name}'][()])
                    except:
                        assert name == 'proprioceptions/eef_to_body'
                        observation_data.append(root[f'/observations/proprioceptions/relative'][()])
                qpos = np.concatenate(observation_data, axis=-1)
            else:
                qpos = np.zeros([original_action_shape[0],40])[()]
                
            image_dict = dict()
            for cam_name in camera_names:
                if 'left' in cam_name or 'right' in cam_name:
                    base_cam_name = cam_name.split('_')[0]
                    full_image = root[f'/observations/images/{base_cam_name}'][()]
                    w  = full_image.shape[2] // 2
                    if 'left' in cam_name:
                        image_dict[cam_name] = full_image[:, :, :w, :]
                    elif 'right' in cam_name:
                        image_dict[cam_name] = full_image[:, :, w:, :]
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                
                    # cam_name = cam_name.split('_')[0]
                    # image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                # else:
                #     image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                
            lang = task_name
            steps = []
            for idx in range(episode_len):
                obs_dict = {"state": qpos[idx]}
                obs_dict.update({'image' + k: v[idx] for k, v in image_dict.items() if v.shape[0] > 0})
                step = {
                    "observation": obs_dict,
                    "action": action[idx],
                    "action_mask": action_mask,
                    "language_instruction": lang,
                }
                steps.append(OrderedDict(step))
            data_dict = {"steps": steps}
            yield data_dict
    

class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(self, env_names, episode_num, save_video=False):
        pass

    @torch.no_grad()
    def run(self, policy, save_video=False, gui=False, video_postfix="", seed=233, env_name=None, **kwargs):
        return 0, 0 # success (boolean), total_reward (float)
