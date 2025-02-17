# This code is from "https://github.com/WangShaoSUN/grasp-transformer/tree/main"
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torch.nn.functional as F

import cv2
import random


class BaseGraspDataset(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=1024, crop_size=224, include_depth=False, include_mask=False, include_rgb=True,
                 include_prompt=False,
                 random_rotate=False, random_zoom=False, input_only=False, grasp_map_split=False, multi_masks=False, 
                 seen=False, train=True):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_rgb = include_rgb
        self.include_prompt = include_prompt

        
        self.grasp_map_split = grasp_map_split
        self.multi_masks = multi_masks
        self.seen = seen
        self.crop_size = crop_size
        self.train = train
        

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()
    
    def get_mask(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_prompt(self, idx):
        raise NotImplementedError()


    def __getitem__(self, idx, vis=False):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the mask
        if self.include_mask:
            mask = self.get_mask(idx, rot, zoom_factor)
            mask = torch.as_tensor(mask).unsqueeze(0)
        
        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        if self.include_prompt:
            prompt = self.get_prompt(idx)
            
        ocid = True
        if ocid:
            pos_img, ang_img, width_img = bbs.draw((self.crop_size, self.crop_size))
            pos_img = cv2.resize(pos_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            ang_img = cv2.resize(ang_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            width_img = cv2.resize(width_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        else:
            pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        # print(np.unique(width_img))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)
        

        if self.grasp_map_split:
            
            obj_ids = np.unique(mask)[1:]
            filtered_ids = []
            th = 5000
            for ids in obj_ids:
                i_mask = (mask.clone() == ids)
                m = pos * i_mask
                m = torch.sum(m)
                if m.item() < th:
                    continue
                filtered_ids.append(ids)
            # print(len(filtered_ids))

            if self.multi_masks:
                num_samples = np.random.randint(1, len(filtered_ids)+1)
                rand_idxs = np.random.choice(filtered_ids, size=num_samples)
                
                semantic_mask = torch.zeros_like(mask, dtype=torch.float32)

                for rand_idx in rand_idxs: 
                    semantic_mask += (mask.clone() == rand_idx) * 1.0

                mask = semantic_mask
            else:
                rand_idx = np.random.choice(filtered_ids, size=1)[0]
                mask = (mask.clone() == rand_idx) * 1.0
            

        if self.grasp_map_split:
            pos[mask==0.0] = 0
            cos[mask==0.0] = 1
            sin[mask==0.0] = 0
            width[mask==0.0] = 0
            
            
        grasps = [pos, cos, sin, width]
        
        if vis:
            # raw_image = Image.open(self.rgb_files[idx]).convert("RGB")
            if self.include_prompt:
                return x, mask, grasps, idx, rot, zoom_factor, bbs, prompt
            else: 
                return x, mask, grasps, idx, rot, zoom_factor, bbs
        else:
            if self.include_prompt:
                return x, mask, grasps, idx, rot, zoom_factor, prompt
            else:
                return x, mask, grasps, idx, rot, zoom_factor
        
        # return x, (pos, cos, sin, width), idx, rot, zoom_factor
        
    def __len__(self):
        return len(self.grasp_files)