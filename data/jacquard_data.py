import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random

import numpy as np
from PIL import Image
import pickle


from .base_grasp_data import BaseGraspDataset

from .utils import grasp_utils as gu
from .utils import image_utils as iu


class JacquardDataset(BaseGraspDataset):
    """
    Dataset wrapper for the Jacquard dataset.
    """
    def __init__(self, root, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)

        import glob
        # graspf = glob.glob(os.path.join(file_path, '*', '*_grasps.txt'))
        graspf = glob.glob('/SSDc/jongwon_kim/Datasets/Jacquard_Dataset' + '/*/*/' + '*_grasps.txt')
        graspf.sort()
        l = len(graspf)
        print("len jaccquard:", l)


        if self.seen:
            with open(os.path.join('split/jacquard/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            graspf = list(filter(lambda x: x.split('.')[0].split('/')[-1].split("_")[0] + "_" + x.split('.')[0].split('/')[-1].split("_")[1] in idxs, graspf))
            # graspf = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, graspf))
            
            split = int(np.floor(0.9 * len(graspf)))
            if self.train:
                graspf = graspf[:split]
                
            else:
                graspf = graspf[split:]
        
        else:
            with open(os.path.join('split/jacquard/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            graspf = list(filter(lambda x: x.split('.')[0].split('/')[-1].split("_")[0] + "_" + x.split('.')[0].split('/')[-1].split("_")[1] in idxs, graspf))


        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(root))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        fl = len(graspf)
        # print("len filtered jaccquard:", fl)


        depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        maskf = [f.replace('perfect_depth.tiff', 'mask.png') for f in depthf]


        self.grasp_files = graspf
        self.depth_files = depthf
        self.rgb_files = rgbf
        self.mask_files = maskf

        # when want to use length
        # self.grasp_files = graspf[int(l*start):int(l*end)]
        # self.depth_files = depthf[int(l*start):int(l*end)]
        # self.rgb_files = rgbf[int(l*start):int(l*end)]
        # self.mask_files = maskf[int(l*start):int(l*end)]


        if self.seen:
            pass
        else:
            pass
        
        
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = gu.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = iu.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = iu.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_mask(self, idx, rot=0, zoom=1.0): 
        mask_image = iu.Mask.from_file(self.mask_files[idx])
        mask_image.rotate(rot)
        mask_image.zoom(zoom)
        mask_image.resize((self.output_size, self.output_size))
        mask_image.normalise()
        return mask_image.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])


