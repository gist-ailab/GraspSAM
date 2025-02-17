import glob
import os
import re

import pickle
import torch
import numpy as np

from .base_grasp_data import BaseGraspDataset

from .utils import grasp_utils as gu
from .utils import image_utils as iu



class GraspAnythingDataset(BaseGraspDataset):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, root, start=0.0, end=1.0,  ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        grasp_files = glob.glob(os.path.join(root, 'grasp_label_positive', '*.pt'))


        if self.seen:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)
            
            grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, grasp_files))
            split = int(np.floor(0.9 * len(grasp_files)))
            if self.train:
                self.grasp_files = grasp_files[:split]
                
            else:
                self.grasp_files = grasp_files[split:]
        
        
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, grasp_files))


        l = len(self.grasp_files)

        self.grasp_files.sort()
   
        # self.grasp_files = self.grasp_files[int(l*start):int(l*end)]
        
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(root))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        gtbbs = gu.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        # gtbbs.resize(self.crop_size, self.output_size)
   
        return gtbbs


    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = re.sub(r"_\d+\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label_positive", "image")
        rgb_img = iu.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img.img = rgb_img.img[...,::-1]
        
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    
    def get_mask(self, idx, rot=0, zoom=1.0):
        mask_file = self.grasp_files[idx].replace("grasp_label_positive", "mask").replace(".pt", ".npy")
        mask_image = iu.Mask.from_npy_file(mask_file)
        mask_image.rotate(rot)
        mask_image.zoom(zoom)
        mask_image.resize((self.output_size, self.output_size))
        return mask_image.img
