from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class Normalize(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.float()
        image = F.normalize(image,
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        
        
        return image, target


class RandomRotate(nn.Module):
    def __init__(
        self,
        low=0,
        high=180,
        image_size = (1024, 1024),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()

        self.low = low
        self.high = high
        self.image_size =image_size
        self.interpolation = interpolation

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        
        self.angle = np.random.randint(low=self.low, high=self.high)
        center = (self.image_size[0]//2, self.image_size[1]//2)
          
        image = F.rotate(image, self.angle, center=center, interpolation=self.interpolation)

        if target is not None:
            if "masks" in target:
                target["masks"] = F.rotate(target["masks"], self.angle, center=center, interpolation=InterpolationMode.NEAREST)
                
            if "grasps" in target:
                radian = self.angle * np.pi / 180
                
                R = np.array([
                    [np.cos(-radian), -1*np.sin(-radian)],
                    [np.sin(-radian), np.cos(-radian)],
                ])
                
                
                for idx in range(len(target["grasps"])):
                    x, y, a, w, h = target["grasps"][idx]
                    
                    point = np.array([x, y]) - np.array(center)
                    x, y = (np.dot(R, point.T)).T + np.array(center)
                    
                    target["grasps"][idx] = torch.tensor([x, y, a-radian, w, h])
       
                
        return image, target
 
    
class RandomZoom(nn.Module):
    def __init__(
        self,
        low=0.8,
        high=1.0,
        image_size = (1024, 1024),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()

        self.low = low
        self.high = high
        
        self.image_size = image_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)
        
        self.factor = np.random.randint(low=self.low*10, high=self.high*10)/10
        
        center = np.array((self.image_size[0]//2, self.image_size[1]//2)).astype(int)
        
        sr = int(orig_height * (1 - self.factor)) // 2
        sc = int(orig_width * (1 - self.factor)) // 2
 
        

        image = image[:, sr:orig_height - sr, sc: orig_width - sc]
 
        
        image = F.resize(image, [orig_height, orig_width], interpolation=self.interpolation)
        

        if target is not None:
            if "masks" in target:
                
                target["masks"] = target["masks"][:, sr:orig_height - sr, sc: orig_width - sc]
                target["masks"] = F.resize(target["masks"], [orig_height, orig_width], interpolation=InterpolationMode.NEAREST)
            
            
            if "grasps" in target:
                
                T = np.array([
                        [1/self.factor, 0],
                        [0, 1/self.factor]
                    ])
                
                
                for idx in range(len(target["grasps"])):
                    x, y, a, w, h = target["grasps"][idx]
                    
                    point = np.array([x, y]) - center
                    x, y = (np.dot(T, point.T)).T + center

                    target["grasps"][idx] = torch.tensor([x, y, a, w*(1/self.factor), h*(1/self.factor)])
            
            
        return image, target
    