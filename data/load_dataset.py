import torch
from torch.utils.data.distributed import DistributedSampler

from data.jacquard_data import JacquardDataset
from data.ocid_grasp_data import OCIDGraspDataset
from data.grasp_anything_data import GraspAnythingDataset


def get_dataset(args):
    
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.0, end=0.9)
        test_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.9, end=1.0)
    
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.9)
        test_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.9, end=1.0)
    
    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.0, end=0.9)
        test_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.9, end=1.0)
        
    print("train_dataset size : {}".format(train_dataset.__len__()))
    print("test_dataset size : {}".format(test_dataset.__len__()))
    
    
    return train_dataset, test_dataset


def get_train_loader(args):
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.0, end=0.9)
    
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.9)
    
    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.0, end=0.9)
        
    print("train_dataset size : {}".format(train_dataset.__len__()))
    
    
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        
        num_workers_ = 1
        if(args.batch_size>1):
            num_workers_ = 2
        if(args.batch_size>4):
            num_workers_ = 4
        if(args.batch_size>8):
            num_workers_ = 8
            
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                        num_workers=num_workers_, pin_memory=True, sampler=train_sampler)

    else:
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=4, pin_memory=False)
        
    return train_loader


