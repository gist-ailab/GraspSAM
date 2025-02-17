from .jacquard_data import JacquardDataset
from .ocid_grasp_data import OCIDGraspDataset
from .grasp_anything_data import GraspAnythingDataset
from .grasp_anywhere_data import GraspAnywhereDataset


def load_data(args):
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, 
                                        random_rotate=True, random_zoom=True,
                                        start=0.0, end=0.9, seen=args.seen)
        test_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False)

    # TODO: change args when evaluate graspsam
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.01)
        test_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.99, end=1.0)

    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=args.seen)
        test_dataset = GraspAnythingDataset(root=root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False)
    
    elif args.dataset_name == "grasp_anywhere":
        root="/ailab_mat/dataset/Grasp-Anything/"
        add_d_root="/SSDc/Grasp-Anything++/"
        train_dataset = GraspAnywhereDataset(root=root, include_mask=True, include_prompt=True, start=0.0, end=0.05, add_d_root=add_d_root)
        test_dataset = GraspAnywhereDataset(root=root, include_mask=True, include_prompt=True, start=0.99, end=1.0, add_d_root=add_d_root)    


    else:
        raise("dataset name should be in [jacquard, vmrd, ocid-grasp, cornell, grasp-anything, graspnet]")

    return train_dataset, test_dataset