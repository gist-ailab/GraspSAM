import os
import time
import argparse
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F

from data.jacquard_data import JacquardDataset
from data.grasp_anything_data import GraspAnythingDataset


from model.planar_grasp_sam import PlanarGraspSAM
from utils.utils import TrainProgress

from skimage.filters import gaussian
from data.utils.grasp_utils import *

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    if len(q_img.shape) == 3:
        q_img = F.interpolate(q_img.unsqueeze(0), size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img.unsqueeze(0), size=(1024, 1024))[0]
        width_img = F.interpolate(width_img.unsqueeze(0), size=(1024, 1024))[0]
        
    elif len(q_img.shape) == 4:
        q_img = F.interpolate(q_img, size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img, size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img, size=(1024, 1024))[0]
        width_img = F.interpolate(width_img, size=(1024, 1024))[0]

    width_scale = 512
    q_img = q_img.data.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).data.detach().cpu().numpy().squeeze()
    width_img = width_img.data.detach().cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def cal_grasp_ious(test_dataset, model, args):
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, pin_memory=False, 
                                               num_workers=4, shuffle=True)
    ld = len(test_loader)
    results = {"correct": 0, "failed": 0, 
               "g_loss":0, 
               "g_losses":{
                "p_loss": 0,
                "cos_loss": 0,
                "sin_loss": 0,
                "width_loss": 0,
                },}
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, masks, grasps, didx, rot, zoom_factor = data
            images = images.to(args.device)    
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]
            
        
            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps    
                
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)    
            
            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)


            loss = lossd["g_loss"]
            results['g_loss'] += loss.item() / ld
            for ln, l in lossd['g_losses'].items():
                if ln not in results['g_losses']:
                    results['g_losses'][ln] = 0
                results['g_losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            success = calculate_iou_match(q_out, ang_out, 
                                          test_dataset.get_gtbb(didx, rot, zoom_factor), 
                                          no_grasps=1, 
                                          grasp_width=w_out)
            
            if success:
                results["correct"] += 1
            else:
                results["failed"] += 1
            
            success_rate = 100 * results["correct"] / (results["correct"] + results["failed"])
    return results["g_loss"], success_rate


def setup_model(model_type, sam_encoder_type, prompt_mode):
    if model_type == "bs_grasp_sam":
        model = PlanarGraspSAM(sam_encoder_type=sam_encoder_type, num_layers=0)

    else:
        raise("please input correct model type")

    return model


def main(args):
    is_save = args.save
    
    if is_save:
        if args.seen:
            args.exp_name = "seen" + "_" + args.sam_encoder_type + "_" + args.prompt
            
        else:
            args.exp_name = "total" + "_" + args.sam_encoder_type + "_" + args.prompt
            
        exp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        save_path = os.path.join(args.save_dir, args.exp_name, args.dataset_name, '{}'.format(exp_time))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
        f = open(os.path.join(save_path, "info.txt"), 'w')
        f.write(str(args))
        f.close()
        

    args.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset_name == "jacquard":
        train_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                        random_rotate=False, random_zoom=False,
                                        start=0.0, end=0.9, seen=args.seen, train=True)
        test_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False, train=False)
    
    elif args.dataset_name == "grasp_anything":
        train_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=args.seen, train=True)
        test_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False, train=False)
            
    print("train_dataset size : {}".format(train_dataset.__len__()))
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, num_workers=4, shuffle=True)
    
    
    model = setup_model(model_type="bs_grasp_sam", sam_encoder_type=args.sam_encoder_type, prompt_mode=args.prompt)

    if args.resume_ckp:
        state_dict = torch.load(args.resume_ckp, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)


    model.to(args.device)
    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, 1e-7)
    
    start_epoch = 0
    
    if args.resume_ckp:
        start_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])

    print("-"*80)
    

    best_success_rate = 0.0
    for epoch in range(start_epoch, args.epochs):
        
        t_progress = TrainProgress(train_loader, epoch)
        for i, data in enumerate(train_loader):
            images, masks, grasps, _, _, _ = data
        
            images = images.to(args.device)
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]

            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps          
            
            optimizer.zero_grad()
            
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)

            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            loss = lossd["loss"]
            loss.backward()

            optimizer.step()
            
            t_progress.progress_update(lossd, args.batch_size)
            if i % args.print_freq == 0:
                t_progress.progress.display(i)
                
        scheduler.step()
        print("-"*80)


        if args.validate:
            g_loss, success_rate = cal_grasp_ious(test_dataset, model, args)
            print("epoch : {} / g_loss : {:.4f} / success_rate : {:.4f}".format(epoch, g_loss, success_rate))

            if args.save: 
                if success_rate > best_success_rate or epoch == 0 or (epoch % 10) == 0:
                    best_success_rate = success_rate
                    save_dict = {
                        "epoch" : epoch,
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "scheduler" : scheduler.state_dict(),
                    }
                    
                    torch.save(save_dict, os.path.join(save_path, "epoch_%02d_sr_%0.2f.pth" % (epoch, success_rate)))
                    print("save best model in {}".format(save_path))
        else:
            if args.save:
                save_dict = {
                        "epoch" : epoch,
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "scheduler" : scheduler.state_dict(),
                    }
                
                torch.save(save_dict, os.path.join(save_path, "epoch{}.pth".format(str(epoch))))
                print("save model in {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-encoder-type", type=str, default="eff_vit_t_w_ad")
    parser.add_argument("--gpu-num", type=int, default=6, help="gpu id number")

    parser.add_argument("--dataset-name", type=str, default="jacquard", help="dataset name")
    parser.add_argument("--root", type=str, help="dataset root")
    parser.add_argument("--seen", action='store_true')
    parser.add_argument("--prompt", type=str, default='default')
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=50)
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="final_result")
    parser.add_argument("--resume-ckp", type=str, default=None)

    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--split", default=0.9, type=float)

    args = parser.parse_args()

    main(args)





