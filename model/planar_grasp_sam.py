import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple

from model.utils import LayerNorm2d, MLP, masks_sample_points, masks_to_boxes, masks_noise, dice_loss


from model.build_grasp_sam import build_grasp_sam

class SamEncoder(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, sam_encoder_type):
        super(SamEncoder, self).__init__()
        
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.sam_encoder_type = sam_encoder_type

    def forward(self, batched_input):
        
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        
        batched_output = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            
            if "eff" in self.sam_encoder_type:
                sparse_embeddings = self.prompt_encoder(
                    coords=image_record["point_coords"],
                    labels=image_record["point_labels"]
                )
                
                batched_output.append(
                    {
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": self.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":None,
                    }
                )
                
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                batched_output.append(
                    {
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": self.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":dense_embeddings,
                    }
                )

        return batched_output, interm_embeddings

class ConvGraspHeader(nn.Module):
    def __init__(self, num_layers=3):
        super(ConvGraspHeader, self).__init__()

        self.layer = []
        for i in range(0, num_layers):
            self.layer.append(nn.Sequential(nn.Conv2d(4, 4, kernel_size=1),
                              nn.BatchNorm2d(4),
                              nn.ReLU()))
        self.early_conv = nn.Sequential(*self.layer)

        self.point_predictor = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.width_predictor = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        
        self.cos_predictor   = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.sin_predictor   = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        
        self.fusion          = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):       # x: [1, 4, 256, 256]
        
        if len (self.layer) > 0:
            x = self.early_conv(x)
        # pos_out   = F.sigmoid(self.point_predictor(x[:,0]))
        # width_out = F.sigmoid(self.width_predictor(x[:,1]))
        pos_out   = F.relu(self.point_predictor(x[:,0]))
        width_out = F.relu(self.width_predictor(x[:,1]))
    
        cos_out   = self.cos_predictor(x[:,2])
        sin_out   = self.sin_predictor(x[:,3])

        return pos_out, cos_out, sin_out, width_out 

class SamDecoder(nn.Module):
    def __init__(self, mask_decoder, sam_encoder_type, grasp_header_type, num_layers):
        super(SamDecoder, self).__init__()

        self.mask_decoder = mask_decoder
        self.sam_encoder_type = sam_encoder_type
        
        self.transformer = self.mask_decoder.transformer
        self.iou_token = self.mask_decoder.iou_token
        self.mask_tokens = self.mask_decoder.mask_tokens
        self.num_mask_tokens = self.mask_decoder.num_mask_tokens
        
        
        if "eff" in sam_encoder_type:
            self.iou_prediction_head = self.mask_decoder.iou_prediction_head
            
            
            def output_upscaling(upscaled_embedding):
                for upscaling_layer in self.mask_decoder.final_output_upscaling_layers:
                    upscaled_embedding = upscaling_layer(upscaled_embedding)
                return upscaled_embedding

            self.output_upscaling = output_upscaling
              
            
            self.output_hypernetworks_mlps = self.mask_decoder.output_hypernetworks_mlps
            
        else:
            self.iou_prediction_head = self.mask_decoder.iou_prediction_head
            self.output_upscaling = self.mask_decoder.output_upscaling
            self.output_hypernetworks_mlps = self.mask_decoder.output_hypernetworks_mlps
        
        
        self.grasp_header_type = grasp_header_type
        
        self.num_grasp_queries = 4 
        transformer_dim=256
        
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280,"vit_t":160, "vit_t_w_ad":160,
                        "eff_vit_t":192, "eff_vit_s":384, "eff_vit_t_w_ad":192, "eff_vit_s_w_ad":384}
        
        vit_dim = vit_dim_dict[sam_encoder_type]
        
                
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        
        self.num_grasp_tokens = 4
        self.grasp_token = nn.Embedding(self.num_grasp_tokens, transformer_dim)
        self.grasp_mlp   = nn.ModuleList([
                            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                            for i in range(self.num_grasp_tokens)])
        
        self.num_hq_tokens = self.num_mask_tokens + 1
        self.num_total_tokens = self.num_hq_tokens + self.num_grasp_tokens


        
        if 'vit_t' in sam_encoder_type:
            mid_compress_dim = transformer_dim//2
            out_compress_dim = transformer_dim//8
        
        else:
            mid_compress_dim = transformer_dim
            out_compress_dim = transformer_dim//8
        
        
        self.compress_vit_feat = nn.Sequential(
                                nn.ConvTranspose2d(vit_dim, mid_compress_dim, kernel_size=2, stride=2),
                                LayerNorm2d(mid_compress_dim),
                                nn.GELU(), 
                                nn.ConvTranspose2d(mid_compress_dim, out_compress_dim, kernel_size=2, stride=2))
        
        self.compress_vit_feat_g = nn.Sequential(
                                nn.ConvTranspose2d(vit_dim, mid_compress_dim, kernel_size=2, stride=2),
                                LayerNorm2d(mid_compress_dim),
                                nn.GELU(), 
                                nn.ConvTranspose2d(mid_compress_dim, out_compress_dim, kernel_size=2, stride=2))
        
        
        
        self.embedding_encoder_mask = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim//4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim//4, transformer_dim//8, kernel_size=2, stride=2))


        self.embedding_maskfeature = nn.Sequential(
                                nn.Conv2d(transformer_dim//8, transformer_dim // 4, 3, 1, 1), 
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim//4, transformer_dim//8, 3, 1, 1))

        
        self.embedding_encoder_grasp = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2))
        

        self.embedding_graspfeature = nn.Sequential(
                                nn.Conv2d(transformer_dim//8, transformer_dim//4, 3, 1, 1), 
                                LayerNorm2d(transformer_dim//4),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim//4, transformer_dim//8, 3, 1, 1))

        
        
        
        self.grasp_header = ConvGraspHeader(num_layers=num_layers)
        
    def predict_grasps(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        hq_feature,
        grasp_feature
    ):

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight, self.grasp_token.weight], dim=0) #[7, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)                                       #[1, 7, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)                                                              #[1, 11, 256]

    
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)                                                          #[1, 256, 64, 64]
        
        if "eff" in self.sam_encoder_type:
            pass
        else:
            src = src + dense_prompt_embeddings
        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)            # [B, 21, 256] /[1, 4096, 256]
        iou_token_out = hs[:, 0, :]                                 # [1, 256]
        mask_tokens_out = hs[:, 1 : (1 + self.num_total_tokens), :]

        
        src = src.transpose(1, 2).view(b, c, h, w)   
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        upscaled_embedding_grasp = self.embedding_graspfeature(upscaled_embedding_sam) + grasp_feature
         
        hyper_in_list = []
        for i in range(self.num_total_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            elif i == 4:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
            elif i > 4:
                hyper_in_list.append(self.grasp_mlp[i-5](mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        
        
        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        # grasp = (hyper_in[:, -1] @ upscaled_embedding_grasp.view(b, c, h * w)).view(b, -1, h, w)           #[1, 1, 256, 256]
        grasp_maps = (hyper_in[:,5:] @ upscaled_embedding_grasp.view(b, c, h * w)).view(b, -1, h, w)
   
        grasp = self.grasp_header(grasp_maps)

        return masks, iou_pred, grasp
    

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        # print(len(interm_embeddings))
        # print(interm_embeddings[0].shape)
        # exit()
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder_mask(image_embeddings) + self.compress_vit_feat(vit_features)
        grasp_features= self.embedding_encoder_grasp(image_embeddings) + self.compress_vit_feat_g(vit_features) 

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        grasps_pos = []
        grasps_cos = []
        grasps_sin = []
        grasps_width = []
        
        for i_batch in range(batch_len):
            mask, iou_pred, grasp_pred = self.predict_grasps(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0),
                grasp_feature = grasp_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
            grasps_pos.append(grasp_pred[0])
            grasps_cos.append(grasp_pred[1])
            grasps_sin.append(grasp_pred[2])
            grasps_width.append(grasp_pred[3])
            
            
        grasps_poses = torch.cat(grasps_pos,0)
        grasps_coses = torch.cat(grasps_cos,0)
        grasps_sines = torch.cat(grasps_sin,0)
        grasps_widthes = torch.cat(grasps_width,0)
        
        grasps = [grasps_poses, grasps_coses, grasps_sines, grasps_widthes]
    
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        if multimask_output:
            mask_slice = slice(1,self.num_hq_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)       
                 
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]            

        
        masks_hq = masks[:,slice(self.num_hq_tokens-1, self.num_hq_tokens), :, :]
        
 
        return grasps, masks_hq

class PlanarGraspSAM(nn.Module):
    def __init__(self, sam_encoder_type, vis=False, num_layers=0):
        super(PlanarGraspSAM, self).__init__()
        
        self.image_encoder, self.prompt_encoder, self.mask_decoder = build_grasp_sam(sam_encoder_type, adapter=False)
        
        self.encoder = SamEncoder(self.image_encoder, self.prompt_encoder, sam_encoder_type)
        
        self.decoder = SamDecoder(self.mask_decoder, sam_encoder_type, grasp_header_type="conv", num_layers=num_layers)
        
        self.sam_encoder_type = sam_encoder_type
        self.vis = vis # for inference
        
    def total_forward(self, imgs, targets, input_type="10point"):
        
        if input_type =='default':
            input_keys = ['box','point']
            k = 10
            
        elif input_type == '1point':
            input_keys = ['point']
            k = 1
            
        elif input_type == '3point':
            input_keys = ['point']
            k = 3
            
        elif input_type == '5point':
            input_keys = ['point']
            k = 5
        
        elif input_type == '10point':
            input_keys = ['point']
            k = 10
            
        elif input_type == "box":
            input_keys = ['box']
            k = 1
        
        
        masks = targets["masks"]

    
        labels_box = masks_to_boxes(masks*255)
        labels_points = masks_sample_points(masks*255, k=k)

    
        batched_input = []
        for b_i in range(len(imgs)):
            
            dict_input = dict()
            dict_input['image'] = imgs[b_i]
            input_type = random.choice(input_keys)
            if input_type == 'box':
                dict_input['boxes'] = labels_box[b_i:b_i+1]
                
                if "eff" in self.sam_encoder_type:
                    x1, y1, x2, y2 = labels_box[b_i:b_i+1][0]
                    dict_input['point_coords'] = torch.tensor([[x1,y1],[x2,y2]], device=labels_box.device)[None,:]
                    dict_input['point_labels'] = torch.tensor([2,3], device=labels_box.device)[None,:]

                
            elif input_type == 'point':
                point_coords = labels_points[b_i:b_i+1]
                
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]

            else:
                raise NotImplementedError
            
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.append(dict_input)
     
        batched_output, interm_embeddings = self.encoder(batched_input)
        
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        
        results = self.decoder(
                    image_embeddings  = encoder_embedding, 
                    image_pe          = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings  = dense_embeddings,
                    multimask_output=False,
                    interm_embeddings = interm_embeddings,
                    )
        
        if self.vis:

            if input_type=="point":
                prompt_input = batched_input[0]['point_coords']
            elif input_type=="box":
                prompt_input = batched_input[0]['boxes']

            return results, prompt_input
        
            
        else:
            return results
    

    def weighted_mse(self, preds, targets, masks, weight=0.01):
        
        preds = preds.view(-1, 256**2)
        targets = targets.view(-1, 256**2)
        masks = masks.view(-1, 256**2)
        
        loss = 0
        for pd, gt, mk in zip(preds, targets, masks):
            fore_pred = pd[mk>0].clone()
            fore_gt = gt[mk>0]
            fore = F.mse_loss(fore_pred, fore_gt)
        
            back_pred = pd[mk==0].clone() 
            back_gt = gt[mk==0]
            back =  F.mse_loss(back_pred, back_gt)
            
            batch_loss = fore + (back * weight)
            loss += batch_loss
        
        mean_loss = loss / masks.shape[0]
            
        return mean_loss
        
        
    def compute_loss(self, grasp_pred, mask_pred, target, g_s, m_s, p, c, s, w):
        g_s = g_s
        m_s = m_s
    
        grasps_gt = target["grasps"]
        masks_gt = target["masks"]
        masks_gt = F.interpolate(masks_gt, size=(256, 256))
 
        pos_pred, cos_pred, sin_pred, width_pred = grasp_pred
        
        pos, cos, sin, width = grasps_gt
        
        pos   = F.interpolate(pos, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        cos   = F.interpolate(cos, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        sin   = F.interpolate(sin, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        width = F.interpolate(width, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)    
    
        p_loss = self.weighted_mse(pos_pred, pos, masks_gt)
        cos_loss = self.weighted_mse(cos_pred, cos, masks_gt)
        sin_loss = self.weighted_mse(sin_pred, sin, masks_gt)
        width_loss = self.weighted_mse(width_pred, width, masks_gt)
    
        grasp_loss = p*p_loss + c*cos_loss + s*sin_loss + w*width_loss
    
        d_loss = dice_loss(mask_pred, masks_gt)
        bce_loss = F.binary_cross_entropy_with_logits(mask_pred, masks_gt)
        mask_loss = d_loss + bce_loss

        total_loss = g_s*grasp_loss + m_s*mask_loss


        return {
            "loss": total_loss,
            "mask_loss" : mask_loss,
            "g_loss" : grasp_loss,
            "g_losses":{
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss,
                },
            "pred":{
                "pos" : pos_pred,
                "cos" : cos_pred,
                "sin" : sin_pred,
                "width" : width_pred

            }
        }
    
