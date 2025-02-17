import torch
import torch.nn as nn
from collections import OrderedDict

from model.segment_anything import sam_model_registry
from model.efficient_sam.build_efficient_sam import eff_sam_model_registry

from model.segment_anything.modeling import MaskDecoder
from model.segment_anything.modeling import TwoWayTransformer

from model.efficient_sam.efficient_sam_decoder import MaskDecoder as EffMaskDecoder
from model.efficient_sam.two_way_transformer import TwoWayTransformer as EffTwoWayTransformer


def load_sam_encoder(sam_encoder_type, checkpoint_dict, adapter):
    checkpoint_path = checkpoint_dict[sam_encoder_type]
    
    if "eff" in sam_encoder_type:
        print("efficient sam encoder is loading")
        sam = eff_sam_model_registry[sam_encoder_type](checkpoint=checkpoint_path)
    
    else:
        print("default sam encoder is loading")
        sam = sam_model_registry[sam_encoder_type](checkpoint=checkpoint_path)
      
    image_encoder  = sam.image_encoder
    prompt_encoder = sam.prompt_encoder
    
    return image_encoder, prompt_encoder


def load_sam_decoder(sam_encoder_type, checkpoint_dict):
    checkpoint_path = checkpoint_dict[sam_encoder_type]
    
    if "eff" in sam_encoder_type:
        print("efficient sam decoder is loading")
        print(sam_encoder_type)
        transformer = EffTwoWayTransformer(
                                    depth=2,
                                    embedding_dim=256,
                                    mlp_dim=2048,
                                    num_heads=8,
                                    activation=nn.GELU,
                                    normalize_before_activation= False,
                                    )
        
        decoder = EffMaskDecoder(transformer_dim=256,
                    transformer=transformer,
                    num_multimask_outputs=3,
                    activation=nn.GELU,
                    iou_head_depth= 2,# eff sam's decoder depth is 2
                    iou_head_hidden_dim= 256,
                    normalization_type = "layer_norm",
                    normalize_before_activation=False,
                    upscaling_layer_dims=[64, 32]
                    )
        

        state_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
   
        for k, v in state_dict["model"].items():
            # print(k)
            if 'mask_decoder' in k:
                # print(k[13:])
                new_state_dict[k[13:]] = v
        
        decoder.load_state_dict(new_state_dict)

    else:
        print("default sam decoder is loading")
        
        transformer = TwoWayTransformer(
                                    depth=2,
                                    embedding_dim=256,
                                    mlp_dim=2048,
                                    num_heads=8,
                                    activation=nn.GELU,
                                    )
        
        decoder = MaskDecoder(transformer_dim=256,
                    transformer=transformer,
                    num_multimask_outputs=3,
                    activation=nn.GELU,
                    iou_head_depth= 3,
                    iou_head_hidden_dim= 256)
        
        
        if "vit_t" in  sam_encoder_type:
    
            state_dict = torch.load(checkpoint_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'mask_decoder' in k:
                    new_state_dict[k[13:]] = v
                    
            decoder.load_state_dict(new_state_dict)
            
        else:
            state_dict = torch.load(checkpoint_path)
            decoder.load_state_dict(state_dict)
            
    return decoder
    
    
def build_grasp_sam(sam_encoder_type="vit_t", 
                    freeze_encoder=True, freeze_decoder=True, freeze_prompt_encoder=True,
                    adapter=False):
    
    assert sam_encoder_type in ["vit_b", "vit_l", "vit_h", "vit_t", "vit_t_w_ad", 
                                "eff_vit_t", "eff_vit_s", "eff_vit_t_w_ad", "eff_vit_s_w_ad"]
    checkpoint_dict = {
                        'vit_t': "pretrained_checkpoint/mobile_sam.pt",
                        "vit_b":"pretrained_checkpoint/sam_vit_b_01ec64.pth",
                        "vit_l":"pretrained_checkpoint/sam_vit_l_0b3195.pth",
                        'vit_h':"pretrained_checkpoint/sam_vit_h_4b8939.pth",

                        'vit_t_w_ad':"pretrained_checkpoint/mobile_sam.pt",
                        
                        "eff_vit_t":"pretrained_checkpoints/efficient_sam/efficient_sam_vitt.pt",
                        "eff_vit_s":"pretrained_checkpoints/efficient_sam/efficient_sam_vits.pt",

                        "eff_vit_t_w_ad":"pretrained_checkpoints/efficient_sam/efficient_sam_vitt.pt",
                        "eff_vit_s_w_ad":"pretrained_checkpoints/efficient_sam/efficient_sam_vits.pt",

                        }
    
    image_encoder, prompt_encoder = load_sam_encoder(sam_encoder_type, checkpoint_dict, adapter)
    mask_decoder                  = load_sam_decoder(sam_encoder_type, checkpoint_dict)
    
    if freeze_encoder:
        for param in image_encoder.parameters():
            param.requires_grad = False
                
    if freeze_prompt_encoder:
        for param in prompt_encoder.parameters():
            param.requires_grad = False
            
    if freeze_decoder:
        for param in mask_decoder.parameters():
            param.requires_grad = False

    return image_encoder, prompt_encoder, mask_decoder

