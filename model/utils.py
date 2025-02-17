import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    
    targets = targets.flatten(1)
 
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
   
    return loss.mean()


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
    
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks>128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks>128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

    
    
def masks_sample_points(masks, k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    # k = 10
    samples = []
    for b_i in range(len(masks)):
        
        select_mask = (masks[b_i]>128)
        x_idx = torch.masked_select(x,select_mask)
        y_idx = torch.masked_select(y,select_mask)
        
        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:,None],samples_y[:,None]),dim=1)
        
        samples.append(samples_xy)
        
    samples = torch.stack(samples)
    
    return samples


def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector,  8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector