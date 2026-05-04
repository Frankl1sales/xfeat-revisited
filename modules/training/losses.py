# Loss functions implemented from the paper and modified from original

import torch
import torch.nn.functional as F
from third_party.alike_wrapper import extract_alike_kpts

def dual_softmax_loss(f1, f2, temp=0.2):
    s = (f1 @ f2.t()) * temp  # similarity matrix
    log_softmax_1 = F.log_softmax(s, dim=1)
    log_softmax_2 = F.log_softmax(s.t(), dim=1)

    with torch.no_grad():
        conf1 = torch.exp(log_softmax_1).max(dim=-1)[0]  # R̄1: max prob for each row
        conf2 = torch.exp(log_softmax_2).max(dim=-1)[0]  # R̄2: max prob for each column
    
    target = torch.arange(len(f1), device=f1.device)
    l_ds = F.nll_loss(log_softmax_1, target) + F.nll_loss(log_softmax_2, target)
    return l_ds, conf1, conf2

def reliability_loss(reliability_map1, reliability_map2, conf1, conf2):
    target = conf1 * conf2  # R̄1 ⊙ R̄2 (element-wise product)
    l_rel = torch.abs(torch.sigmoid(reliability_map1) - target) + \
            torch.abs(torch.sigmoid(reliability_map2) - target)
    return l_rel.mean()

def fine_loss(feats1, feats2, pts1, pts2, fine_matcher, stride=8):
    # Extract features at correspondence locations
    # feats1, feats2: [C, H, W] at 1/8 resolution
    # pts1, pts2: [N, 2] in feature map (1/8) coordinates
    
    fa = feats1[:, pts1[:,1].long(), pts1[:,0].long()].T  # [N, C]
    fb = feats2[:, pts2[:,1].long(), pts2[:,0].long()].T  # [N, C]
    
    # Concatenate features and pass through MLP
    fine_input = torch.cat([fa, fb], dim=-1)  # [N, 2C]
    logits = fine_matcher(fine_input)  # [N, 64] where 64 = 8×8
    
    # Convert feature map coordinates to pixel coordinates
    pts1_px = pts1 * stride  # [N, 2]
    pts2_px = pts2 * stride  # [N, 2]
    
    # Compute ground truth offset within 8×8 cell (subpixel offset)
    # The offset should be in range [0, 7] for both x and y
    offset1 = (pts1_px / stride) - (pts1_px / stride).long()  # [N, 2], fractional part
    offset1 = (offset1 * stride).long()  # Convert to [0, 7] integer grid
    
    # Convert to linear index in 8×8 grid
    gt_indices = offset1[:, 1] * 8 + offset1[:, 0]  # [N]
    
    # NLL loss over offset logits
    l_fine = F.cross_entropy(logits, gt_indices)
    return l_fine

def keypoint_loss(kpts, img):
    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0)
    img = img.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor(extract_alike_kpts(img), device=kpts.device)
        labels = torch.ones((H, W), dtype=torch.long, device=kpts.device) * 64  # dustbin
        offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
        offsets = offsets[:, 0] + 8*offsets[:, 1]
        labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

    kpts = kpts.view(-1, C)
    labels = labels.view(-1)

    # Balance negatives (dustbin) to avoid class imbalance
    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    if len(idxs_neg) > 0 and len(idxs_pos) > 0:
        perm = torch.randperm(idxs_neg.size(0))[:max(1, len(idxs_pos)//32)]
        idxs_neg = idxs_neg[perm]
        idxs = torch.cat([idxs_pos, idxs_neg])
        kpts = kpts[idxs]
        labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc = (labels == predicted)
        acc = acc.sum() / len(acc)

    kpts = F.log_softmax(kpts, dim=-1)
    l_kp = F.nll_loss(kpts, labels, reduction='mean')
    return l_kp, acc