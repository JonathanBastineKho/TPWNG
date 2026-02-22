import torch
from torch import nn
import torch.nn.functional as F

class DistributionalInconsistencyLoss(nn.Module):
    """
    Distributional Inconsistency Loss (DIL)
    
    Formula: 
    L_dil = (1/M) * sum( (S_aa · S_an) / (||S_aa||_2 * ||S_an||_2) )
    """
    def __init__(self):
        super().__init__()

    def forward(self, S_aa: torch.Tensor, S_an: torch.Tensor):
        # Calculate cosine similarity along the temporal dimension (F)
        # By taking dim=1, we treat the sequence of F frames as a single vector.
        # F.cosine_similarity automatically computes the dot product divided by the L2 norms.
        cos_sim = F.cosine_similarity(S_aa, S_an, dim=1)
        
        # Average the loss across the batch of M abnormal videos
        l_dil = cos_sim.mean()
        
        return l_dil