import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ClipLoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, z_i, z_hat_i, z_j, z_hat_j):
        logits_i_j = torch.matmul(z_i, z_hat_j.T) / self.temperature
        logits_j_i = torch.matmul(z_j, z_hat_i.T) / self.temperature

        labels = torch.arange(z_i.size(0)).to(z_i.device)
        
        loss_i_j = F.cross_entropy(logits_i_j, labels)
        loss_j_i = F.cross_entropy(logits_j_i, labels)
        
        return (loss_i_j + loss_j_i) / 2