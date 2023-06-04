from torch import nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):

    def __init__(self, w2v_model):
        super(ReconstructionLoss, self).__init__()
        self.w2v_model = w2v_model

    def forward(self, word_idx, ref_vector, freq=None, normalize=False):
        vec = self.w2v_model(word_idx)
        if normalize:
            vec = F.normalize(vec)
            ref_vector = F.normalize(ref_vector)
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(vec, ref_vector).mean(axis=1)
        if freq != None:
            loss = loss * freq.log()

        return loss.mean()
