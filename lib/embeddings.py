import math
import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
      https://arxiv.org/pdf/1907.05321.pdf — Time2Vec: Learning a Vector Representation of Time
      Also the attention you need paper does this
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
