import torch
import logging, warnings
import string
import typing as tp
import gc

from adp import NumberEmbedder
import numpy as np
from torch import nn

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class TimeConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                ):
        super().__init__(output_dim, output_dim)

    def audio_matrix(self, begin, end, speed=1):
        length=self.output_dim
        linespace = torch.linspace(begin, end, int(length*speed))
        integer_part = linespace.floor().long()
        fractional_part = linespace - integer_part.float()
        
        water_matrix = torch.zeros((6, length))
        water_matrix[integer_part-1, torch.arange(int(length*speed))] = 1 - fractional_part
        water_matrix[integer_part, torch.arange(int(length*speed))] = fractional_part
        water_matrix[int(end)-1, int(length*speed):] = 1
        return water_matrix[:-1]

    def forward(self, floats: tp.List[tp.Tuple[float]], device=None) -> tp.Any:
        ret = []
        for i in floats:
            ret.append(self.audio_matrix(i[0],i[1]))
        floats = torch.stack(ret).to(device)
        return [floats, torch.ones(floats.shape[0], 1).to(device)]


class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:

        # Cast the inputs to floats
        floats = [float(x) for x in floats]

        floats = torch.tensor(floats).to(device)

        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]
if __name__ == '__main__':
    con = NumberConditioner(768)
    # print(con([(0,0),(1,0),(0,1)])[0].shape,con([(0,0),(1,0),(0,1)])[1])
    print(con([1])[0].shape,con([1])[1])