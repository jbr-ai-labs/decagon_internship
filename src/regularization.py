import torch
import torch.nn as nn
from typing import Tuple


# Code from
# https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/utils.py
# with small changes
class WeightBasis(nn.Module):
    r"""Basis decomposition module.
    Basis decomposition is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:
    .. math::
        W_o = \sum_{b=1}^B a_{ob} V_b
    Each weight output :math:`W_o` is essentially a linear combination of basis
    transformations :math:`V_b` with coefficients :math:`a_{ob}`.
    If is useful as a form of regularization on a large parameter matrix. Thus,
    the number of weight outputs is usually larger than the number of bases.
    Parameters
    ----------
    shape : tuple[int]
        Shape of the basis parameter.
    num_bases : int
        Number of bases.
    num_outputs : int
        Number of outputs.
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 num_bases: int,
                 num_outputs: int):
        super(WeightBasis, self).__init__()
        self.shape = shape
        self.num_bases = num_bases
        self.num_outputs = num_outputs

        if num_outputs <= num_bases:
            raise ValueError('The number of weight outputs should be larger '
                             'than the number of bases.')

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, *shape))
        nn.init.xavier_uniform_(self.weight)
        # linear combination coefficients
        self.w_comp = nn.Parameter(torch.Tensor(self.num_outputs,
                                                self.num_bases))
        nn.init.xavier_uniform_(self.w_comp)

    def forward(self) -> torch.Tensor:
        r"""Forward computation
        Returns
        -------
        weight : torch.Tensor
            Composed weight tensor of shape ``(num_outputs,) + shape``
        """
        # generate all weights from bases
        weight = torch.matmul(self.w_comp, self.weight.view(self.num_bases, -1))
        return weight.view(self.num_outputs, *self.shape)


class BlockDiagDecomp(nn.Module):
    """
    Block diagonal decomposition module.
    It is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    Parameters
    ----------
    shape : tuple[int]
        Shape of the weight matrix need to decompose
    num_bases : int
        Number of blocks in decomposition.
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 num_bases: int):
        super(BlockDiagDecomp, self).__init__()
        self.shape = shape
        self.num_bases = num_bases
        self.submat_shape = shape[0] // self.num_bases, \
                            shape[1] // self.num_bases
        # last block can be smaller
        self.last_submat_shape = (
            shape[0] - self.num_bases * self.submat_shape[0],
            shape[1] - self.num_bases * self.submat_shape[1])
        self.weight = nn.Parameter(torch.Tensor(self.num_bases,
                                                *self.submat_shape))
        self.last_weight = nn.Parameter(torch.Tensor(*self.last_submat_shape))
        nn.init.xavier_uniform_(self.weight)
        if sum(self.last_submat_shape) > 0:
            nn.init.xavier_uniform_(self.last_weight)

    def forward(self):
        """
        Forward computation
        Returns
        -------
        weight : torch.Tensor
            Weight tensor (block-diagonal) with same shape as self.shape.
        """
        # Create matrix from blocks
        return torch.block_diag(*torch.unbind(self.weight, dim=0),
                                self.last_weight)