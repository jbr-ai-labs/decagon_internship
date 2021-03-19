import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.heterograph import DGLHeteroGraph
from typing import Dict, Tuple, Callable, Optional
from .regularization import WeightBasis, BlockDiagDecomp


# Used code from
# https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero
class RelGraphConvLayer(nn.Module):
    """
    Relational graph convolution layer.

    Parameters
    ----------
    ntype2in_feat_dim : dict[str, int]
        From node type to corresponding input feature size.
    out_feat_dim : int
        Output feature size.
    etype2ntypes : dict[str, tuple[str, str]]
        From edge type to begin and end nodes types.
    bias : bool, optional
        True if bias is added. Default: True.
    activation : callable, optional
        Activation function. Default: None
    dropout : float, optional
        Dropout rate. Default: 0.0
    regularizer : str
        Type of weight regularization. Default: None
        None = no regularization,
        'basis' = basis decomposition,
        'bdd' = block diagonal decomposition.
    num_basis : int
        Number of bases in basis reg or number of blocks in bdd.
    ntype_need_basis_reg : str
        Basis regularization applies only for
        ntype_need_basis_reg -> ntype_need_basis_reg edges types.
        Default: 'drug'.


    Notes
    -----
    1. About self_loop.
    If you need self_loop (add node embedding from previous layer)
    just make loops in your graph.

    """

    def __init__(self,
                 ntype2in_feat_dim: Dict[str, int],
                 out_feat_dim: int,
                 etype2ntypes: Dict[str, Tuple[str, str]],
                 bias: bool = True,
                 activation: Optional[Callable] = None,
                 dropout: float = 0.0,
                 regularizer: Optional[str] = None,
                 ntype_need_basis_reg: Optional[str] = None,
                 num_bases: Optional[int] = None,
                 ):
        super(RelGraphConvLayer, self).__init__()
        self.ntype2in_feat_dim = ntype2in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.etype2ntypes = etype2ntypes
        self.etypes = list(etype2ntypes.keys())
        self.bias = bias
        self.activation = activation
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.ntype_need_basis_reg = ntype_need_basis_reg
        self.etypes_need_reg = []

        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat_dim))
            nn.init.zeros_(self.h_bias)

        self.dropout = nn.Dropout(dropout)

        etype2in_feat_dim = {
            etype: self.ntype2in_feat_dim[etype2ntypes[etype][0]]
            for etype in self.etypes
        }

        # weight = False, because we initialize weights in this class
        # to can adding weights regularization
        self.conv = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(etype2in_feat_dim[etype], out_feat_dim,
                                   norm='both', weight=False, bias=False)
            for etype in self.etypes
        })

        if regularizer == 'bdd':
            self.bdds = {etype: BlockDiagDecomp((etype2in_feat_dim[etype],
                                                 out_feat_dim), num_bases)
                         for etype in self.etypes}
            self.bdds = nn.ModuleDict(self.bdds)
            self.etypes_need_reg = self.etypes
            self.etypes_without_reg = []
            return

        self.etypes_need_reg = []
        if regularizer == 'basis':
            self.etypes_need_reg = [
                etype for etype, ntypes in self.etype2ntypes.items()
                if ntypes[0] == ntype_need_basis_reg and
                   ntypes[1] == ntype_need_basis_reg]
            self.basis = WeightBasis((ntype2in_feat_dim[ntype_need_basis_reg],
                                      out_feat_dim), num_bases,
                                     len(self.etypes_need_reg))

        self.weights_without_reg = nn.ParameterDict()
        self.etypes_without_reg = list(
            set(self.etypes) - set(self.etypes_need_reg))
        for etype in self.etypes_without_reg:
            self.weights_without_reg[etype] = nn.Parameter(
                torch.Tensor(etype2in_feat_dim[etype], out_feat_dim))
            nn.init.xavier_uniform_(self.weights_without_reg[etype])

    def forward(self, g: DGLHeteroGraph, ntype2features: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        Forward computation.

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph. It can be a block.
        ntype2features : dict[str, torch.Tensor]
            Node features for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        wdict = {}
        if self.regularizer == 'bdd':
            wdict = {etype: {'weight': self.bdds[etype]()}
                     for etype in self.etypes_need_reg}
        elif self.regularizer == 'basis':
            weight = self.basis()
            wdict = {self.etypes_need_reg[i]: {'weight': w}
                     for i, w in enumerate(torch.unbind(weight, dim=0))}

        for etype in self.etypes_without_reg:
            wdict[etype] = {'weight': self.weights_without_reg[etype]}

        hs = self.conv(g, ntype2features, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return F.normalize(self.dropout(h), p=2, dim=1)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
