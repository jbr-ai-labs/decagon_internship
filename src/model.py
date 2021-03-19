import torch
import torch.nn as nn
from .rgcn import RelGraphConvLayer
from .decoders import HeteroMLPPredictor
from typing import Dict, Tuple, Union
from dgl.heterograph import DGLBlock, DGLHeteroGraph
from typing import List, Optional

EType_canon = Tuple[str, str, str]

class Model(nn.Module):
    """
    Implementation of Decagon model.

    Parameters
    ----------
    ntype2in_feat_dim : Dict[str, int]
        From node type to corresponding input feature size.
    hidden_dim : int
        Output feature size for hidden layer.
    embed_dim : int
        Output feature size after all two layers.
    bias : bool
        True if bias is added in rgcn layers. Default: True.
    dropout : float, optional
        Dropout rate. Default: 0.0
    regularizer : str
        Type of weight regularization. Default: None
        None = no regularization,
        'basis' = basis decomposition,
        'bdd' = block diagonal decomposition.
    num_basis_1 : int
        Number of bases in basis reg or number of blocks in bdd
        (first rgcn layer). Default: None.
    num_basis_2 : int
        Number of bases in basis reg or number of blocks in bdd
        (second rgcn layer). Default: None.
    ntype_need_basis_reg : str
        Basis regularization applies only for
        ntype_need_basis_reg -> ntype_need_basis_reg edges types.
        Default: 'drug'.

    """
    def __init__(self,
                 ntype2in_feat_dim: Dict[str, int],
                 hidden_dim: int,
                 embed_dim: int,
                 etypes_canon: List[EType_canon],
                 bias: bool = True,
                 dropout: float = 0.0,
                 regularizer: Optional[str] = None,
                 ntype_need_basis_reg: Optional[str] = 'drug',
                 num_bases_1: Optional[int] = None,
                 num_bases_2: Optional[int] = None
                 ):
        super().__init__()
        ntypes = ntype2in_feat_dim.keys()
        etype2ntypes = {etype[1]: (etype[0], etype[2])
                        for etype in etypes_canon}

        self.hidden1 = RelGraphConvLayer(
            ntype2in_feat_dim=ntype2in_feat_dim,
            out_feat_dim=hidden_dim,
            etype2ntypes=etype2ntypes,
            bias=bias,
            dropout=dropout,
            activation=torch.nn.ReLU(),
            regularizer=regularizer,
            ntype_need_basis_reg=ntype_need_basis_reg,
            num_bases=num_bases_1,
        )

        self.hidden2 = RelGraphConvLayer(
            ntype2in_feat_dim={node: hidden_dim for node in ntypes},
            out_feat_dim=embed_dim,
            etype2ntypes=etype2ntypes,
            bias=bias,
            dropout=dropout,
            activation=None,
            regularizer=regularizer,
            ntype_need_basis_reg=ntype_need_basis_reg,
            num_bases=num_bases_2,
        )
        self.pred = HeteroMLPPredictor(embed_dim, list(etype2ntypes.keys()))

    def forward(self,
                g: Union[DGLHeteroGraph, List[DGLBlock]],
                ntype2features: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        Create embeddings after two rgcn layers.

        Parameters
        ----------
        g : Union[DGLHeteroGraph, List[DGLBlock]]
            Input graph. It can be a block. 
        ntype2features : Dict[str, torch.Tensor]
            Node features for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.

        """
        if isinstance(g, DGLHeteroGraph):
            h = self.hidden1(g, ntype2features)
            h = self.hidden2(g, h)
        else:
            blocks = g
            h = self.hidden1(blocks[0], ntype2features)
            h = self.hidden2(blocks[1], h)
        return h

    def calc_score(self,
                   subgraph: DGLHeteroGraph,
                   node2features: Dict[str, torch.Tensor]):
        return self.pred(subgraph, node2features)

