import dgl
import torch
import torch.nn as nn
from typing import List, Dict
from dgl.heterograph import DGLHeteroGraph


class HeteroMLPPredictor(nn.Module):
    def __init__(self, embed_dim: int, rel_list: List[str]) -> None:
        """
        Decoder from original decagon article.

        Parameters
        ----------
        embed_dim : int
            Size of embedding
        rel_list : List[str]
            List of all types of nodes. For pairwise side effect edge type
            name should starts with "side".

        """
        super().__init__()
        self.matrixes = torch.nn.ModuleDict({
            'R': nn.Linear(embed_dim, embed_dim, False)})
        self.params = nn.ParameterDict()
        for rel in rel_list:
            if rel[:4] == 'side':
                params = torch.Tensor(embed_dim, 1)
                nn.init.xavier_uniform_(params)
                self.params[rel] = nn.Parameter(params.squeeze())
            else:
                self.matrixes[rel] = nn.Linear(embed_dim, embed_dim)
                nn.init.xavier_uniform_(self.matrixes[rel].weight)

    def apply_edges(self, edges: dgl.udf.EdgeBatch,
                    edge_type: str) -> Dict[str, torch.Tensor]:
        """
        Apply transformation to all edges, suppose it have type edge_type.

        Parameters
        ----------
        edges : dgl.udf.EdgeBatch
            Edges.
        edge_type : str
            Edge type name.

        Returns
        -------
        score : Dict[str, torch.Tensor]
            Score for each edge.
        """
        h_u = edges.src['h']
        h_v = edges.dst['h']
        # TODO: fix it
        if edge_type[:4] == 'side':
            lft = h_u.mul(self.params[edge_type])
            lft = self.matrixes['R'](lft)
            lft = lft.mul(self.params[edge_type])
        else:
            lft = self.matrixes[edge_type](h_u)
        y = lft.mul(h_v)
        score = torch.sum(y, dim=1)
        return {'score': score}

    def forward(self,
                subgraph: DGLHeteroGraph,
                node2features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        subgraph : DGLHeteroGraph
            Subgraph with only edges from minibatch.
        node2features : Dict[str, torch.Tensor]
            From node type to node features (computed from RGCN).
        etype : str
            Edge type.

        Returns
        -------
        score : torch.Tensor
            Score for every edge in minibatch.

        """
        with subgraph.local_scope():
            # assigns 'h' of all node types in one shot
            subgraph.ndata['h'] = node2features
            for etype in subgraph.etypes:
                if not subgraph.num_edges(etype):
                    continue
                subgraph.apply_edges(
                    lambda edge: self.apply_edges(edge, etype), etype=etype)
            return subgraph.edata['score']
