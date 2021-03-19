import dgl
from typing import Dict, Union, Tuple, NoReturn, List
import scipy.sparse as sp
import numpy as np

EType_canon = Tuple[str, str, str]


class DataSplitter:
    """
    Class for splitting edges into train/test/val

    Parameters
    ----------
    etype2uniq_edges : Dict[EType_canon, Tuple[np.ndarray, np.ndarray]]
        From edge type to edges.
        It should contatins only one copy of every edge!
        I.e. if edge (u, v) is in this dict, edge (v, u) can not be here.
    reverse_etypes : Dict[EType_canon, EType_canon]
        For every edge type its reversed type.
        E.g.
        ('drug', 'interaction', 'protein') ->
            ('protein', 'interaction_by', 'drug')
        ('drug', 'side_effect_0', 'drug') ->
            ('drug', 'side_effect_0', 'drug')
    split_ratio : float
        Ratio for train/val/test split.
    min_split_size : int
        Min size of val and test sets.

    """
    def __init__(self,
                 etype2uniq_edges: Dict[EType_canon,
                                        Tuple[np.ndarray, np.ndarray]],
                 reverse_etypes: Dict[EType_canon, EType_canon],
                 split_ratio: float = 0.1,
                 min_split_size: int = 50
                 ):
        self.etype2uniq_edges = etype2uniq_edges
        self.reverse_etypes = reverse_etypes
        self.split_ratio = split_ratio
        self.min_split_size = min_split_size

        self.etype2train_edges = {}
        self.etype2test_edges = {}
        self.etype2val_edges = {}

        for etype, reverse_etype in self.reverse_etypes.items():
            self._split(etype, reverse_etype)

    @staticmethod
    def _reverse_edges(edges: np.ndarray) -> np.ndarray:
        """
        Inverse all given edges.

        Parameters
        ----------
        edges : np.array[Tuple[int, int]]
            Edges to reverse.

        Returns
        -------
        np.array[Tuple[int, int]]
            Reversed edges (e.g. edge [1, 2] -> [2, 1]).

        """
        reversed_edges = edges.copy()
        reversed_edges[:, [0, 1]] = reversed_edges[:, [1, 0]]
        return reversed_edges

    def _split(self,
               etype: EType_canon,
               reverse_etype: EType_canon
               ) -> NoReturn:
        """
        Split edges of one type.

        Parameters
        ----------
        etype : EType_canon
            Edge type for splitting.
        reverse_etype
            Corresponding reverse edge type.
            If it differs from etype,
            edges for reverse_etype are reversed edges for etype.
            (E.g. if edge (u, v) is in train set for etype,
            edge (v, u) is in train set for reverse_etype).

        Returns
        -------

        """
        edges = np.array(list(zip(*self.etype2uniq_edges[etype])))
        min_split_size = self.min_split_size
        if reverse_etype == etype:
            min_split_size = min_split_size // 2
        num_test = max(min_split_size,
                       int(np.floor(edges.shape[0] * self.split_ratio)))
        num_val = num_test

        edges_idx = list(range(edges.shape[0]))
        np.random.shuffle(edges)

        val_edges_idx = edges_idx[:num_val]
        val_edges = edges[val_edges_idx]

        test_edges_idx = edges_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edges_idx]

        train_edges = np.delete(edges,
                                np.hstack([test_edges_idx, val_edges_idx]),
                                axis=0)

        reversed_train_edges = self._reverse_edges(train_edges)
        reversed_test_edges = self._reverse_edges(test_edges)
        reversed_val_edges = self._reverse_edges(val_edges)

        if reverse_etype == etype:
            train_edges = np.vstack((train_edges, reversed_train_edges))
            test_edges = np.vstack((test_edges, reversed_test_edges))
            val_edges = np.vstack((val_edges, reversed_val_edges))

        self.etype2train_edges[etype] = (train_edges[:, 0],
                                         train_edges[:, 1])
        self.etype2val_edges[etype] = (val_edges[:, 0],
                                       val_edges[:, 1])
        self.etype2test_edges[etype] = (test_edges[:, 0],
                                        test_edges[:, 1])
        if reverse_etype != etype:
            self.etype2train_edges[reverse_etype] = (reversed_train_edges[:, 0],
                                                     reversed_train_edges[:, 1])
            self.etype2val_edges[reverse_etype] = (reversed_val_edges[:, 0],
                                                   reversed_val_edges[:, 1])
            self.etype2test_edges[reverse_etype] = (reversed_test_edges[:, 0],
                                                    reversed_test_edges[:, 1])
