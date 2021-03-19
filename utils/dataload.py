import torch
import scipy.sparse as sp
from abc import ABCMeta
import os
from itertools import combinations
from typing import Tuple, Dict, NoReturn, Optional
from .adj_matrix import *
from .real_files_load import *

EType_canon = Tuple[str, str, str]


class DataLoad(metaclass=ABCMeta):
    """
    Abstract class of data loading.
    Different subclasses define specific behavior
    (e.g. synthetic data or real).

    Attributes
    ----------
    etype2adj : Dict[EType_canon, sp.csr_matrix]
        From edge type to adjacency matrix.
    etype2uniq_edges : Dict[EType_canon, Tuple[np.ndarray, np.ndarray]
        From edge type to edges.
        It should contatins only one copy of every edge!
        I.e. if edge (u, v) is in this dict, edge (v, u) can not be here.
    etype2edges : Dict[EType_canon, Tuple[np.ndarray, np.ndarray]
        From edge type to edges.
    ntype2features : Dict[str, torch.Tensor]
        From node type to matrix with nodes features.
    ntype2num_nodes : Dict[str, int]
        From node type to number of nodes.

    """

    def __init__(self):
        self.etype2adj = {}
        self.etype2edges = {}
        self.etype2uniq_edges = {}
        self.ntype2features = {}
        self.ntype2num_nodes = {}

    def _adjacency(self, adj_path: Optional[str]) -> NoReturn:
        """
        Create self.etype2adj and self.ntype2num_nodes.

        Parameters
        ----------
        adj_path : str
            path for saving/loading adjacency matrices.

        """
        raise NotImplementedError()

    def _nodes_features(self) -> NoReturn:
        """
        Create self.ntype2features.

        """
        raise NotImplementedError()

    def calc(self, adj_path: Optional[str]) -> NoReturn:
        """
        Calculate self.etype2adj, self.ntype2features,
        self.etype2uniq_edges, self.etype2edges.

        Parameters
        ----------
        adj_path : str
            path for saving/loading adjacency matrices.

        """
        if adj_path and not os.path.exists(adj_path):
            os.makedirs(adj_path)
        self._adjacency(adj_path)
        self._nodes_features()

        self.etype2edges = {etype: adj.nonzero()
                            for etype, adj in self.etype2adj.items()}

        self.etype2uniq_edges = {etype: sp.triu(adj).nonzero()
                                 for etype, adj in self.etype2adj.items()}


class SyntheticLoad(DataLoad):
    def __init__(self,
                 n_genes: int = 500,
                 n_drugs: int = 400,
                 n_drugdrug_rel_types: int = 3):
        """
        Parameters
        ----------
        n_genes : int
            Number of genes.
        n_drugs : int
            Number of drugs.
        n_drugdrug_rel_types : int
            Number of side effects.
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_drugs = n_drugs
        self.n_drugdrug_rel_types = n_drugdrug_rel_types

    def _adjacency(self, adj_path: Optional[str]) -> NoReturn:
        """
        Create self.etype2adj.

        Parameters
        ----------
        adj_path : str
            path for saving/loading adjacency matrices.

        """
        # gene -> gene
        gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)
        gene_gene_adj = nx.adjacency_matrix(gene_net)
        gene_degrees = np.array(gene_gene_adj.sum(axis=0)).squeeze()

        # gene -> drug and drug -> gene
        gene_drug_adj = sp.csr_matrix(
            (10 * np.random.randn(self.n_genes, self.n_drugs) > 15).astype(int))
        drug_gene_adj = gene_drug_adj.transpose(copy=True)

        # drug -> drug
        drug_drug_adj_list = []
        tmp = np.dot(drug_gene_adj, gene_drug_adj)
        for i in range(self.n_drugdrug_rel_types):
            mat = np.zeros((self.n_drugs, self.n_drugs))
            for d1, d2 in combinations(list(range(self.n_drugs)), 2):
                if tmp[d1, d2] == i + 4:
                    mat[d1, d2] = mat[d2, d1] = 1.
            drug_drug_adj_list.append(sp.csr_matrix(mat))
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for
                             drug_adj in drug_drug_adj_list]

        # Create self.etype2adj
        self.etype2adj = {('protein', 'association', 'protein') : gene_gene_adj,
                          ('drug', 'interaction', 'protein') : drug_gene_adj,
                          ('protein', 'interaction_by', 'drug') : gene_drug_adj}
        for se in range(len(drug_drug_adj_list)):
            self.etype2adj[('drug', 'side_effect_' + str(se), 'drug')] = \
                drug_drug_adj_list[se]

        # Create self.ntype2num_nodes
        self.ntype2num_nodes = {'drug': self.n_drugs, 'protein': self.n_genes}

    def _nodes_features(self) -> NoReturn:
        """
        Create self.ntype2features.

        Notes
        -----
        One-hot encoding for genes and drugs.

        """
        self.ntype2features = {'drug': torch.eye(self.n_drugs),
                              'protein': torch.eye(self.n_genes)}


class RealLoad(DataLoad):
    def __init__(self, combo_path: str, ppi_path: str, mono_path: str,
                 targets_path: str, min_se_freq: int, min_se_freq_mono: int,
                 drug_embed_mode: str = 'mono_se'):
        """
        Parameters
        ----------
        combo_path : str
            Path to file with table drug-drug-side.
        ppi_path : str
            Path to file with ppi.
        mono_path : str
            Path to file with side effects of drugs (individual).ntype2features
        targets_path
        min_se_freq : int
            Only se with frequency >= min_se_freq will be saved.
        min_se_freq_mono : int
            Only individual se with frequency > min_se_freq_mono will be saved.
        drug_embed_mode : str
            Type of init drug embeddings.
            'mono_se' -- individual side effects,
            'one-hot' -- one-hot encoding.
        """
        super().__init__()
        if drug_embed_mode not in ['one-hot', 'mono_se']:
            raise ValueError("drug_embed_mode should be 'one-hot' or 'mono_se'")
        self.drug_embed_mode = drug_embed_mode
        frequent_combo_path = self._leave_frequent_se(combo_path, min_se_freq)
        self.drug_drug_net, self.combo2stitch, self.combo2se, self.se2name = \
            load_combo_se(combo_path=frequent_combo_path)
        self.gene_net, self.node2idx = load_ppi(ppi_path=ppi_path)
        self.stitch2se, self.se2name_mono, se2stitch = load_mono_se(
            mono_path=mono_path)
        self.stitch2proteins = load_targets(targets_path=targets_path)

        self.ordered_list_of_drugs = list(self.drug_drug_net.nodes.keys())
        self.ordered_list_of_se = list(self.se2name.keys())
        self.ordered_list_of_proteins = list(self.gene_net.nodes.keys())

        drugs_set = set(self.ordered_list_of_drugs)
        self.ordered_list_of_se_mono = [
            se_mono for se_mono, stitch_set in se2stitch.items() if
            len(stitch_set.intersection(drugs_set)) > min_se_freq_mono]

    @staticmethod
    def _leave_frequent_se(combo_path: str, min_se_freq: int) -> str:
        """
        Create pre-processed file that only has frequent side effects.

        Parameters
        ----------
        min_se_freq : int
            Only se with frequency >= min_se_freq will be saved.

        Returns
        -------
        str
            Path to combo data considering only frequent se.
        """
        all_combo_df = pd.read_csv(combo_path)
        se_freqs = all_combo_df["Polypharmacy Side Effect"].value_counts()
        frequent_se = se_freqs[se_freqs >= min_se_freq].index.tolist()
        frequent_combo_df = all_combo_df[
            all_combo_df["Polypharmacy Side Effect"].isin(frequent_se)]

        filename, file_extension = os.path.splitext(combo_path)
        frequent_combo_path = filename + '-freq-only' + file_extension
        frequent_combo_df.to_csv(frequent_combo_path, index=False)
        return frequent_combo_path

    def _adjacency(self, adj_path: str) -> NoReturn:
        """
        Create self.etype2adj.

        Parameters
        ----------
        adj_path : str
            path for saving/loading adjacency matrices.

        """
        # gene -> gene
        gene_gene_adj = nx.adjacency_matrix(self.gene_net)
        # Number of connections for each gene
        gene_degrees = np.array(gene_gene_adj.sum(axis=0)).squeeze()

        # drug -> gene and gene -> drug
        drug_gene_adj = create_adj_matrix(
            a_item2b_item=self.stitch2proteins,
            ordered_list_a_item=self.ordered_list_of_drugs,
            ordered_list_b_item=self.ordered_list_of_proteins)
        gene_drug_adj = drug_gene_adj.transpose(copy=True)

        # drug -> drug
        num_se = len(self.ordered_list_of_se)
        drug_drug_adj_list = []
        try:
            print("Try to load drug-drug adjacency matrices from file.")
            if len(os.listdir(adj_path)) < num_se:
                raise IOError('Not all drug-drug adjacency matrices are saved')
            for i in range(num_se):
                drug_drug_adj_list.append(sp.load_npz(
                    adj_path + '/sparse_matrix%04d.npz' % i).tocsr())
        except IOError:
            print('Calculate drug-drug adjacency matrices')
            drug_drug_adj_list = create_combo_adj(
                combo_a_item2b_item=self.combo2se,
                combo_a_item2a_item=self.combo2stitch,
                ordered_list_a_item=self.ordered_list_of_drugs,
                ordered_list_b_item=self.ordered_list_of_se)
            print("Saving matrices to file")
            for i in range(len(drug_drug_adj_list)):
                sp.save_npz(f'{adj_path}/sparse_matrix%04d.npz' % (i,),
                            drug_drug_adj_list[i].tocoo())

        # Create self.etype2adj
        self.etype2adj = {
            ('protein', 'association', 'protein'): gene_gene_adj,
            ('drug', 'interaction', 'protein'): drug_gene_adj,
            ('protein', 'interaction_by', 'drug'): gene_drug_adj}
        for se in range(len(drug_drug_adj_list)):
            self.etype2adj[('drug', 'side_effect_' + str(se), 'drug')] = \
                drug_drug_adj_list[se]

        # Create self.ntype2num_nodes
        self.ntype2num_nodes = {'drug': len(self.ordered_list_of_drugs),
                                'protein': len(self.ordered_list_of_proteins)}

    def _nodes_features(self) -> NoReturn:
        """
        Create self.ntype2features.

        Notes
        -----
        One-hot encoding as genes features.
        Binary vectors with presence of different side effects
        as drugs features.

        """
        # One-hot for genes
        n_genes = self.ntype2num_nodes['protein']
        gene_feat = torch.eye(n_genes, dtype=torch.float32)

        # features for drugs
        if self.drug_embed_mode == 'one-hot':
            n_drugs = self.ntype2num_nodes['drug']
            drug_feat = torch.eye(n_drugs, dtype=torch.float32)
            self.ntype2features = {'drug': drug_feat, 'protein': gene_feat}
            return

        # Create sparse matrix with rows -- genes features.
        # Gene feature -- binary vector with length = num of mono se.
        # feature[i] = 1 <=> gene has ith mono se
        drug_feat = create_adj_matrix(
            a_item2b_item=self.stitch2se,
            ordered_list_a_item=self.ordered_list_of_drugs,
            ordered_list_b_item=self.ordered_list_of_se_mono)
        # Check if some gene has zero embedding (i.e. it has no frequent se)
        drugs_zero_features = np.array(
            self.ordered_list_of_drugs)[drug_feat.getnnz(axis=1) == 0]
        # assert 0 not in drug_feat.getnnz(axis=1), \
        # 'All genes should have nonzero embeddings! '
        print(f'Length of drugs features vectors: {drug_feat.shape[1]}')
        print(f'Number of unique vectors: '
              f'{np.unique(drug_feat.toarray(), axis=0).shape[0]}')
        if len(drugs_zero_features) > 0:
            print('Warning! All genes should have nonzero embeddings! ')
            print(f'Where are {len(drugs_zero_features)} zero embeddings')
            print(f'Bad drugs: {drugs_zero_features}')

        self.ntype2features = {
            'drug': torch.from_numpy(drug_feat.todense()).to(
                dtype=torch.float32),
            'protein': gene_feat}
