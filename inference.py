import numpy as np
import dgl
import torch

from src.model import Model
from utils.dataload import SyntheticLoad, RealLoad
from utils.loss_metrics import compute_loss
from utils.train_utils import optimizer_to, create_parser
from utils.negative import not_exist_edges
from constants import INPUT_FILE_PATH, SPLIT_SAVE_PATH


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.neg_sample_size > 1 and args.loss_type == 'hinge':
        raise ValueError('Hinge loss should be only with neg sample size 1!')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Use device: {device}')


    ############################################################################
    ### Data loading
    ############################################################################
    if args.real:
        data = RealLoad(
            combo_path=f'{INPUT_FILE_PATH}/bio-decagon-combo.csv',
            ppi_path=f'{INPUT_FILE_PATH}/bio-decagon-ppi.csv',
            mono_path=f'{INPUT_FILE_PATH}/bio-decagon-mono.csv',
            targets_path=f'{INPUT_FILE_PATH}/bio-decagon-targets-all.csv',
            min_se_freq=500, min_se_freq_mono=40,
            drug_embed_mode=args.drug_embed_mode)
    else:
        data = SyntheticLoad()
    data.calc(adj_path='data/adj/real')
    etype2uniq_edges = data.etype2uniq_edges
    etype2edges = data.etype2edges
    ntype2features = data.ntype2features
    ntype2num_nodes = data.ntype2num_nodes
    num_se = len(etype2edges) - 3
    ntype2in_feat_dim = {ntype: features.shape[1]
                         for ntype, features in ntype2features.items()}

    etype2train_edges = np.load(SPLIT_SAVE_PATH + '/etype2train_edges.npy',
                                allow_pickle=True).item()
    etype2test_edges = np.load(SPLIT_SAVE_PATH + '/etype2test_edges.npy',
                               allow_pickle=True).item()

    ############################################################################
    ### Create graphs
    ############################################################################

    # Graph with all edges.
    graph = dgl.heterograph(etype2edges, num_nodes_dict=ntype2num_nodes)
    graph.nodes['protein'].data['feature'] = ntype2features['protein']
    graph.nodes['drug'].data['feature'] = ntype2features['drug']

    # We need to know indexes of train and test edges in full graph
    etype_idx2train_edges_ids = {}
    etype_idx2test_edges_ids = {}
    for etype in graph.canonical_etypes:
        etype_idx2train_edges_ids[etype] = graph.edge_ids(
            u=etype2train_edges[etype][0],
            v=etype2train_edges[etype][1],
            etype=etype)
        etype_idx2test_edges_ids[etype] = graph.edge_ids(
            u=etype2test_edges[etype][0],
            v=etype2test_edges[etype][1],
            etype=etype)

    # Create train and test subgraph
    train_graph = graph.edge_subgraph(etype_idx2train_edges_ids,
                                      preserve_nodes=True)
    test_graph = graph.edge_subgraph(etype_idx2test_edges_ids,
                                     preserve_nodes=True)
    etype2neg_test_edges = {
        etype: not_exist_edges(sparse=data.etype2adj[etype],
                               n_of_samples=args.neg_sample_size * \
                                            test_graph.num_edges(etype))
        for etype in test_graph.canonical_etypes}
    neg_test_graph = dgl.heterograph(etype2neg_test_edges,
                                     num_nodes_dict=ntype2num_nodes)

    # Add self loop in train graph to adding information
    # from node when updating its embedding
    for edge_type in (set(train_graph.etypes) -
                      {'interaction', 'interaction_by'}):
        train_graph = dgl.add_self_loop(train_graph, edge_type)



    ############################################################################
    ### Create model and optimizer
    ############################################################################

    model = Model(ntype2in_feat_dim=ntype2in_feat_dim,
                  hidden_dim=args.hidden_dim,
                  embed_dim=args.embed_dim,
                  etypes_canon = train_graph.canonical_etypes,
                  bias=args.bias,
                  dropout=args.dropout,
                  regularizer=args.regularizer,
                  ntype_need_basis_reg='drug',
                  num_bases_1=args.num_bases_1,
                  num_bases_2=args.num_bases_2)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint
    torch.cuda.empty_cache()
    model = model.to(device)
    optimizer_to(opt, device)

    ############################################################################
    ### Test
    ############################################################################
    print()
    print('Test')
    model.eval()
    with torch.no_grad():
        test_metrics = compute_loss(positive_graph=test_graph,
                                    negative_graph=neg_test_graph,
                                    device=device,
                                    model=model,
                                    train_graph=train_graph,
                                    loss_type=args.loss_type,
                                    calc_metrics=True,
                                    pos_weight=args.neg_sample_size,
                                    max_margin=args.max_margin,
                                    reduction=args.reduction)
    print('Mean metrics')

    print("test_loss=", "{:.5f}".format(test_metrics[0].item()),
          "test_roc=", "{:.5f}".format(test_metrics[1].mean()),
          "test_auprc=", "{:.5f}".format(test_metrics[2].mean()),
          "test_apk=", "{:.5f}".format(test_metrics[3].mean()))

    print('Metrics for different edge types')
    for i, etype in enumerate(graph.canonical_etypes):
        print("Edge type=", "{:38.38}".format(str(etype)),
              "test_roc=", "{:.5f}".format(test_metrics[1][i]),
              "test_auprc=", "{:.5f}".format(test_metrics[2][i]),
              "test_apk=", "{:.5f}".format(test_metrics[3][i]))
