import numpy as np
import dgl
import torch
from typing import Tuple, Union
from collections import deque
import time

from src.model import Model
from src.split import DataSplitter
from utils.dataload import SyntheticLoad, RealLoad
from utils.loss_metrics import compute_loss
from utils.train_utils import optimizer_to, create_parser, create_folders
from constants import PARAMS, INPUT_FILE_PATH, MODEL_SAVE_PATH, SPLIT_SAVE_PATH


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.neg_sample_size > 1 and args.loss_type == 'hinge':
        raise ValueError('Hinge loss should be only with neg sample size 1!')

    # for correct parameters logging with neptune
    dict_args = vars(args)
    for key in PARAMS.keys():
        PARAMS[key] = dict_args[key]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Use device: {device}')

    create_folders([MODEL_SAVE_PATH, SPLIT_SAVE_PATH])

    if args.log:
        import neptune

        neptune.init('Pollutants/sandbox')
        neptune_experiment_name = 'Real data' if args.real else 'Toy data'
        neptune.create_experiment(name=neptune_experiment_name,
                                  params=PARAMS,
                                  upload_stdout=True,
                                  upload_stderr=True,
                                  send_hardware_metrics=True,
                                  upload_source_files='**/*.py')
        neptune.append_tag('pytorch')

        if args.gpu:
            neptune.append_tag('gpu')
        if args.real:
            neptune.append_tag('real data')
        else:
            neptune.append_tag('synthetic data')

    ############################################################################
    ### Data loading, split edges into train/val/test
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
    reverse_etypes = {etype: etype for etype in etype2edges.keys()
                      if etype not in [('drug', 'interaction', 'protein'),
                                       ('protein', 'interaction_by', 'drug')]}
    reverse_etypes[('drug', 'interaction', 'protein')] = \
        ('protein', 'interaction_by', 'drug')
    splitter = DataSplitter(etype2uniq_edges, reverse_etypes,
                            split_ratio=args.split_ratio)

    ############################################################################
    ### Create graphs
    ############################################################################

    # Graph with all edges.
    graph = dgl.heterograph(etype2edges, num_nodes_dict=ntype2num_nodes)
    graph.nodes['protein'].data['feature'] = ntype2features['protein']
    graph.nodes['drug'].data['feature'] = ntype2features['drug']

    etype_idx2train_edges_ids = {}
    etype_idx2test_edges_ids = {}
    etype_idx2val_edges_ids = {}

    for etype in graph.canonical_etypes:
        etype_idx2train_edges_ids[etype] = graph.edge_ids(
            u=splitter.etype2train_edges[etype][0],
            v=splitter.etype2train_edges[etype][1],
            etype=etype)
        etype_idx2val_edges_ids[etype] = graph.edge_ids(
            u=splitter.etype2val_edges[etype][0],
            v=splitter.etype2val_edges[etype][1],
            etype=etype)
        etype_idx2test_edges_ids[etype] = graph.edge_ids(
            u=splitter.etype2test_edges[etype][0],
            v=splitter.etype2test_edges[etype][1],
            etype=etype)

    np.save(SPLIT_SAVE_PATH + '/etype2train_edges.npy',
            splitter.etype2train_edges)
    np.save(SPLIT_SAVE_PATH + '/etype2test_edges.npy',
            splitter.etype2test_edges)
    np.save(SPLIT_SAVE_PATH + '/etype2val_edges.npy',
            splitter.etype2val_edges)

    # We need a train graph, to sample edges from it.
    train_graph = graph.edge_subgraph(etype_idx2train_edges_ids,
                                      preserve_nodes=True)
    # Also we need test and val graphs to calculate val and test metrics
    val_graph = graph.edge_subgraph(etype_idx2val_edges_ids,
                                    preserve_nodes=True)
    test_graph = graph.edge_subgraph(etype_idx2test_edges_ids,
                                     preserve_nodes=True)
    # Add self loop in train graph to adding information
    # from node when updating its embedding
    for edge_type in (set(train_graph.etypes) -
                      {'interaction', 'interaction_by'}):
        train_graph = dgl.add_self_loop(train_graph, edge_type)

    ############################################################################
    ### Create minibatch loaders
    ############################################################################
    if args.load:
        # Temp part only for continue failed experiment
        np.random.seed(args.seed_load)
        torch.manual_seed(args.seed_load)

    negative_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.neg_sample_size)

    train_loader = dgl.dataloading.EdgeDataLoader(
        graph, etype_idx2train_edges_ids,
        block_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2),
        g_sampling=train_graph,
        negative_sampler=negative_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

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

    last_epoch = 0
    last_iter = 0

    if args.load:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        last_iter = checkpoint['step']
        loss = checkpoint['train_loss']
        del checkpoint
        torch.cuda.empty_cache()
    model = model.to(device)
    optimizer_to(opt, device)

    epochs_val_loss_deq = deque(maxlen=args.early_stopping_window + 1)

    ############################################################################
    ### Train
    ############################################################################
    print('Train')

    for epoch in range(last_epoch, args.epoch):
        val_loss = None
        if args.load and epoch == last_epoch:
            cur_iter = last_iter + 1
        else:
            cur_iter = 0
        for input_nodes, positive_graph, negative_graph, blocks in train_loader:
            t = time.time()
            model.train()
            train_loss = compute_loss(positive_graph=positive_graph,
                                      negative_graph=negative_graph,
                                      device=device,
                                      model=model,
                                      blocks=blocks,
                                      loss_type=args.loss_type,
                                      calc_metrics=False,
                                      pos_weight=args.neg_sample_size,
                                      max_margin=args.max_margin,
                                      reduction=args.reduction)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            last_iter = cur_iter == len(train_loader) - 1
            if not cur_iter % args.print_progress_every or last_iter:
                model.eval()
                with torch.no_grad():
                    positive_graph = val_graph
                    negative_graph = dgl.heterograph(
                        negative_sampler(graph, etype_idx2val_edges_ids),
                        num_nodes_dict=ntype2num_nodes)
                    val_metrics = compute_loss(positive_graph=positive_graph,
                                               negative_graph=negative_graph,
                                               device=device,
                                               model=model,
                                               train_graph=train_graph,
                                               loss_type=args.loss_type,
                                               calc_metrics=True,
                                               pos_weight=args.neg_sample_size,
                                               max_margin=args.max_margin,
                                               reduction=args.reduction)
                    val_roc, val_auprc, val_apk = tuple(val_metrics[i].mean()
                                                        for i in range(1, 4))
                    val_loss = val_metrics[0].item()

                print("Epoch:", "%04d" % (epoch + 1), "Iter:",
                      "%04d" % (cur_iter + 1),
                      "train_loss=", "{:.5f}".format(train_loss.item()),
                      "val_loss=", "{:.5f}".format(val_loss),
                      "val_roc=", "{:.5f}".format(val_roc),
                      "val_auprc=", "{:.5f}".format(val_auprc),
                      "val_apk=", "{:.5f}".format(val_apk),
                      "time=", "{:.5f}".format(time.time() - t))

                if args.log:
                    neptune.log_metric("train_loss", train_loss,
                                       timestamp=time.time())
                    neptune.log_metric("val_roc", val_roc,
                                       timestamp=time.time())
                    neptune.log_metric("val_apk", val_apk,
                                       timestamp=time.time())
                    neptune.log_metric("val_auprc", val_auprc,
                                       timestamp=time.time())
                    neptune.log_metric("val_loss", val_loss,
                                       timestamp=time.time())
            if not cur_iter % args.save_every:
                model.eval()
                path = (MODEL_SAVE_PATH + '/' + str(epoch) + '_'
                        + str(cur_iter) + '.pt')
                torch.save({
                    'epoch': epoch,
                    'step': cur_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': train_loss,
                }, path)

            cur_iter += 1
        epochs_val_loss_deq.append(val_loss)

        # Early stopping
        is_increasing = np.all(np.diff(epochs_val_loss_deq) >= 0)
        if len(epochs_val_loss_deq) == epochs_val_loss_deq.maxlen \
                and is_increasing:
            print(f'Early stopping! {len(epochs_val_loss_deq)} epochs losses' +
                  f' are increasing. Its values: {list(epochs_val_loss_deq)}')
            print('Model stops training.')

            break

    ############################################################################
    ### Test
    ############################################################################
    print()
    print('Test')
    model.eval()
    with torch.no_grad():
        positive_graph = test_graph
        negative_graph = dgl.heterograph(
            negative_sampler(graph, etype_idx2test_edges_ids),
            num_nodes_dict=ntype2num_nodes)
        test_metrics = compute_loss(positive_graph=positive_graph,
                                    negative_graph=negative_graph,
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

        if args.log:
            neptune.log_metric("ROC-AUC", test_metrics[1][i])
            neptune.log_metric("AUPRC", test_metrics[2][i])
            neptune.log_metric("AP@k score", test_metrics[3][i])
    if args.log:
        neptune.stop()
