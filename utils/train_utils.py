import torch
import argparse
from constants import PARAMS
import os
from typing import List


# Code from https://github.com/pytorch/pytorch/issues/8741
# for moving optimizer to device.
# We use it to avoid gpu memory problems
# while loading model and optimizer from checkpoint.
# (Firstly, load to cpu, secondly move to cuda)
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', default=False,
                        action='store_true',
                        help='Run on real data or toy example, default False')
    parser.add_argument('--log', default=False,
                        action='store_true',
                        help='Whether to log run or not, default False')
    parser.add_argument('--gpu', default=False,
                        action='store_true', help="Use gpu or not.")
    parser.add_argument('--batch_size', default=PARAMS['batch_size'], type=int,
                        help='Batch size')
    parser.add_argument('--hidden_dim', default=PARAMS['hidden_dim'], type=int,
                        help="Output embedding size after hidden layer")
    parser.add_argument('--embed_dim', default=PARAMS['embed_dim'], type=int,
                        help="Output embedding size after last layer")
    parser.add_argument('--epoch', default=PARAMS['epoch'], type=int,
                        help="Number of epochs")
    parser.add_argument('--learning_rate', default=PARAMS['learning_rate'],
                        type=float, help="Learning rate for optimizer")
    parser.add_argument('--split_ratio', default=PARAMS['split_ratio'],
                        type=float, help="Ratio for train/val/test split")
    parser.add_argument('--dropout', default=PARAMS['dropout'],
                        type=float, help="Dropout rate")
    parser.add_argument('--bias', default=PARAMS['bias'],
                        type=bool, help="Add bias in model or not")
    parser.add_argument('--neg_sample_size', default=PARAMS['neg_sample_size'],
                        type=int,
                        help="How many negative edges for every real." +
                             "Can be > 1 only if loss is cross entropy")
    parser.add_argument('--loss_type', default=PARAMS['loss_type'], type=str,
                        help="Loss function ('cross_entropy' or 'hinge').")
    parser.add_argument('--max_margin', default=PARAMS['max_margin'],
                        type=float, help="Margin parameter for hinge loss")
    parser.add_argument('--reduction', default=PARAMS['reduction'], type=str,
                        help="Reduction for loss ('mean' or 'sum')")
    parser.add_argument('--regularizer', default=PARAMS['regularizer'],
                        type=str,
                        help="Type of regularizer" +
                             "(not given = no regularization, 'bdd' or 'basis'")
    parser.add_argument('--num_bases_1', default=PARAMS['num_bases_1'],
                        type=int,
                        help="Parameter for regularization (first layer)" +
                             "Number of basis matrixes for basis and " +
                             "number of diagonal blocks in bdd.")
    parser.add_argument('--num_bases_2', default=PARAMS['num_bases_2'],
                        type=int,
                        help="Parameter for regularization (second layer)" +
                             "Number of basis matrixes for basis and " +
                             "number of diagonal blocks in bdd.")
    parser.add_argument('--print_progress_every', default=150, type=int,
                        help="Frequency (in iterations) of printing progress")
    parser.add_argument('--seed', default=PARAMS['seed'],
                        type=int, help="Random seed")
    parser.add_argument('--seed_load', default=PARAMS['seed_load'], type=int,
                        help="Random seed for train after loading from file")
    parser.add_argument('--num_workers', default=4,
                        type=int, help="Number of workers for EdgeDataLoader")
    parser.add_argument('--save_every', default=1024, type=int,
                        help="Frequency (in iterations) of saving " +
                             " model weights  and optimizer state")
    parser.add_argument('--load', default=False,
                        action='store_true',
                        help='Load model checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--early_stopping_window', type=int,
                        default=PARAMS['early_stopping_window'],
                        help='Window size for early stopping')
    parser.add_argument('--drug_embed_mode', type=str,
                        default=PARAMS['drug_embed_mode'],
                        help="Type of init drug embeddings" +
                             "'mono_se' -- individual side effects, " +
                             "'one-hot' -- one-hot encoding.")
    return parser


def create_folders(folders: List[str]):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
