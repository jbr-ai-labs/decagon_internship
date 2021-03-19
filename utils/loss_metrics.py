import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Union


EType_canon = Tuple[str, str, str]


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a, p in zip(actual, predicted)])



def cross_entropy(pos_score: Dict[EType_canon, torch.Tensor],
                  neg_score: Dict[EType_canon, torch.Tensor],
                  device: torch.device,
                  reduction: str = 'sum',
                  pos_weight: float = 1.
                  ) -> torch.Tensor():
    """
    Computes cross entropy loss.

    Parameters
    ----------
    pos_score : Dict[EType_canon, torch.Tensor]
        From etype to tensor with scores of real edges.
    neg_score
        From etype to tensor with scores of fake edges.
    device : torch.device
        Device on which we compute loss.
    reduction : str
        Type of loss reduction across minibatch ('mean' or 'sum').
    pos_weight: float
        Weight of positive examples.
        E.g. if for every positive example we have 3 negative examples,
        we should take pos_weight = 3.

    Returns
    -------
    torch.Tensor
        Loss.

    """
    predicted_pos = torch.cat(list(pos_score.values()))
    predicted_neg = torch.cat(list(neg_score.values()))
    predicted = torch.cat([predicted_pos, predicted_neg])

    true_pos = torch.ones(len(predicted_pos)).to(device)
    true_neg = torch.zeros(len(predicted_neg)).to(device)
    true = torch.cat([true_pos, true_neg])

    pos_weight = torch.Tensor([pos_weight]).to(device)
    loss = F.binary_cross_entropy_with_logits(predicted, true,
                                              reduction=reduction,
                                              pos_weight=pos_weight
                                              ) / pos_weight
    return loss

def hinge(pos_score: Dict[EType_canon, torch.Tensor],
          neg_score: Dict[EType_canon, torch.Tensor],
          device: torch.device,
          reduction: str = 'sum',
          max_margin: float = 1e-1
          ) -> torch.Tensor():
    """
    Computes hinge loss.

    Parameters
    ----------
    pos_score : Dict[EType_canon, torch.Tensor]
        From etype to tensor with scores of real edges.
    neg_score
        From etype to tensor with scores of fake edges.
    device : torch.device
        Device on which we compute loss.
    reduction : str
        Type of loss reduction across minibatch ('mean' or 'sum').
    max_margin : float
        Parameter for margin_ranking_loss.

    Returns
    -------
    torch.Tensor
        Loss.

    """
    predicted_pos = torch.cat(list(pos_score.values()))
    predicted_neg = torch.cat(list(neg_score.values()))
    # predicted_pos should be higher than predicted_neg, so target = [1, ..., 1]
    target = torch.ones(len(predicted_pos)).to(device)
    loss = F.margin_ranking_loss(predicted_pos, predicted_neg, target,
                                 reduction=reduction, margin=max_margin)
    return loss


def get_accuracy_scores(pos_score: Dict[EType_canon, torch.Tensor],
                        neg_score: Dict[EType_canon, torch.Tensor]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    pos_score : Dict[EType_canon, torch.Tensor]
        From etype to tensor with scores of real edges.
    neg_score
        From etype to tensor with scores of fake edges.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays with roc, aupr and apk for every edge type.

    """
    rocs = np.zeros(shape=(len(pos_score.keys()),))
    auprs = np.zeros(shape=(len(pos_score.keys()),))
    apks = np.zeros(shape=(len(pos_score.keys()),))

    for i, etype in enumerate(pos_score.keys()):
        predicted = torch.sigmoid(torch.cat([pos_score[etype],
                                             neg_score[etype]])).cpu().numpy()
        true = np.hstack([np.ones(len(pos_score[etype])),
                          np.zeros(len(neg_score[etype]))])

        rocs[i] = roc_auc_score(true, predicted)
        auprs[i] = average_precision_score(true, predicted)

        # Real existing edges (local indexes)
        actual_idx = range(len(pos_score[etype]))
        # All local indexes with probability (sorted)
        predicted_idx_all = sorted(range(len(predicted)), reverse=True,
                                   key=lambda i: predicted[i])
        apks[i] = apk(actual_idx, predicted_idx_all, k=50)
    return rocs, auprs, apks


def compute_loss(positive_graph,
                 negative_graph,
                 device,
                 model,
                 blocks=None,
                 train_graph=None,
                 loss_type='cross_entropy',
                 calc_metrics: bool = False,
                 reduction: str = 'sum',
                 pos_weight: float = 1.,
                 max_margin: float = 1e-1,
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray,
                                                np.ndarray, np.ndarray]]:
    positive_graph = positive_graph.to(device)
    negative_graph = negative_graph.to(device)
    if blocks is not None:
        blocks = [block.to(device) for block in blocks]
        input_features = blocks[0].srcdata['feature']
        node2embed = model(blocks, input_features)
    else:
        train_graph = train_graph.to(device)
        input_features = train_graph.srcdata['feature']
        node2embed = model(train_graph, input_features)
    pos_score = model.calc_score(positive_graph, node2embed)
    neg_score = model.calc_score(negative_graph, node2embed)

    if loss_type == 'cross_entropy':
        loss = cross_entropy(pos_score, neg_score, device,
                             reduction, pos_weight)
    elif loss_type == 'hinge':
        loss = hinge(pos_score, neg_score, device,
                             reduction, max_margin)
    else:
        raise ValueError('Unknown loss type')
    if not calc_metrics:
        return loss
    metrics = get_accuracy_scores(pos_score, neg_score)
    return (loss, *metrics)
