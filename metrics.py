import numpy as np
import torch


def softmax_cross_entropy_with_logits():
    def fn(logits, target):
        confidence = torch.nn.functional.log_softmax(logits, 1)
        return torch.nn.NLLLoss()(confidence, target)

    return fn




def map3():
    ARR_SCORE_BASE = torch.tensor(np.asarray([1.0, 1.0 / 2.0, 1.0 / 3.0]))
    def fn(logits, target):
        val_logits_top_k, idx_logits_top_k = torch.topk(logits, 3)
        idx_logits_top_k = idx_logits_top_k.t()
        correct = idx_logits_top_k.eq(target.view(1, -1).expand_as(idx_logits_top_k))
        pred = torch.sum(correct, dim=1).type(torch.DoubleTensor)
        score = torch.sum(torch.mul(pred, ARR_SCORE_BASE))
        return score

    return fn


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
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

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
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
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])