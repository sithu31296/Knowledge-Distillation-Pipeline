import torch

def accuracy(pred: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> float:
    maxk = max(topk)
    _, pred = pred.topk(maxk, 1)
    pred = pred.t()
    correct = pred == target.reshape(1, -1).expand_as(pred)

    return [correct[:k].reshape(-1).float().sum(0)*100. /  target.shape[0] for k in topk]