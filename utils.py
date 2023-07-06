import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torch


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.count_nonzero(preds == labels) / preds.shape[0]


def get_model_gpu_memory(model):
    torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    memory_cached = torch.cuda.memory_cached() / 1024 ** 2
    return memory_allocated, memory_cached


def average_precision(prediction, target, iou_threshold=0.6):
    '''
    Computes the average precision (AP) for an instance segmentation task on a single image.

    Parameters
    ----------
    prediction : np.array
        Boolean prediction mask
    target : np.array
        Boolean ground-truth mask
    iou_threshold : float, optional
        Threshold at which to count predictions as positive predictions

    Returns
    -------
    AP : float
        Average precision score
    '''
    pred_regions, pred_region_count = measure.label(prediction, return_num=True)
    target_regions, target_region_count = measure.label(target, return_num=True)
    true_positives = 0
    false_positives = 0
    for p in range(1, pred_region_count + 1):
        pred_region_mask = pred_regions == p
        max_iou = 0.0
        for t in range(1, target_region_count + 1):
            target_region_mask = target_regions == t
            # Compute IoU this region pair
            intersection = np.logical_and(pred_region_mask, target_region_mask)
            union = np.logical_or(pred_region_mask, target_region_mask)
            intersection_count = np.count_nonzero(intersection)
            union_count = np.count_nonzero(union)
            if intersection_count > 0 and union_count > 0:
                iou = intersection_count / union_count
                max_iou = max(max_iou, iou)
        if max_iou > iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
    if (true_positives == 0 and false_positives == 0) or target_region_count == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / target_region_count

    return precision * recall


def mAP(predictions, targets, iou_threshold=0.6):
    averages = 0.0
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    for p, t in zip(predictions, targets):
        averages += average_precision(p, t, iou_threshold=iou_threshold)
    return averages / len(predictions)
