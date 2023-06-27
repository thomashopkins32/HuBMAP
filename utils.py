import numpy as np
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


def mAP(predictions, targets, iou_threshold=0.6):
    '''
    Computes the mean average precision (mAP) for an instance segmentation task.

    Parameters
    ----------
    predictions : torch.tensor
        Boolean prediction masks
    targets : torch.tensor
        Boolean ground-truth masks
    iou_threshold : float, optional
        Threshold at which to count predictions as positive predictions

    Returns
    -------
    mAP : float
        Mean average precision score
    '''
    true_positives = []
    false_positives = []
    total_targets = len(targets)

    intersection = torch.logical_and(predictions, targets)
    union = torch.logical_or(predictions, targets)

    intersection_labels = measure.label(intersection)
    union_labels = measure.label(union)

    num_intersecting_regions = np.max(intersection_labels, axis=(1, 2))
    num_union_regions = np.max(union_labels, axis=(1, 2))

    ious = num_intersecting_regions / num_union_regions

    for i in range(len(predictions)):
        best_iou = np.max(ious[i])
        best_index = np.argmax(ious[i])

        if best_iou >= iou_threshold and best_index not in true_positives:
            true_positives.append(best_index)
        else:
            false_positives.append(predictions[i])

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / total_targets

    ap = precision * recall

    return ap
