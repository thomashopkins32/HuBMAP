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
    preds = predictions.cpu().detach().numpy()
    labels = targets.cpu().detach().numpy()

    intersection = np.logical_and(preds, labels).astype(int)
    union = np.logical_or(preds, labels).astype(int)

    average_precisions = []
    for img_idx in range(len(intersection)):
        true_positives = 0
        false_positives = 0
        intersection_labels, intersection_region_count = measure.label(intersection[img_idx], return_num=True)
        union_labels, union_region_count = measure.label(union[img_idx], return_num=True)

        if intersection_region_count == 0:
            break

        ious = []
        for j in range(1, intersection_region_count + 1):
            intersection_count = np.count_nonzero(intersection_labels == j)
            union_count = np.count_nonzero(union_labels == j)
            
            iou = intersection_count / union_count

            ious.append(iou)

        print(ious)

        for iou in ious:
            if iou >= iou_threshold:
                true_positives += 1
            else:
                false_positives += 1

        print(true_positives)
        print(false_positives)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / intersection_region_count

        print(precision)
        print(recall)

        ap = precision * recall

        average_precisions.append(ap)

    return np.mean(average_precisions)
