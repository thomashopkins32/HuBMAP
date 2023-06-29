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
    total_targets = len(targets)

    preds = predictions.cpu().detach().numpy()
    labels = targets.cpu().detach().numpy()

    intersection = np.logical_and(preds, labels).astype(int)
    union = np.logical_or(preds, labels).astype(int)

    average_precisions = []
    for i in range(len(intersection)):
        true_positives = []
        false_positives = []
        intersection_labels, intersection_count = measure.label(intersection[i], return_num=True)
        union_labels, union_count = measure.label(union[i], return_num=True)

        print(intersection_labels)
        print(intersection_count)
        print(union_labels)
        print(union_count)

        ious = []
        for j in range(1, intersection_count + 1):
            intersection_count = np.count_nonzero(intersection_labels == j)
            union_count = np.count_nonzero(union_labels == j)
            
            iou = intersection_count / union_count
            ious.append(iou)

        print(ious)

        # TODO: Fix this part, not sure why we take the maximum here
        for j in range(len(ious)):
            best_iou = np.max(ious[j])
            best_index = np.argmax(ious[j])

            if best_iou >= iou_threshold and best_index not in true_positives:
                true_positives.append(best_index)
            else:
                false_positives.append(j)

        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / total_targets

        ap = precision * recall

        average_precisions.append(ap)

    return np.mean(average_precisions)
