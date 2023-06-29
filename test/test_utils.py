from torch import tensor
import matplotlib.pyplot as plt
from ..utils import *

def test_accuracy():
    
    logits = tensor([[0.2, 0.8], [0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.49999, 0.50001]], dtype=torch.float)
    labels = tensor([1, 1, 0, 1, 1], dtype=torch.long)

    assert accuracy(logits, labels) == 1.0

    labels = tensor([0, 0, 1, 0, 0], dtype=torch.long)

    assert accuracy(logits, labels) == 0.0


def test_mAP_perfect_prediction():

    # 5, 20x20 image masks
    predictions = torch.zeros((3, 20, 20), dtype=torch.long)
    targets = torch.zeros((3, 20, 20), dtype=torch.long)

    # nothing to predict
    #assert mAP(predictions, targets) == 0.0

    # single square
    predictions[0, 5:10, 5:10] = 1
    targets[0, 5:10, 5:10] = 1
    predictions[1, 2:3, 2:3] = 1
    targets[1, 2:3, 2:3] = 1
    predictions[2, 1:10, 1:10] = 1
    targets[2, 1:10, 1:10] = 1

    assert mAP(predictions, targets) == 1.0