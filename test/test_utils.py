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
    assert mAP(predictions, targets) == 0.0

    # single square
    predictions[0, 5:10, 5:10] = 1
    targets[0, 5:10, 5:10] = 1
    predictions[1, 2:3, 2:3] = 1
    targets[1, 2:3, 2:3] = 1
    predictions[2, 1:10, 1:10] = 1
    targets[2, 1:10, 1:10] = 1

    assert mAP(predictions, targets) == 1.0

    # two squares
    predictions[0, 1:3, 1:3] = 1
    targets[0, 1:3, 1:3] = 1
    predictions[0, 5:10, 5:10] = 1
    targets[0, 5:10, 5:10] = 1
    predictions[1, 10:19, 10:19] = 1
    targets[1, 10:19, 10:19] = 1
    predictions[1, 1:8, 10:19] = 1
    targets[1, 1:8, 10:19] = 1
    predictions[2, 10:19, 1:8] = 1
    targets[2, 10:19, 1:8] = 1
    predictions[2, 1:8, 10:19] = 1
    targets[2, 1:8, 10:19] = 1

    assert mAP(predictions, targets) == 1.0


def test_mAP_imperfect_prediction():

    # 5, 20x20 image masks
    predictions = torch.zeros((3, 20, 20), dtype=torch.long)
    targets = torch.zeros((3, 20, 20), dtype=torch.long)

    # single square per image
    predictions[0, 5:10, 5:10] = 1
    targets[0, 9:10, 9:10] = 1
    predictions[1, 2:3, 2:3] = 1
    targets[1, 2:3, 2:3] = 1
    predictions[2, 1:10, 1:10] = 1
    targets[2, 1:10, 1:10] = 1

    assert np.isclose(mAP(predictions, targets), 2 / 3)

    # first image has two squares
    # one matching one not
    predictions[0, 1:3, 1:3] = 1
    targets[0, 1:3, 1:3] = 1

    assert np.isclose(mAP(predictions, targets), 2.25 / 3)


def test_softmax():

    logits = torch.randn((5, 2, 512, 512))

    preds = torch.softmax(logits, dim=1)

    assert preds.shape == (5, 2, 512, 512)
    total = torch.sum(preds, dim=1)
    assert total.shape == (5, 512, 512)
    assert torch.allclose(total, torch.ones((5, 512, 512)))

    predictions = torch.argmax(preds, dim=1).type(torch.long)

    assert predictions.shape == (5, 512, 512)
    assert predictions.dtype == torch.long
