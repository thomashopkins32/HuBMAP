from torch import tensor
from torchmetrics.classification import MulticlassJaccardIndex
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

    # 3, 20x20 image masks
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


def test_logits_to_blood_vessel_mask():

    logits = torch.randn((5, 3, 512, 512))
    mask = logits_to_blood_vessel_mask(logits)

    assert mask.shape == (5, 512, 512)
    assert mask.dtype == torch.long

    logits = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
    # make a box of blood vessel predictions in the corner
    logits[0, 2, 0:100, 0:100] = 1.0
    # fill the rest of the predictions for the background
    logits[0, 0, 100:, :] = 1.0
    logits[0, 0, :, 100:] = 1.0

    # covert box to mask
    mask = logits_to_blood_vessel_mask(logits)

    assert mask.shape == (1, 512, 512)
    assert mask.dtype == torch.long
    assert torch.all(logits[0, 2, :, :].type(torch.bool) == mask.type(torch.bool))


def test_kaggle_prediction():
    test_tensor = torch.zeros((3, 512, 512))
    test_tensor[2, 0:100, 0:100] = 1.0
    test_tensor[0, 100:, :] = 1.0
    test_tensor[0, :, 100:] = 1.0

    prediction_entry = kaggle_prediction(1, test_tensor)

    lead, prob, encoding = prediction_entry['prediction_string'].split(' ')

    assert prediction_entry['id'] == 1
    assert prediction_entry['height'] == 512
    assert prediction_entry['width'] == 512
    assert lead == '0'
    assert prob == '0.5761168599128723'
    assert encoding == 'eNozCDHOsTEYDiAgIM4MAPjlJ4Q='


def test_jaccard_index():
    # 3, 20x20 image masks
    predictions = torch.zeros((3, 3, 20, 20), dtype=torch.float)
    targets = torch.zeros((3, 20, 20), dtype=torch.long)

    # single square per image
    predictions[0, 2, 5:10, 5:10] = 1
    targets[0, 9:10, 9:10] = 2 
    predictions[1, 1, 2:3, 2:3] = 1
    targets[1, 2:3, 2:3] = 1
    predictions[2, 1, 1:10, 1:10] = 1
    targets[2, 1:10, 1:10] = 1

    metric = MulticlassJaccardIndex(num_classes=3, average='macro', validate_args=True)
    out = metric(predictions, targets)
    assert np.isclose(out.item(), 0.6728379726409912) 
    