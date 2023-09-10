import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import *
from models import *
from utils import *

# PARAMETERS
RUN_NAME = 'full_baseline'
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = 32
CHECKPOINT_LOAD_PATH = os.path.join('checkpoints', f'{RUN_NAME}.pt')

dataset = HuBMAP(submission=False)
metric = MulticlassJaccardIndex(num_classes=3, average='macro').to(DEVICE)
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.8, 0.2], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
model = UNet2d().to(DEVICE)
if CHECKPOINT_LOAD_PATH:
    EPOCH_START = load_model_checkpoint(CHECKPOINT_LOAD_PATH, model)

model.eval()
with torch.no_grad():
    for d in valid_loader:
        id = d['id']
        x = d['image']
        y = d['mask']
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        metric(logits, y)
        iou = metric.compute()
        mask = logits_to_mask(logits)
        print(f'Id: {id[0]}')
        print(f'IoU: {iou}')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(torch.permute(x[0].detach().cpu(), (1, 2, 0)))
        ax1.set_title('Original Image')
        ax2.imshow(y[0].detach().cpu())
        ax2.set_title('Ground Truth Mask')
        ax3.imshow(mask[0].detach().cpu())
        ax3.set_title('Predicted Mask')
        plt.show()
