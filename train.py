import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import *
from models import *
from utils import *

# Needed for multiprocessing
if __name__ == '__main__':
    # PARAMETERS
    RUN_NAME = 'full_baseline'
    BATCH_SIZE = 4
    LR = 0.1
    WD = 0.0
    MOMENTUM = 0.95
    DATA_TRANSFORMATIONS = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VALID_STEP = 10
    RNG = 32
    EPOCH_START = 0
    EPOCH_END = 200
    CHECKPOINT_STEP = 10
    CHECKPOINT_LOAD_PATH = None
    CHECKPOINT_SAVE_PATH = os.path.join('checkpoints', f'{RUN_NAME}.pt')
    INCLUDE_UNSURE = True

    torch.manual_seed(RNG)
    dataset = HuBMAP(include_unsure=INCLUDE_UNSURE)
    generator = torch.Generator().manual_seed(RNG)
    train_data, valid_data = random_split(dataset, [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f'Calculating weight rescaling from {len(train_data)} training points')
    # get raw counts of classes for weight rescaling
    bv_count = 0
    glom_count = 0
    bg_count = 0
    for batch in tqdm(train_loader):
        m = batch['mask']
        bv_count += torch.count_nonzero(m == 2).item()
        glom_count += torch.count_nonzero(m == 1).item()
        bg_count += torch.count_nonzero(m == 0).item()
    total_count = bv_count + glom_count + bg_count

    weight_rescale = torch.tensor([
        total_count / (3 * bg_count),
        total_count / (3 * glom_count),
        total_count / (3 * bv_count)
    ], dtype=torch.float, device=DEVICE)
    print(f'Class rescaling will be done based on the following: {weight_rescale}')


    train_metric = MulticlassJaccardIndex(num_classes=3, average='macro').to(DEVICE)
    val_metric = MulticlassJaccardIndex(num_classes=3, average='macro').to(DEVICE)

    writer = SummaryWriter()
    model = UNet2d().to(DEVICE)
    loss_func = nn.CrossEntropyLoss(weight=weight_rescale)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    #scheduler = None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH_END, eta_min=1e-6, verbose=True)
    if CHECKPOINT_LOAD_PATH:
        EPOCH_START = load_model_checkpoint(CHECKPOINT_LOAD_PATH, model, optimizer, scheduler=scheduler) 

    for e in tqdm(range(EPOCH_START + 1, EPOCH_END + 1)):
        if e % VALID_STEP == 0:
            train_one_epoch(
                e,
                model,
                train_loader,
                loss_func,
                optimizer,
                train_metric,
                writer=writer,
                data_transforms=DATA_TRANSFORMATIONS,
                device=DEVICE
            )
            val_loss, metric = validate_one_epoch(
                e,
                model,
                valid_loader,
                loss_func,
                val_metric,
                writer,
                data_transforms=False,
                device=DEVICE
            )
        else:
            train_one_epoch(
                e,
                model,
                train_loader,
                loss_func,
                optimizer, 
                train_metric,
                data_transforms=DATA_TRANSFORMATIONS,
                device=DEVICE
            )
        if scheduler:
            scheduler.step()
        if e % CHECKPOINT_STEP == 0:
            save_model_checkpoint(CHECKPOINT_SAVE_PATH, e, model, optimizer, scheduler=scheduler)
    writer.close()
