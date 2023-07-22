import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from dataset import *
from models import *
from utils import *

# PARAMETERS
RUN_NAME = 'tesing_run'
BATCH_SIZE = 4
LR = 1e-4
WD = 0.0
MOMENTUM = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 5
RNG = 32
EPOCHS = 10
CHECKPOINT_STEP = 5
CHECKPOINT_LOAD_PATH = None
CHECKPOINT_SAVE_PATH = os.path.join('checkpoints', f'{RUN_NAME}.pt')
INCLUDE_UNSURE = True

torch.manual_seed(RNG)

writer = SummaryWriter(max_queue=1000, flush_secs=300)
dataset = HuBMAP(include_unsure=INCLUDE_UNSURE)
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.9, 0.1], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=False)
model = UNet2d().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

for e in tqdm(range(EPOCHS)):
    if e % VALID_STEP == 0:
        loss = train_one_epoch(e, model, train_loader, loss_func, optimizer, writer=writer, device=DEVICE)
        val_loss, val_metric = validate_one_epoch(e, model, valid_loader, loss_func, writer, device=DEVICE)
    else:
        loss = train_one_epoch(e, model, train_loader, loss_func, optimizer, device=DEVICE)
    if e % CHECKPOINT_STEP == 0:
        save_model_checkpoint(CHECKPOINT_SAVE_PATH, e, model, optimizer, loss, scheduler=scheduler, device=DEVICE)
    writer.add_scalar('gpu_memory_usage', torch.cuda.memory_allocated(DEVICE), global_step=e)
    if scheduler:
        scheduler.step(val_metric)
writer.close()