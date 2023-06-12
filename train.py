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
BATCH_SIZE = 5
LR = 0.001
WD = 0.0
MOMENTUM = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 10
RNG = 16
EPOCHS = 10

SHOW_MODEL = False
SHOW_SAMPLES = False

torch.manual_seed(RNG)

writer = SummaryWriter("runs", max_queue=1000, flush_secs=300)
dataset = HuBMAP()
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.9, 0.1], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=False)
model = UNet2d().half().to(DEVICE)
allocated, cached = get_model_gpu_memory(model)
print(f"GPU memory allocated: {allocated:.2f} MB")
print(f"GPU memory cached: {cached:.2f} MB")
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
global_step = 0

if SHOW_MODEL:
    images, labels = next(iter(train_loader))
    x = images.to(DEVICE)

    writer.add_graph(model, x)
if SHOW_SAMPLES:
    images, labels = next(iter(train_loader))
    x = images.to(DEVICE)
    y = labels.long().to(DEVICE)
    logits = model(x)
    loss = loss_func(logits, y)
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)

for e in tqdm(range(EPOCHS)):
    for i, d in enumerate(train_loader):
        x = d['image']
        y = d['blood_vessel_mask']
        glom = d['glomerulus_mask']
        uns = d['unsure_mask']
        x = x.to(DEVICE)
        y = y.long().to(DEVICE)
        logits = model(x)
        loss = loss_func(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), global_step)

        if i % VALID_STEP == 0:
            global_step += 1
    model.eval()
    for i, d in enumerate(valid_loader):
        x = d['image']
        y = d['blood_vessel_mask']
        glom = d['glomerulus_mask']
        uns = d['unsure_mask']
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = loss_func(logits, y)
        writer.add_scalar("Loss/valid", loss.item(), global_step)
    model.train()
    # scheduler.step(acc)
writer.close()