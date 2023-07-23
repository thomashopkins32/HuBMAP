import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import *
from models import *
from utils import *

# PARAMETERS
RUN_NAME = 'tesing_run'
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = 32
CHECKPOINT_LOAD_PATH = os.path.join('checkpoints', f'{RUN_NAME}.pt')

torch.manual_seed(RNG)

dataset = HuBMAP(submission=True)
generator = torch.Generator().manual_seed(RNG)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
model = UNet2d().to(DEVICE)
if CHECKPOINT_LOAD_PATH:
    EPOCH_START = load_model_checkpoint(CHECKPOINT_LOAD_PATH, model)
model.eval()

submission_entries = pd.DataFrame(columns=['id', 'height', 'width', 'prediction_string'])
for d in loader:
    id = d['id']
    x = d['image']
    x = x.to(DEVICE)
    logits = model(x)
    for i in range(logits.shape[0]):
        submission = kaggle_prediction(id, logits[i, :, :, :])
        submission_df = pd.DataFrame(submission)
        submission_entries = pd.concat((submission_entries, submission_df), axis=0)

submission_entries.to_csv('submission.csv')
