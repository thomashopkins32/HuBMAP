import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import torch
from tqdm import tqdm


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.count_nonzero(preds == labels) / preds.shape[0]


def memory_usage_stats(model, optimizer, batch_size=1, device='cuda'):
    print(f'Starting memory: {torch.cuda.memory_allocated(device) * 1e-6}')
    model.to(device)
    print(f'After model sent to {device}: {torch.cuda.memory_allocated(device) * 1e-6}')
    for i in range(3):
        sample = torch.randn((batch_size, 3, 512, 512))
        print(f'Step {i}')
        before = torch.cuda.memory_allocated(device) * 1e-6
        out = model(sample.to(device)).sum()
        after = torch.cuda.memory_allocated(device) * 1e-6
        print(f'After forward pass: {after}')
        print(f'Memory used by forward pass: {after - before}')
        out.backward()
        after = torch.cuda.memory_allocated(device) * 1e-6
        print(f'After backward pass: {after}')
        optimizer.step()
        print(f'After optimizer step: {torch.cuda.memory_allocated(device) * 1e-6}')
    torch.cuda.empty_cache()


def memory_usage_stats_grad_scaler(model, optimizer, batch_size=1, device='cuda'):
    if device != 'cuda':
        print('This function requires device to be "cuda".')
        return
    print(f'Starting memory: {torch.cuda.memory_allocated(device) * 1e-6}')
    model.to(device)
    print(f'After model sent to {device}: {torch.cuda.memory_allocated(device) * 1e-6}')
    scaler = torch.cuda.amp.GradScaler()
    for i in range(3):
        sample = torch.randn((batch_size, 3, 512, 512))
        print(f'Step {i}')
        before = torch.cuda.memory_allocated(device) * 1e-6
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(sample.to(device)).sum()
        after = torch.cuda.memory_allocated(device) * 1e-6
        print(f'After forward pass: {after}')
        print(f'Memory used by forward pass: {after - before}')
        scaler.scale(out).backward()
        after = torch.cuda.memory_allocated(device) * 1e-6
        print(f'After backward pass: {after}')
        scaler.step(optimizer)
        print(f'After optimizer step: {torch.cuda.memory_allocated(device) * 1e-6}')
        scaler.update()
    torch.cuda.empty_cache()


def average_precision(prediction, target, iou_threshold=0.6):
    '''
    Computes the average precision (AP) for an instance segmentation task on a single image.

    Parameters
    ----------
    prediction : np.array
        Boolean prediction mask
    target : np.array
        Boolean ground-truth mask
    iou_threshold : float, optional
        Threshold at which to count predictions as positive predictions

    Returns
    -------
    AP : float
        Average precision score
    '''
    pred_regions, pred_region_count = measure.label(prediction, return_num=True)
    target_regions, target_region_count = measure.label(target, return_num=True)
    true_positives = 0
    false_positives = 0
    for p in range(1, pred_region_count + 1):
        pred_region_mask = pred_regions == p
        max_iou = 0.0
        for t in range(1, target_region_count + 1):
            target_region_mask = target_regions == t
            # Compute IoU this region pair
            intersection = np.logical_and(pred_region_mask, target_region_mask)
            union = np.logical_or(pred_region_mask, target_region_mask)
            intersection_count = np.count_nonzero(intersection)
            union_count = np.count_nonzero(union)
            if intersection_count > 0 and union_count > 0:
                iou = intersection_count / union_count
                max_iou = max(max_iou, iou)
        if max_iou > iou_threshold:
            true_positives += 1
        else:
            false_positives += 1
    if (true_positives == 0 and false_positives == 0) or target_region_count == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / target_region_count

    return precision * recall


def mAP(predictions, targets, iou_threshold=0.6):
    averages = 0.0
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    for p, t in zip(predictions, targets):
        averages += average_precision(p, t, iou_threshold=iou_threshold)
    return averages / len(predictions)


def logits_to_blood_vessel_mask(logits):
    return (torch.argmax(torch.softmax(logits, dim=1), dim=1) == 2).type(torch.long)


def train_one_epoch(epoch, model, train_loader, loss_func, optimizer, writer=None, device='cpu', **kwargs):
    model.train()
    for d in train_loader:
        optimizer.zero_grad()
        x = d['image']
        y = d['mask']
        x = x.to(device)
        y = y.long().to(device)
        logits = model(x)
        loss = loss_func(logits, y)
        loss.backward()
        optimizer.step()
        if writer:
            with torch.no_grad():
                predictions = logits_to_blood_vessel_mask(logits)
                mAP_value = mAP(predictions, y, **kwargs)
                writer.add_scalar("Loss/train", loss.item(), global_step=epoch)
                writer.add_scalar("mAP/train", mAP_value, global_step=epoch)


def validate_one_epoch(epoch, model, valid_loader, loss_func, writer, device='cpu', **kwargs):
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(valid_loader):
            x = d['image']
            y = d['mask']
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_func(logits, y)
            predictions = logits_to_blood_vessel_mask(logits)
            mAP_value = mAP(predictions, y, **kwargs)
            writer.add_scalar("Loss/valid", loss.item(), global_step=epoch)
            writer.add_scalar("mAP/valid", mAP_value, global_step=epoch)
