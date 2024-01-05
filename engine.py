import torch
import math
import sys
from torch.nn import functional as F
from tqdm import tqdm
from utils.metrics import Metrics
import utils.distributed_utils as utils
from utils.losses import DiceLoss, OhemCrossEntropy


def train_one_epoch(model, optimizer,
                    ce_loss_fn: OhemCrossEntropy, dice_loss_fn: DiceLoss,
                    dataloader, scheduler,
                    epoch, device, print_freq, clip_grad, clip_mode, loss_scaler):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for iter, samples in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        img, lbl = samples['image'], samples['label']
        img = img.to(device)
        lbl = lbl.squeeze(1).to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(img)
            ce_loss = ce_loss_fn(logits, lbl[:].long())
            dice_loss = dice_loss_fn(logits, lbl, softmax=True)
            loss = 0.4 * ce_loss + 0.6 * dice_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        scheduler.step()
        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value, lr=lr)

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()

    return metric_logger.meters["loss"].global_avg, lr



@torch.no_grad()
def evaluate(args, model, dataloader, device, print_freq):
    model.eval()

    confmat = utils.ConfusionMatrix(args.num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples in metric_logger.log_every(dataloader, print_freq, header):
        images, labels = samples['image'], samples['label']
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze(1).to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)

        confmat.update(labels.flatten(), outputs.argmax(1).flatten())

    confmat.reduce_from_all_processes()
    torch.cuda.empty_cache()

    return confmat



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for idx, samples in tqdm(dataloader):
        images, labels = samples['image'], samples['label']
        labels = labels.squeeze(1).to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou