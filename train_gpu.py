import os
import re
import torch
import datetime
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DistributedSampler, RandomSampler
# from torch import distributed as dist
from timm.models import create_model
from timm.utils import NativeScaler
from models import bra_unet_128, bra_unet_224, bra_unet_256
from datasets import Synapse_dataset
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import get_loss, DiceLoss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer
from utils import distributed_utils as dist
import utils
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from engine import train_one_epoch, evaluate


def get_argparser():
    parser = argparse.ArgumentParser('Pytorch BRAU-Net++ Model training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='E:/Synapse/Synapse', help="path to Dataset")
    parser.add_argument("--list_path", type=str, default='./lists/lists_Synapse')
    parser.add_argument("--image_size", type=int, default=256, help="input size")
    # parser.add_argument("--base_size", type=int, default=256, help="base size")
    parser.add_argument("--ignore_label", type=int, default=255, help="the dataset ignore_label")
    parser.add_argument("--dataset", type=str, default='synapse',
                        choices=['cityscapes', 'pascal', 'coco', 'synapse'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=9, help="num classes (default: 19 for synapse)")
    parser.add_argument("--pin_mem", type=bool, default=True, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=8, help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 2)')

    # BRAU-Net++ Options
    parser.add_argument("--model", type=str, default='bra_unet_256',
                        choices=['bra_unet_128', 'bra_unet_224', 'bra_unet_256'], help='model type')

    # Train Options
    parser.add_argument("--amp", type=bool, default=True, help='auto mixture precision')
    parser.add_argument("--epochs", type=int, default=2, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=8, help='num_workers, set it equal 0 when run programs in windows platform')
    # parser.add_argument("--DDP", type=bool, default=False)
    parser.add_argument("--train_print_freq", type=int, default=50)
    parser.add_argument("--val_print_freq", type=int, default=10)

    # Loss Options
    parser.add_argument("--loss_fn_name", type=str, default='OhemCrossEntropy')

    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    parser.add_argument("--lr_scheduler", type=str, default='WarmupPolyLR')
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--lr_warmup", type=int, default=10)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1)

    # save checkpoints
    parser.add_argument("--save_weights_dir", default='./save_weights', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    # parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    # training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if not os.path.exists(args.save_weights_dir):
        os.makedirs(args.save_weights_dir)

    # start = time.time()
    best_mIoU = 0.0
    device = args.device

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train_transform = get_train_augmentation(args.image_size, seg_fill=args.ignore_label)
    # val_transform = get_val_augmentation(args.image_size)

    x_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    y_transforms = T.Compose([T.ToTensor()])


    train_set = Synapse_dataset(base_dir=f'{args.data_root}/train_npz', list_dir=args.list_path,
                                split="train", img_size=args.image_size,
                                norm_x_transform=x_transforms, norm_y_transform=y_transforms)
    valid_set = Synapse_dataset(base_dir=f'{args.data_root}/test_vol_h5', list_dir=args.list_path,
                                split="test_vol", img_size=args.image_size)

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        args=args
    )

    model = model.to(device)

    if args.distributed:
        sampler = DistributedSampler(train_set, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        sampler = RandomSampler(train_set)

    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem)


    ce_loss_fn = get_loss(args.loss_fn_name, args.ignore_label, None)
    dice_loss_fn = DiceLoss(args.num_classes)

    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # scheduler = get_scheduler(args.lr_scheduler, optimizer, args.epochs * iters_per_epoch, args.lr_power,
    #                           iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)

    scheduler = create_lr_scheduler(optimizer, len(trainloader), args.epochs, warmup=True)

    loss_scaler = NativeScaler()
    # scaler = GradScaler(enabled=args.amp) if torch.cuda.is_bf16_supported() else None

    # writer = SummaryWriter(str(args.save_dir / 'logs'))

    if args.resume:
        checkpoint_save_path = './save_weights/best_model.pth'
        checkpoint = torch.load(checkpoint_save_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_mIoU = checkpoint['best_mIoU']
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        print(f'The Best MeanIou is {best_mIoU:.4f}')

    for epoch in range(args.epochs):

        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch(model, optimizer, ce_loss_fn, dice_loss_fn, trainloader, scheduler,
                                        epoch, device, args.train_print_freq, args.clip_grad, args.clip_mode,
                                        loss_scaler)

        confmat = evaluate(args, model, valloader, device, args.val_print_freq)

        val_info = str(confmat)
        print(val_info)

        if dist.is_main_process():
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                f.write(train_info + val_info + "\n\n")

        with open(results_file, 'r') as file:
            text = file.read()
        match = re.search(r'mean IoU:\s+(\d+\.\d+)', text)
        if match:
            mean_iou = float(match.group(1))

        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            checkpoint_save = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_mIoU": best_mIoU
            }
            if args.amp:
                checkpoint_save['scaler'] = loss_scaler.state_dict()
            torch.save(checkpoint_save, f'{args.save_weights_dir}/best_model.pth')

    # writer.close()
    # end = time.gmtime(time.time() - start)

    # table = [
    #     ['Best mIoU', f"{best_mIoU:.2f}"],
    #     ['Total Training Time', time.strftime("%H:%M:%S", end)]
    # ]
    # print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch BRAU-Net++ Model training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    fix_seeds(2024)
    setup_cudnn()
    # gpu = setup_ddp()
    main(args)
    # cleanup_ddp()