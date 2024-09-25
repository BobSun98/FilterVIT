import os

import argparse


import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from torchvision.datasets.imagenet import ImageNet


from my_utils import train_one_epoch, evaluate#, CustomCosineAnnealingLR
from models.convformer import ConvFormer_xxs,ConvFormer_xs,ConvFormer_s
from timm.data import Mixup
from torchvision.transforms import autoaugment
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import torch.nn as nn


def main(args):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device(args.device)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    mixup_fn = Mixup(mixup_alpha=0, cutmix_alpha=0, cutmix_minmax=None, prob=0, mode="batch", label_smoothing=0.1,
                     num_classes=args.num_classes)

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     autoaugment.TrivialAugmentWide(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0,
                                     #                         inplace=False)
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = ImageNet(root=args.data_path, split="train", transform=data_transform["train"])
    val_dataset = ImageNet(root=args.data_path, split="val", transform=data_transform["val"])
    # train_dataset = torchvision.datasets.ImageFolder(root=args.data_path + "/train", transform=data_transform["train"])
    # val_dataset = torchvision.datasets.ImageFolder(root=args.data_path + "/val", transform=data_transform["val"])
    # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=data_transform["train"])
    # val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = ConvFormer_xs(num_calsses=args.num_classes).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, args.schedule_epochs, args.min_lr)

    base_epoch = 0
    if args.weights != "":  # 加载权重
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        checkpoint = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print("load model")
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("load optimizer")
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("load lr_scheduler")
        base_epoch = checkpoint["epoch"] + 1

    for epoch in range(base_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                mixup_fn=mixup_fn
                                                )

        if epoch < args.schedule_epochs:
            scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, "./weights/checkpoint-model-{}.pth".format(epoch))
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--schedule_epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--warm_iters', type=int, default=5000)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/bobsun/bob/data/imagenet1000") # ImageNet/ILSVRC2012"
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--transfer_weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else "cpu",
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
