import os
import argparse
from datetime import datetime

import timm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from torchvision.transforms import autoaugment


from models.convformer import ConvFormer_xs,DropOutVIT_xs
# Import your utility functions
from my_utils import train_one_epoch, evaluate


def get_data_transforms():
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]),
    }
    return data_transforms


def get_dataloaders(data_path, batch_size, num_workers, data_transforms):
    train_dataset = ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=data_transforms['train']
    )
    val_dataset = ImageFolder(
        root=os.path.join(data_path, 'val'),
        transform=data_transforms['val']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(model_name, num_classes, device):
    model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    return model


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, mixup_fn, writer, start_epoch=0):
    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            mixup_fn=mixup_fn
        )

        # Step the scheduler
        scheduler.step()

        # Validate
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        # torch.save(checkpoint, f"./weights/checkpoint-model-{epoch}.pth")
        print(f"Epoch {epoch} completed. Model saved.")


def train(model, data_path, log_path):
    parser = argparse.ArgumentParser(description='Train a model on ImageNet-100.')
    parser.add_argument('--num-classes', type=int, default=100, help='Number of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate for the scheduler')
    parser.add_argument('--weights', type=str, default='', help='Path to pretrained weights (optional)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    args = parser.parse_args()

    device = torch.device(args.device)
    model.to(device)
    os.makedirs("./weights", exist_ok=True)



    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=log_path)

    # Get data transforms and loaders
    data_transforms = get_data_transforms()
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader, val_loader = get_dataloaders(data_path, args.batch_size, num_workers, data_transforms)

    # Initialize Mixup (if needed)
    mixup_fn = Mixup(
        mixup_alpha=0,
        cutmix_alpha=0,
        cutmix_minmax=None,
        prob=0,
        mode="batch",
        label_smoothing=0.1,
        num_classes=args.num_classes
    )

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Optionally load weights
    start_epoch = 0
    if args.weights:
        assert os.path.exists(args.weights), f"Weights file '{args.weights}' not found."
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        mixup_fn=mixup_fn,
        writer=writer,
        start_epoch=start_epoch
    )

    writer.close()


model_names = [
    'mobilenetv2_100',  # MobileNetV2
    'mobilenetv3_large_100',  # MobileNetV3 Large
    'mobilenetv3_small_100',  # MobileNetV3 Small
    'efficientnet_lite0',  # EfficientNet Lite 0
    'tiny_vit_11m_224',  # TinyViT 11M
    'levit_128',  # LeViT 128z
    'ghostnet_100',  # GhostNet 100
    'tinynet_a',  # TinyNet A
    'lcnet_100',  # LCNet 100
    'mobilevit_s'
]

skip_list = [
    'mobilenetv2_100',  # MobileNetV2
    'mobilenetv3_large_100',  # MobileNetV3 Large
    'mobilenetv3_small_100',  # MobileNetV3 Small
    'efficientnet_lite0',  # EfficientNet Lite 0
]
data_path = os.path.join("/home/bobsun/bob/data/imagenet100", "1")
log_path = f"/home/bobsun/bob/convformer/runs/DropOutVit_xs_1"
model = DropOutVIT_xs(num_calsses=100)
train(model=model,
      data_path=data_path,
      log_path=log_path)
#
# for i in range(1, 6):
#     for model_name in model_names:
#         if model_name in skip_list and i == 1:
#             continue
#         data_path = os.path.join("/home/bobsun/bob/data/imagenet100", str(i))
#         model = timm.create_model(model_name, pretrained=False, num_classes=100)
#         log_path = f"/home/bobsun/bob/convformer/runs/{model_name}_{i}"
#         print(f"开始训练{model_name},第{i}次")
#         train(    model=model,
#     data_path=data_path,
#     log_path=log_path)
#
# model = ConvFormer_xs(num_calsses=100)
# for i in range(2, 6):
#     data_path = os.path.join("/home/bobsun/bob/data/imagenet100", str(i))
#     log_path = f"/home/bobsun/bob/convformer/runs/FilterMobileVit_xs_{i}"
#     train(model=model,
#           data_path=data_path,
#           log_path=log_path)