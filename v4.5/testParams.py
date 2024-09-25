from datetime import time

import timm
import torch
from models.convformer import ConvFormer_xs, ConvFormer_xxs,DropOutVIT_xs


def create_and_print_model_params(model_name):
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = timm.create_model(model_name, pretrained=False, num_classes=100)
    # 打印模型名称和参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, Parameters: {num_params:,}")

def create_and_print_model_params_ConvFormer_xxs():
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = ConvFormer_xs(num_calsses=100)
    # 打印模型名称和参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {'ConvFormer'}, Parameters: {num_params:,}")

def create_and_print_model_params_Dropout():
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = DropOutVIT_xs(num_calsses=100)
    # 打印模型名称和参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {'ConvFormer'}, Parameters: {num_params:,}")

# 模型名称列表
model_names = [
    'mobilenetv2_100',  # MobileNetV2
    'mobilenetv3_large_100',  # MobileNetV3 Large
    'mobilenetv3_small_100',  # MobileNetV3 Small
    'efficientnet_lite0',  # EfficientNet Lite 0
    'efficientnet_lite1',  # EfficientNet Lite 1
    'efficientnet_lite2',  # EfficientNet Lite 2
    'efficientnet_lite3',  # EfficientNet Lite 3
    'efficientnet_lite4',  # EfficientNet Lite 4
    'tiny_vit_5m_224',  # TinyViT 5M
    'tiny_vit_11m_224',  # TinyViT 11M
    'tiny_vit_21m_224',  # TinyViT 21M
    'levit_128',  # LeViT 128
    'levit_256',  # LeViT 256
    'ghostnet_100',  # GhostNet 100
    'tinynet_a',  # TinyNet A
    'lcnet_100',  # LCNet 100
    'mobilevit_s'
]

# 遍历模型列表并打印每个模型的参数量
for model_name in model_names:
    create_and_print_model_params(model_name)

create_and_print_model_params_ConvFormer_xxs()
create_and_print_model_params_Dropout()
