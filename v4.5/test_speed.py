import torch
import time
from models.convformer import DropOutVIT_xs, ConvFormer_xs,ConvFormer_xxs
import timm

def test_speed_fps(model):
    # 设置模型为评估模式
    model.eval()

    # 创建一个虚拟输入
    input = torch.randn(1, 3, 224, 224)  # change to your actual batch size
    if torch.cuda.is_available():
        input = input.cuda()
        model = model.cuda()

    # 预热 GPU（如果使用）
    for _ in range(10):
        model(input)

    # 开始计时
    start_time = time.time()

    # 循环运行模型推理以获得更准确的平均时间
    num_runs = 100
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(input)

    # 停止计时
    end_time = time.time()

    # 计算总推理时间
    total_time = end_time - start_time

    # 计算每秒处理的帧数 (FPS)
    fps = num_runs / total_time

    return fps


def create_and_print_model_speed(model_name):
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = timm.create_model(model_name, pretrained=False, num_classes=100)
    # 打印模型名称和参数量
    fps = test_speed_fps(model)

    print(f"model:{model_name},fps: {fps:.6f} ")

def create_and_print_model_speed_ConvFormer_xxs():
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = DropOutVIT_xs(num_calsses=100)
    # 打印模型名称和参数量
    fps = test_speed_fps(model)
    print(f"model:{'DropOutVIT_xs'},fps: {fps:.6f} ")

def create_and_print_model_speed_ConvFormer_xxs1():
    # 创建模型，不使用预训练权重，并设置num_classes为100
    model = ConvFormer_xs(num_calsses=100)
    # 打印模型名称和参数量
    fps = test_speed_fps(model)
    print(f"model:{'FilterMobileVit'},fps: {fps:.6f} ")

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

create_and_print_model_speed_ConvFormer_xxs()
create_and_print_model_speed_ConvFormer_xxs1()

# 遍历模型列表并打印每个模型的参数量
for model_name in model_names:
    create_and_print_model_speed(model_name)


