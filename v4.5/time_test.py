import torch
import time
from models.convformer import ConvFormer_xxs,ConvFormer_xs,ConvFormer_s
import timm
# 确保模型在评估模式
def test_model(model):
    model.eval()

    # 将模型放在合适的设备上，例如GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置批量大小、通道数、高度和宽度
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    # 创建输入张量
    input_tensor = torch.randn(batch_size, channels, height, width).to(device)

    # 预热模型
    for _ in range(10):
        _ = model(input_tensor)

    # 计时开始
    start_time = time.time()

    # 进行多次运行以获得平均值
    num_runs = 100
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)

    # 计时结束
    end_time = time.time()

    # 计算平均推理时间
    avg_inference_time = (end_time - start_time) / num_runs
    print(f"平均推理时间：{avg_inference_time:.4f}秒")

num_classes = 1000
model_dict = {
    "mobilevit_xxs":timm.create_model("mobilevit_xxs", num_classes=num_classes),
    "conVformer_xxs":ConvFormer_xxs(num_calsses=num_classes),
    "mobilevit_xs":timm.create_model("mobilevit_xs", num_classes=num_classes),
    "conVformer_xs": ConvFormer_xs(num_calsses=num_classes),
    "mobilevit_s":timm.create_model("mobilevit_s", num_classes=num_classes),
    "conVformer_s":ConvFormer_s(num_calsses=num_classes),
}

for k in model_dict:
    print(k)
    test_model(model_dict[k])