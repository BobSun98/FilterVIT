import os
import time

from PIL import Image
import torch
from torchvision import transforms
import shutil
from models.convformer import ConvFormer_xs

weight_path = "./weights/checkpoint-model-151.pth"
img_path = "./explainPicture"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvFormer_xs(num_calsses=1000)
checkpoint = torch.load(weight_path,map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()
transform = transforms.Compose([
transforms.Resize(256),  # 将图像缩放到 256x256，以确保可以进行居中裁剪
transforms.CenterCrop(224),  # 居中裁剪到 224x224
transforms.ToTensor(),
transforms.Normalize([0.5] * 3, [0.5] * 3)])

inv_normalize = transforms.Normalize(
    mean=[-1 * 0.5 / 0.5] * 3,  # 与 normalize 中的 mean 相反
    std=[1 / 0.5] * 3  # 与 normalize 中的 std 相反
)
to_pil = transforms.ToPILImage()

def clear_folder(folder_path):
    """
    删除文件夹中的所有内容，但保留文件夹本身
    :param folder_path: 需要清理的文件夹路径
    """
    # 遍历文件夹中的所有文件和子文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # 如果是文件，删除文件
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            # 如果是文件夹，递归删除文件夹
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除文件夹及其所有内容
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


for file in os.listdir(img_path):
    img = Image.open(os.path.join(img_path, file))
    img_tensor = transform(img)
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(0).to(device))
        i = 1
        path = f"0/{i}-pic.png"
        while(os.path.exists(path)):
            i += 1
            path = f"0/{i}-pic.png"
        to_pil(inv_normalize(img_tensor)).save(path)
        input(file)
        clear_folder("./0")



