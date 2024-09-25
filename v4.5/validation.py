import torch
import torchvision
from PIL import Image
from torchvision import transforms

from models.convformer import ConvFormer_s,ConvFormer_xs,ConvFormer_xxs

import os
from PIL import Image


def crop_to_square(image):
    width, height = image.size

    if width == height:
        return image
    elif width > height:
        left = (width - height) // 2
        right = left + height
        return image.crop((left, 0, right, height))
    else:
        top = (height - width) // 2
        bottom = top + width
        return image.crop((0, top, width, bottom))


def process_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', ".JPEG")):
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                square_image = crop_to_square(image)
                square_image.save(file_path)
                print(f"Processed {file_path}")


process_images("pics")

data_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

model = ConvFormer_xxs(num_calsses=100)  # .to("cuda")

checkpoint = torch.load("convFormer_4.6_xxs_img100.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])


def imshow(img):
    img = torchvision.utils.make_grid(img)
    torchvision.utils.save_image(img, "111.png")


input = torch.zeros(8, 3, 224, 224)
i = 0
img_list = []
for root, _, files in os.walk("pics"):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', ".JPEG")):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            img_list.append(image)
            image = data_transform(image)
            input[i] = image
            i += 1


output = model(input)
print(output)
for idx,i in enumerate(img_list):
    i.save(f"{idx}/pic.png")
# val_loss, val_acc = evaluate(model=model,
#                              data_loader=val_loader,
#                              device="cuda",
#                              epoch=0)
