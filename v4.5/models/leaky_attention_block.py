import os

import torch
import torch.nn as nn
import torchvision

from models.conv_layer import InvertedResidual


def save(img):
    img = img.to(torch.float32)
    for idx,i in enumerate(img):
        if not os.path.exists(str(idx)):
            os.makedirs(str(idx))
        suffix = 1
        name = f"0/{suffix}-{i.shape[-1]}.png"
        while os.path.exists(name):
            suffix +=1
            name = f"0/{suffix}-{i.shape[-1]}.png"
        torchvision.utils.save_image(i, name)

class DropAttention(nn.Module):
    def __init__(self, in_channels, num_heads,height_width,num_encoder=1,expand_ratio=2):
        super(DropAttention, self).__init__()
        self.trans =nn.Sequential(*[nn.TransformerEncoderLayer(d_model=in_channels,nhead=num_heads,dim_feedforward=expand_ratio * in_channels,batch_first=True) for _ in range(num_encoder)])

        self.dict = {
            110:512 + 256,
            55:256,
            28:128,
        }
        self.token_num = self.dict[height_width]
        self.h_w = height_width
        self.sigmoid = nn.Sigmoid()
        self.position = self.create_1d_sinusoidal_position_encoding(height_width ** 2, in_channels)

    def create_1d_sinusoidal_position_encoding(self,seq_length, dim):
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_encoding = torch.zeros(seq_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.to("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_mask(self, x, a):
        # x should be a tensor of shape [batch_num, channels, n, n]
        batch_num, _, n, _ = x.shape  # channels are ignored
        bool_tensor = torch.zeros((batch_num, n * n), dtype=torch.bool)

        # 生成 n*n 的随机排列，并选择前 a 个不重复的随机索引
        indices = torch.randperm(n * n)[:a].unsqueeze(0).expand(batch_num, -1)

        # 将这些索引在每个 batch 上设置为 True
        bool_tensor.scatter_(1, indices, True)

        # Reshape the result back to [batch_num, n, n]
        random_bool_tensor = bool_tensor.view(batch_num, n, n)

        return random_bool_tensor

    def select_pixels_to_attn_with_pos(self,mask,feature_map,token_num,token_dim,batch_size,indices):
        # 使用mask获取True像素的索引
        # 获取mask中True像素对应的feature map像素
        selected_pixels = feature_map[indices[:, 0], :, indices[:, 1], indices[:, 2]]

        # 重新组织选取的像素，形状为[B, K, C]
        selected_pixels = selected_pixels.view(batch_size, token_num, token_dim)

        selected_position_encoding = self.position[indices[:, 1] * self.h_w + indices[:, 2]]
        selected_position_encoding = selected_position_encoding.view(batch_size, token_num, token_dim)

        return selected_pixels+selected_position_encoding

    def forward(self, x):
        token_dim = x.shape[1]
        token_num = self.token_num
        batch_size = x.shape[0]
        mask_map = self.normalize_mask(x, token_num)
        indices = torch.nonzero(mask_map)
        tokens = self.select_pixels_to_attn_with_pos(mask=mask_map,feature_map=x,token_num=token_num,token_dim=token_dim,batch_size=batch_size,indices=indices)
        tokens = self.trans(tokens)
        x[indices[:, 0], :, indices[:, 1], indices[:, 2]] = tokens.view(token_num * batch_size, token_dim)
        return x

class LeakyAttention(nn.Module):
    def __init__(self, in_channels, num_heads,height_width,num_encoder=1,expand_ratio=2):
        super(LeakyAttention, self).__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=1, stride=1, kernel_size=7)
        self.trans =nn.Sequential(*[nn.TransformerEncoderLayer(d_model=in_channels,nhead=num_heads,dim_feedforward=expand_ratio * in_channels,batch_first=True) for _ in range(num_encoder)])

        self.dict = {
            110:512 + 256,
            55:256,
            28:128,
        }
        self.token_num = self.dict[height_width]
        self.h_w = height_width
        self.sigmoid = nn.Sigmoid()
        self.position = self.create_1d_sinusoidal_position_encoding(height_width ** 2, in_channels)

    def create_1d_sinusoidal_position_encoding(self,seq_length, dim):
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_encoding = torch.zeros(seq_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.to("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_mask(self, x, a):
        # assert isinstance(a, int) and a > 0, "a should be a positive integer"
        batch_num, _, n, n = x.shape

        feature_map = self.sigmoid(self.conv(x))
        # 创建一个全为False的布尔张量，用于保存是否为topk值的信息
        topk_mask = torch.zeros(batch_num, n, n, dtype=torch.bool, device=x.device)

        # 获取前a个最大值的索引（在批次维度上操作）
        _, top_indices = torch.topk(feature_map[:, 0].view(batch_num, -1), a, dim=1)

        # 计算二维索引
        rows = torch.div(top_indices, n, rounding_mode='floor')
        cols = top_indices % n

        # 使用一个批次维度的索引张量
        batch_indices = torch.arange(batch_num, device=x.device).view(-1, 1).expand_as(rows)

        # 将对应的topk_mask位置设置为True
        topk_mask[batch_indices, rows, cols] = True
        # if not self.training:
        #     save(topk_mask)
        return topk_mask, feature_map

    def select_pixels_to_attn_with_pos(self,mask,feature_map,token_num,token_dim,batch_size,indices):
        # 使用mask获取True像素的索引


        # 获取mask中True像素对应的feature map像素
        selected_pixels = feature_map[indices[:, 0], :, indices[:, 1], indices[:, 2]]

        # 重新组织选取的像素，形状为[B, K, C]
        selected_pixels = selected_pixels.view(batch_size, token_num, token_dim)

        selected_position_encoding = self.position[indices[:, 1] * self.h_w + indices[:, 2]]
        selected_position_encoding = selected_position_encoding.view(batch_size, token_num, token_dim)

        return selected_pixels+selected_position_encoding

    def forward(self, x):
        token_dim = x.shape[1]
        token_num = self.token_num
        batch_size = x.shape[0]
        mask_map, importance_map = self.normalize_mask(x, token_num)
        x = x * importance_map
        indices = torch.nonzero(mask_map)
        tokens = self.select_pixels_to_attn_with_pos(mask=mask_map,feature_map=x,token_num=token_num,token_dim=token_dim,batch_size=batch_size,indices=indices)
        tokens = self.trans(tokens)
        x[indices[:, 0], :, indices[:, 1], indices[:, 2]] = tokens.view(token_num * batch_size, token_dim)
        return x

class ConvBlock(nn.Module):
    """
    普通卷积块,在前几层用在
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride,padding=15,dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class LeakyConvFormerBlock(nn.Module):
    """
    前几层的attention
    """

    def __init__(self, in_channels, out_channels, num_heads,stride,height_width,expand_ratio=2,num_encoder=1):
        super(LeakyConvFormerBlock, self).__init__()
        self.leaky_attention = LeakyAttention(in_channels, num_heads,height_width,num_encoder,expand_ratio)
        self.conv_block = InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           expand_ratio=expand_ratio)

    def forward(self, x):
        x = self.leaky_attention(x)
        x = self.conv_block(x)
        return x


class DropOutBlock(nn.Module):
    """
    前几层的attention
    """

    def __init__(self, in_channels, out_channels, num_heads,stride,height_width,expand_ratio=2,num_encoder=1):
        super(DropOutBlock, self).__init__()
        self.leaky_attention = DropAttention(in_channels, num_heads,height_width,num_encoder,expand_ratio)
        self.conv_block = InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           expand_ratio=expand_ratio)

    def forward(self, x):
        x = self.leaky_attention(x)
        x = self.conv_block(x)
        return x
