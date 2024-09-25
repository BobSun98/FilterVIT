import torch
import torch.nn as nn
from functools import partial

from .attention import TransBlock
from .utils import transfer_att_shape_to_featuremap, transfer_featuremap_to_att_shape
from .conv_layer import InvertedResidual


class ConvFormerBlock(nn.Module):
    """
    输入是x+class_token
    是直接可以attention的格式
    输出也是直接可以attention的格式
    """

    def __init__(self, num_in, num_out, kernel_size, num_heads, expand_ratio=2, num_encoder=1, reduce=1,stride =2):
        super(ConvFormerBlock, self).__init__()

        # self.trans = nn.Sequential(
        #     *[TransBlock(dim=num_in, num_heads=num_heads, mlp_ratio=2, qkv_bias=True, qk_scale=None,
        #                  drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0,
        #                  norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU) for _ in range(num_encoder)])
        self.reduce = reduce
        self.trans = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=num_in // reduce, nhead=num_heads,
                                         dim_feedforward=expand_ratio * num_in // reduce,
                                         batch_first=True) for _ in range(num_encoder)])
        self.conv = InvertedResidual(in_channels=num_in, out_channels=num_out, stride=stride, expand_ratio=expand_ratio)

        self.reduce = nn.Identity() if reduce == 1 else InvertedResidual(in_channels=num_in, out_channels=num_in // reduce,
                                                                       stride=1, expand_ratio=1)
        self.expand = nn.Identity() if reduce == 1 else InvertedResidual(in_channels=num_in // reduce, out_channels=num_in,
                                                                       stride=1, expand_ratio=1)

    def forward(self, x):  # 进来的时候是直接可以attention的格式
        if self.reduce != 1:
            y = x.clone()
        x = self.reduce(x)
        x = transfer_featuremap_to_att_shape(x)  # 转化为attention形式
        x = self.trans(x)  # attrntion]  # 取出class token
        x = transfer_att_shape_to_featuremap(x)  # 转化为feature map
        x = self.expand(x)
        if self.reduce != 1:
            x = y+x
        x = self.conv(x)  # x[:, 1:] 就是要把 cls_token取掉
        return x  # 出去的时候也是直接可以attention的格式


class FirstConvFormerBlock(nn.Module):
    """
    输入是x+class_token
    是直接可以attention的格式
    输出也是直接可以attention的格式
    """

    def __init__(self, num_in, num_out, kernel_size, num_heads, height_width, num_encoder=1, expand_ratio=2):
        super(FirstConvFormerBlock, self).__init__()

        # self.trans = nn.Sequential(
        #     *[TransBlock(dim=num_in, num_heads=num_heads, mlp_ratio=2, qkv_bias=True, qk_scale=None,
        #                  drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0,
        #                  norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU) for _ in range(num_encoder)])
        self.trans = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=num_in, nhead=num_heads, dim_feedforward=expand_ratio * num_in,
                                         batch_first=True) for _ in range(num_encoder)])
        self.conv = InvertedResidual(in_channels=num_in, out_channels=num_out, stride=2, expand_ratio=expand_ratio)
        self.avg = nn.AdaptiveAvgPool2d(height_width)
        self.position = self.create_1d_sinusoidal_position_encoding(height_width ** 2, num_in)

    def create_1d_sinusoidal_position_encoding(self, seq_length, dim):
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_encoding = torch.zeros(seq_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        y = self.avg(x)
        y = transfer_featuremap_to_att_shape(y) + self.position
        y = self.trans(y)
        y = transfer_att_shape_to_featuremap(y)  # 转化为feature map
        y = nn.AdaptiveAvgPool2d(x.shape[-1])(y)
        x = x + y
        x = self.conv(x)
        return x
