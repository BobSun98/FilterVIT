import torch
import torch.nn as nn
from models.transfomer_conv_block import ConvFormerBlock, FirstConvFormerBlock
from models.leaky_attention_block import LeakyConvFormerBlock, DropOutBlock
from models.conv_layer import InvertedResidual, ConvBlock
from .utils import transfer_att_shape_to_featuremap, transfer_featuremap_to_att_shape

class ConvFormer_xxs(nn.Module):
    def __init__(self,
                 num_calsses=1000,
                 batch_size=8,
                 ):
        super(ConvFormer_xxs, self).__init__()
        self.first_conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, stride=2, kernel_size=5, num_group=1),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=2),
        )  # in 224 out 110)
        self.first_ConvFormer = FirstConvFormerBlock(num_in=64, num_out=64, kernel_size=3, num_heads=2, height_width=20)
        self.leaky_att_network = nn.Sequential(
            # LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=2,stride = 1,height_width = 110),  # in 110 , out 53
            LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=2, stride=1, height_width=55),
            # in 110 , out 53
            # LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=4 ,stride=1,height_width = 55),  # in 53 out 53
            LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=4, stride=2, height_width=55),
            # in 53 out 25
            # FirstConvFormerBlock(num_in=64, num_out=64, kernel_size=3, num_heads=2, height_width=14),
            LeakyConvFormerBlock(in_channels=64, out_channels=128, num_heads=4, stride=1, height_width=28),
            # in 53 out 25
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=28),
            # in 53 out 25
            # LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2, height_width = 28),  # in 53 out 25
        )

        self.network = nn.Sequential(
            ConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=8),  # in25 out11
            ConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=8),  # in25 out11,  # in11 out4
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(128, 384),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(384, num_calsses),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(110)
        # 权重初始化 30:48

    # def pre_process(self, x):  # 加位置编码 加cls_token
    #      # [batch_num,128, 110, 110]
    #     # x = transfer_featuremap_to_att_shape(x)
    #
    #     # x = x + self.position_embed  # position embedding
    #     # x = transfer_att_shape_to_featuremap(x)
    #     return x

    def post_process(self, x):
        x = self.avg(x)
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x

    def forward(self, x):
        # 位置编码,分类logit
        x = self.first_conv(x)
        x = self.first_ConvFormer(x)
        x = self.leaky_att_network(x)
        # x = transfer_featuremap_to_att_shape(x)
        x = self.network(x)
        x = self.post_process(x)
        return x


class DropOutVIT_xs(nn.Module):
    def __init__(self,
                 num_calsses=1000,
                 batch_size=8,
                 ):
        super(DropOutVIT_xs, self).__init__()
        self.first_conv = nn.Sequential(ConvBlock(in_channels=3, out_channels=64, kernel_size=5, stride=2, num_group=1),
                                        InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=3),
                                        nn.AdaptiveAvgPool2d((110, 110))
                                        )  # in 224 out 110)
        self.leaky_att_network = nn.Sequential(
            # LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=2,stride = 2,height_width =110),  # in 110 , out 53
            FirstConvFormerBlock(num_in=64, num_out=128, kernel_size=3, num_heads=2, height_width=11),
            DropOutBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=55,
                                 expand_ratio=3),  # in 53 out 53
            # LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2,height_width =55,expand_ratio=3),  # in 53 out 25
            FirstConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=2, height_width=11),
            DropOutBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=28,
                                 expand_ratio=3),  # in 53 out 25
            DropOutBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=28,
                                 expand_ratio=3),  # in 53 out 25
            FirstConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=2, height_width=7),
            # LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2,height_width =28),  # in 53 out 25
        )

        self.network = nn.Sequential(
            ConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=8),  # in25 out11
            ConvFormerBlock(num_in=128, num_out=256, kernel_size=3, num_heads=8),  # in25 out11,  # in11 out4
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(256, 384),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(384, num_calsses),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 权重初始化 30:48

    def post_process(self, x):
        x = self.avg(x)
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x

    def forward(self, x):
        # 位置编码,分类logit
        x = self.first_conv(x)
        x = self.leaky_att_network(x)
        x = self.network(x)
        x = self.post_process(x)
        return x

class ConvFormer_xs(nn.Module):
    def __init__(self,
                 num_calsses=1000,
                 batch_size=8,
                 ):
        super(ConvFormer_xs, self).__init__()
        self.first_conv = nn.Sequential(ConvBlock(in_channels=3, out_channels=64, kernel_size=5, stride=2, num_group=1),
                                        InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=3),
                                        nn.AdaptiveAvgPool2d((110, 110))
                                        )  # in 224 out 110)
        self.leaky_att_network = nn.Sequential(
            # LeakyConvFormerBlock(in_channels=64, out_channels=64, num_heads=2,stride = 2,height_width =110),  # in 110 , out 53
            FirstConvFormerBlock(num_in=64, num_out=128, kernel_size=3, num_heads=2, height_width=11),
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=55,
                                 expand_ratio=3),  # in 53 out 53
            # LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2,height_width =55,expand_ratio=3),  # in 53 out 25
            FirstConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=2, height_width=11),
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=28,
                                 expand_ratio=3),  # in 53 out 25
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4, stride=1, height_width=28,
                                 expand_ratio=3),  # in 53 out 25
            FirstConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=2, height_width=7),
            # LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2,height_width =28),  # in 53 out 25
        )

        self.network = nn.Sequential(
            ConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=8),  # in25 out11
            ConvFormerBlock(num_in=128, num_out=256, kernel_size=3, num_heads=8),  # in25 out11,  # in11 out4
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(256, 384),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(384, num_calsses),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 权重初始化 30:48

    def post_process(self, x):
        x = self.avg(x)
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x

    def forward(self, x):
        # 位置编码,分类logit
        x = self.first_conv(x)
        x = self.leaky_att_network(x)
        x = self.network(x)
        x = self.post_process(x)
        return x


class ConvFormer_s(nn.Module):
    def __init__(self,
                 num_calsses=1000,
                 batch_size=8,
                 ):
        super(ConvFormer_s, self).__init__()
        self.first_conv = nn.Sequential(ConvBlock(in_channels=3, out_channels=128, kernel_size=5, stride=2,num_group=1),
                                        InvertedResidual(in_channels=128,out_channels=128,stride=1,expand_ratio=3)
                                        )  # in 224 out 110)
        self.leaky_att_network = nn.Sequential(
            FirstConvFormerBlock(num_in=128,num_out=128,kernel_size=3,num_heads=2,height_width=16,num_encoder=2),
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=1,height_width =55,expand_ratio=2,num_encoder=2),  # in 53 out 53
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=2,height_width =55,expand_ratio=3,num_encoder=2),  # in 53 out 53
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=4 ,stride=1,height_width =28,expand_ratio=3,num_encoder=2),  # in 53 out 25
            LeakyConvFormerBlock(in_channels=128, out_channels=256, num_heads=4 ,stride=2,height_width =28,expand_ratio=3,num_encoder=2),  # in 53 out 25

        )

        self.network = nn.Sequential(
            ConvFormerBlock(num_in=256, num_out=256, kernel_size=3, num_heads=8,expand_ratio=2,num_encoder=2),  # in25 out11
            ConvFormerBlock(num_in=256, num_out=256, kernel_size=3, num_heads=8,expand_ratio=2,num_encoder=2),  # in25 out11,  # in11 out4
        )


        self.mlp_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(512,num_calsses),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 权重初始化 30:48

    def post_process(self, x):
        x = self.avg(x)
        x = x.flatten(1)
        x = self.mlp_head(x)
        return x

    def forward(self, x):
        # 位置编码,分类logit
        x = self.first_conv(x)
        x = self.leaky_att_network(x)
        # x = transfer_featuremap_to_att_shape(x)
        x = self.network(x)
        x = self.post_process(x)
        return x


class ConvFormer_s2(nn.Module):
    def __init__(self,num_calsses=1000,):
        super(ConvFormer_s2, self).__init__()
        self.first_conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=128, kernel_size=5, stride=2, num_group=1),
            InvertedResidual(in_channels=128, out_channels=128, stride=1, expand_ratio=3)
            )  # in 224 out 110)
        self.leaky_att_network = nn.Sequential(
            FirstConvFormerBlock(num_in=128, num_out=128, kernel_size=3, num_heads=2, height_width=20, num_encoder=2),
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=8, stride=1, height_width=55,
                                 expand_ratio=2, num_encoder=2),  # in 53 out 53
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=8, stride=2, height_width=55,
                                 expand_ratio=2, num_encoder=2),  # in 53 out 53
            LeakyConvFormerBlock(in_channels=128, out_channels=128, num_heads=8, stride=2, height_width=28,
                                 expand_ratio=2, num_encoder=2),  # in 53 out 25
            LeakyConvFormerBlock(in_channels=128, out_channels=256, num_heads=8, stride=2, height_width=28,
                                 expand_ratio=2, num_encoder=2),

        )

        self.network = nn.Sequential(
            ConvFormerBlock(num_in=256, num_out=256, kernel_size=3, num_heads=16, expand_ratio=2, num_encoder=2,reduce=4,stride=1),
            ConvFormerBlock(num_in=256, num_out=256, kernel_size=3, num_heads=16, expand_ratio=2, num_encoder=2,reduce=2),
            ConvFormerBlock(num_in=256, num_out=512, kernel_size=3, num_heads=16, expand_ratio=2, num_encoder=2,reduce=8, stride=1),
            ConvFormerBlock(num_in=512, num_out=512, kernel_size=3, num_heads=16, expand_ratio=2, num_encoder=2,reduce=4),
            # in25 out11,  # in11 out4
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_calsses),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 权重初始化 30:48

    def post_process(self, x):
        x = self.avg(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        # 位置编码,分类logit
        x = self.first_conv(x)
        x = self.leaky_att_network(x)
        # x = transfer_featuremap_to_att_shape(x)
        x = self.network(x)
        x = self.post_process(x)
        return x
