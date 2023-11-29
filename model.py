import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = 1-attn
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class FCN16s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN16s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=False)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=False)  # 1/4

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        return h


class FeverTransformer(nn.Module):
    def __init__(self, args):
        super(FeverTransformer, self).__init__()
        self.args = args
        # self.ln_pre = nn.LayerNorm([256, 256])

        self.InfraredNet = FCN16s()

        if self.args.Mode == 'Base':
            self.head = nn.Sequential(
                        nn.Linear(4096, args.num_classes),
                        )

        elif self.args.Mode == 'FAM':
            self.to_patch_embedding = nn.Sequential(
                                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size),
                                        nn.Linear(args.channels * args.patch_size * args.patch_size, args.dim),
                                        )
            self.attention          = Attention(args.dim, heads = args.heads, dim_head = args.dim_head, dropout = args.dropout)
            self.reduction          = nn.Sequential(
                                        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = args.pool_size, w = args.pool_size, p1=args.patch_size, p2=args.patch_size),
                                        )
            self.head               = nn.Sequential(
                                        nn.Linear(4096, args.num_classes),
                                        )

        elif self.args.Mode == 'DCB':
            self.depth_branch = nn.Sequential(
                                nn.Conv2d(1, 64, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.seg_branch = nn.Sequential(
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.head       = nn.Sequential(
                                nn.Linear(4096, 4096),
                                nn.ReLU(), 
                                nn.Dropout(0.1),
                                nn.Linear(4096, args.num_classes),
                                )

        elif self.args.Mode == 'EDCL':
            self.ln = nn.LayerNorm([64, 64])
            self.head = nn.Sequential(
                        nn.Linear(4096, args.num_classes),
                        )


        elif self.args.Mode == 'FAM+DCB':
            self.depth_branch = nn.Sequential(
                                nn.Conv2d(1, 64, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.seg_branch = nn.Sequential(
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.to_patch_embedding = nn.Sequential(
                                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size),
                                        nn.Linear(args.channels * args.patch_size * args.patch_size, args.dim),
                                        )
            self.attention          = Attention(args.dim, heads = args.heads, dim_head = args.dim_head, dropout = args.dropout)
            self.reduction          = nn.Sequential(
                                        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = args.pool_size, w = args.pool_size, p1=args.patch_size, p2=args.patch_size),
                                        )
            self.head               = nn.Sequential(
                                        nn.Linear(4096, args.num_classes),
                                        )

        elif self.args.Mode == 'FAM+EDCL':
            self.ln = nn.LayerNorm([64, 64])
            self.to_patch_embedding = nn.Sequential(
                                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size),
                                        nn.Linear(args.channels * args.patch_size * args.patch_size, args.dim),
                                        )
            self.attention          = Attention(args.dim, heads = args.heads, dim_head = args.dim_head, dropout = args.dropout)
            self.reduction          = nn.Sequential(
                                        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = args.pool_size, w = args.pool_size, p1=args.patch_size, p2=args.patch_size),
                                        )
            self.head               = nn.Sequential(
                                        nn.Linear(4096, args.num_classes),
                                        )

        elif self.args.Mode == 'DCB+EDCL':
            self.depth_branch = nn.Sequential(
                                nn.Conv2d(1, 64, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.seg_branch = nn.Sequential(
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.ln = nn.LayerNorm([64, 64])
            self.head       = nn.Sequential(
                                nn.Linear(4096, 4096),
                                nn.ReLU(), 
                                nn.Dropout(0.1),
                                nn.Linear(4096, args.num_classes),
                                )

        elif self.args.Mode == 'FAM+DCB+EDCL':
            self.depth_branch = nn.Sequential(
                                nn.Conv2d(1, 64, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.seg_branch = nn.Sequential(
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.ln = nn.LayerNorm([64, 64])
            self.to_patch_embedding = nn.Sequential(
                                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size),
                                        nn.Linear(args.channels * args.patch_size * args.patch_size, args.dim),
                                        )
            self.attention          = Attention(args.dim, heads = args.heads, dim_head = args.dim_head, dropout = args.dropout)
            self.reduction          = nn.Sequential(
                                        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = args.pool_size, w = args.pool_size, p1=args.patch_size, p2=args.patch_size),
                                        )
            self.head               = nn.Sequential(
                                        nn.Linear(4096, args.num_classes),
                                        )
            
        elif self.args.Mode == 'FAM+DCB+EDCL-CTMap':
            self.depth_branch = nn.Sequential(
                                nn.Conv2d(1, 64, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.seg_branch = nn.Sequential(
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True),
                                )
            self.ln = nn.LayerNorm([64, 64])
            self.to_patch_embedding = nn.Sequential(
                                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size),
                                        nn.Linear(args.channels * args.patch_size * args.patch_size, args.dim),
                                        )
            self.attention          = Attention(args.dim, heads = args.heads, dim_head = args.dim_head, dropout = args.dropout)
            self.reduction          = nn.Sequential(
                                        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = args.pool_size, w = args.pool_size, p1=args.patch_size, p2=args.patch_size),
                                        )
            self.head               = nn.Sequential(
                                        nn.Linear(4096, args.num_classes),
                                        )


        else:
            assert 'Mode Type Error'

        
        
    def forward(self, x, dp, dv):
        # x ==> b, 1, 256, 256
        # x = self.ln_pre(x)
        if self.args.Mode == 'Base':

            # InfraredNet
            x = self.InfraredNet(x) 
            feature = x.clone()

            x = x.mean(dim=1, keepdim=True) 
            heat = x.clone()

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode == 'FAM':

            # InfraredNet
            x = self.InfraredNet(x) 
            feature = x.clone()

            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            heat = x.clone()

            # FAM
            x = self.to_patch_embedding(x)
            x = self.attention(x)+x
            x = self.reduction(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif  self.args.Mode == 'DCB':

            # InfraredNet
            x = self.InfraredNet(x)
            feature = x.clone()

            # DCB
            dp = self.depth_branch(dp)
            x = torch.cat([x, dp], dim=1)
            x = self.seg_branch(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            heat = x.clone()

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode == 'EDCL':

            # InfraredCNN
            x = self.InfraredNet(x) 
            feature = x.clone()

            x = x.mean(dim=1, keepdim=True) 
            x = self.ln(x)
            heat = x.clone()

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode in 'FAM+DCB':

            # InfraredNet
            x = self.InfraredNet(x)  #b, 128, 64, 64
            feature = x.clone()

            # DCB
            dp = self.depth_branch(dp)
            x = torch.cat([x, dp], dim=1)
            x = self.seg_branch(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            heat = x.clone()

            # FAM
            x = self.to_patch_embedding(x)
            x = self.attention(x)+x
            x = self.reduction(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            
            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode == 'FAM+EDCL':

            # InfraredNet
            x = self.InfraredNet(x) 
            feature = x.clone()

            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            x = self.ln(x)
            heat = x.clone()

            # FAM
            x = self.to_patch_embedding(x)
            x = self.attention(x)+x
            x = self.reduction(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif  self.args.Mode == 'DCB+EDCL':

            # InfraredNet
            x = self.InfraredNet(x)
            feature = x.clone()

            # DCB
            dp = self.depth_branch(dp)
            x = torch.cat([x, dp], dim=1)
            x = self.seg_branch(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            x = self.ln(x)
            heat = x.clone()

            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode == 'FAM+DCB+EDCL':

            # InfraredNet
            x = self.InfraredNet(x)  #b, 128, 64, 64
            feature = x.clone()

            # DCB
            dp = self.depth_branch(dp)
            x = torch.cat([x, dp], dim=1)
            x = self.seg_branch(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            x = self.ln(x)
            heat = x.clone()

            # FAM
            x = self.to_patch_embedding(x)
            x = self.attention(x)+x
            x = self.reduction(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            
            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature

        elif self.args.Mode == 'FAM+DCB+EDCL-CTMap':

            # InfraredNet
            x = self.InfraredNet(x)  #b, 128, 64, 64
            feature = x.clone()

            # DCB
            dp = self.depth_branch(dp)
            x = torch.cat([x, dp], dim=1)
            x = self.seg_branch(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            x = self.ln(x)
            heat = x.clone()

            # FAM
            x = self.to_patch_embedding(x)
            x = self.attention(x)+x
            x = self.reduction(x)
            x = x.mean(dim=1, keepdim=True) #b, 1, 64, 64
            
            # Head
            x = x.view(x.shape[0], -1)
            pre = self.head(x)
            return pre, heat, feature