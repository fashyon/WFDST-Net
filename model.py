import torch
from torch import nn
import torch.nn.functional as F
import settings
from timm.models.layers import DropPath, to_2tuple
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    ZeroPad2d = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    img = ZeroPad2d(input_data)
    col = torch.zeros([N, C, filter_h,filter_w, out_h,out_w]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride,]

    col = col.reshape(N, C, filter_h*filter_w, out_h*out_w)

    return col


def col2im(col,orisize,filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = orisize
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, filter_h, filter_w,out_h, out_w)
    img = torch.zeros([N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class inlwt_catone(nn.Module):
    def __init__(self):
        super(inlwt_catone, self).__init__()
        self.requires_grad = False
        self.P2mulU2 = np.array([[1.3968, -0.2212, -0.5412, 1.3066],
                            [0.2212, 1.3968, -1.3066, -0.5412],
                            [-0.2212, -1.3968, -1.3066, -0.5412],
                            [-1.3968, 0.2212, -0.5412, 1.3066]])
        self.P1mulU1div4 = np.array([[0.2166, -0.0256, 0.1213, 0.0144],
                                [0.0256, 0.2166, 0.0144, -0.1213],
                                [0.1213, -0.0144, -0.2166, -0.0256],
                                [-0.0144, -0.1213, 0.0256, -0.2166]])
        self.P2mulU2 = torch.cuda.FloatTensor(self.P2mulU2).unsqueeze(0).unsqueeze(0)
        self.P1mulU1div4 = torch.cuda.FloatTensor(self.P1mulU1div4).unsqueeze(0).unsqueeze(0)

    def forward(self, decoder_one,orisize):
        out_channel = orisize[1]

        A = decoder_one[:, 0:out_channel, :, :]
        B = decoder_one[:, out_channel:out_channel * 2, :, :]
        C = decoder_one[:, out_channel * 2:out_channel * 3, :, :]
        D = decoder_one[:, out_channel * 3:out_channel * 4, :, :]

        b, c, h1, w1 = A.size()
        A = A.reshape(b, c, 1, h1 * w1);
        B = B.reshape(b, c, 1, h1 * w1);
        C = C.reshape(b, c, 1, h1 * w1);
        D = D.reshape(b, c, 1, h1 * w1);

        Y1 = torch.cat([A, B, C, D], dim=2)
        Y2 = self.P2mulU2 @ Y1;
        t2 = Y2[:, :, 1, :].reshape(b, c, h1, w1);
        t3 = Y2[:, :, 2, :].reshape(b, c, h1, w1);
        t4 = Y2[:, :, 3, :].reshape(b, c, h1, w1);

        t22 = torch.roll(t2, shifts=1, dims=2)
        t32 = torch.roll(t3, shifts=1, dims=3)
        t42 = torch.roll(t4, shifts=(1, 1), dims=(2, 3))

        Y2[:, :, 1, :] = t22.flatten(2)
        Y2[:, :, 2, :] = t32.flatten(2)
        Y2[:, :, 3, :] = t42.flatten(2)

        Y3 = self.P1mulU1div4 @ Y2;
        rst = col2im(Y3, orisize, 2, 2, stride=2, pad=0);

        return rst

class nlwt_catone(nn.Module):
    def __init__(self):
        super(nlwt_catone, self).__init__()
        self.requires_grad = False
        self.U1ImulP1I = np.array([[0.8664, 0.1026, 0.4852, -0.0574],
                              [-0.1026, 0.8664, -0.0574, -0.4852],
                              [0.4852, 0.0574, -0.8664, 0.1026],
                              [0.0574, -0.4852, -0.1026, -0.8664]])
        self.U2ImulP2Imul4 = np.array([[1.3968, 0.2212, -0.2212, -1.3968],
                                  [-0.2212, 1.3968, -1.3968, 0.2212],
                                  [-0.5412, -1.3066, -1.3066, -0.5412],
                                  [1.3066, -0.5412, -0.5412, 1.3066]])
        self.U1ImulP1I = torch.cuda.FloatTensor(self.U1ImulP1I).unsqueeze(0).unsqueeze(0)
        self.U2ImulP2Imul4 = torch.cuda.FloatTensor(self.U2ImulP2Imul4).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.size()
        orisize = x.size()

        xT_col = im2col(x, 2, 2, stride=2, pad=0);
        x1 = self.U1ImulP1I @ xT_col;

        h1 = h // 2
        w1 = w // 2
        T2 = x1[:, :, 1, :].reshape(b, c, h1, w1);
        T3 = x1[:, :, 2, :].reshape(b, c, h1, w1);
        T4 = x1[:, :, 3, :].reshape(b, c, h1, w1);

        T22 = torch.roll(T2, shifts=-1, dims=2)
        T32 = torch.roll(T3, shifts=-1, dims=3)
        T42 = torch.roll(T4, shifts=(-1, -1), dims=(2, 3))

        x1[:, :, 1, :] = T22.flatten(2);
        x1[:, :, 2, :] = T32.flatten(2);
        x1[:, :, 3, :] = T42.flatten(2);

        x2 = self.U2ImulP2Imul4 @ x1;

        A = x2[:, :, 0, :].reshape(b, c, h1, w1);
        B = x2[:, :, 1, :].reshape(b, c, h1, w1);
        C = x2[:, :, 2, :].reshape(b, c, h1, w1);
        D = x2[:, :, 3, :].reshape(b, c, h1, w1);
        out_catone = torch.cat([A, B, C, D], dim=1)

        return out_catone, orisize

class nlwt(nn.Module):
    def __init__(self):
        super(nlwt, self).__init__()
        self.requires_grad = False
        self.U1ImulP1I = np.array([[0.8664, 0.1026, 0.4852, -0.0574],
                              [-0.1026, 0.8664, -0.0574, -0.4852],
                              [0.4852, 0.0574, -0.8664, 0.1026],
                              [0.0574, -0.4852, -0.1026, -0.8664]])
        self.U2ImulP2Imul4 = np.array([[1.3968, 0.2212, -0.2212, -1.3968],
                                  [-0.2212, 1.3968, -1.3968, 0.2212],
                                  [-0.5412, -1.3066, -1.3066, -0.5412],
                                  [1.3066, -0.5412, -0.5412, 1.3066]])
        self.U1ImulP1I = torch.cuda.FloatTensor(self.U1ImulP1I).unsqueeze(0).unsqueeze(0)
        self.U2ImulP2Imul4 = torch.cuda.FloatTensor(self.U2ImulP2Imul4).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.size()
        orisize = x.size()

        xT_col = im2col(x, 2, 2, stride=2, pad=0);
        x1 = self.U1ImulP1I @ xT_col;

        h1 = h // 2
        w1 = w // 2
        T2 = x1[:, :, 1, :].reshape(b, c, h1, w1);
        T3 = x1[:, :, 2, :].reshape(b, c, h1, w1);
        T4 = x1[:, :, 3, :].reshape(b, c, h1, w1);

        T22 = torch.roll(T2, shifts=-1, dims=2)
        T32 = torch.roll(T3, shifts=-1, dims=3)
        T42 = torch.roll(T4, shifts=(-1, -1), dims=(2, 3))

        x1[:, :, 1, :] = T22.flatten(2);
        x1[:, :, 2, :] = T32.flatten(2);
        x1[:, :, 3, :] = T42.flatten(2);

        x2 = self.U2ImulP2Imul4 @ x1;

        A = x2[:, :, 0, :].reshape(b, c, h1, w1);
        B = x2[:, :, 1, :].reshape(b, c, h1, w1);
        C = x2[:, :, 2, :].reshape(b, c, h1, w1);
        D = x2[:, :, 3, :].reshape(b, c, h1, w1);

        return A, B, C, D, orisize

class ilwtbA(nn.Module):
    def __init__(self):
        super(ilwtbA, self).__init__()
        self.requires_grad = False
        self.P2mulU2 = np.array([[1.3968, -0.2212, -0.5412, 1.3066],
                            [0.2212, 1.3968, -1.3066, -0.5412],
                            [-0.2212, -1.3968, -1.3066, -0.5412],
                            [-1.3968, 0.2212, -0.5412, 1.3066]])
        self.P1mulU1div4 = np.array([[0.2166, -0.0256, 0.1213, 0.0144],
                                [0.0256, 0.2166, 0.0144, -0.1213],
                                [0.1213, -0.0144, -0.2166, -0.0256],
                                [-0.0144, -0.1213, 0.0256, -0.2166]])
        self.P2mulU2 = torch.cuda.FloatTensor(self.P2mulU2).unsqueeze(0).unsqueeze(0)
        self.P1mulU1div4 = torch.cuda.FloatTensor(self.P1mulU1div4).unsqueeze(0).unsqueeze(0)

    def forward(self, A,B,C,D,orisize):
        b, c, h1, w1 = A.size()
        A = A.reshape(b, c, 1, h1 * w1);
        B = B.reshape(b, c, 1, h1 * w1);
        C = C.reshape(b, c, 1, h1 * w1);
        D = D.reshape(b, c, 1, h1 * w1);

        Y1 = torch.cat([A, B, C, D], dim=2)
        Y2 = self.P2mulU2 @ Y1;
        t2 = Y2[:, :, 1, :].reshape(b, c, h1, w1);
        t3 = Y2[:, :, 2, :].reshape(b, c, h1, w1);
        t4 = Y2[:, :, 3, :].reshape(b, c, h1, w1);

        t22 = torch.roll(t2, shifts=1, dims=2)
        t32 = torch.roll(t3, shifts=1, dims=3)
        t42 = torch.roll(t4, shifts=(1, 1), dims=(2, 3))

        Y2[:, :, 1, :] = t22.flatten(2)
        Y2[:, :, 2, :] = t32.flatten(2)
        Y2[:, :, 3, :] = t42.flatten(2)

        Y3 = self.P1mulU1div4 @ Y2;
        rst = col2im(Y3, orisize, 2, 2, stride=2, pad=0);

        return rst


# class convd(nn.Module):
#     def __init__(self, inputchannel, outchannel, kernel_size, stride):
#         super(convd, self).__init__()
#         # self.relu = nn.ReLU()
#         self.padding = nn.ReflectionPad2d(kernel_size // 2)
#         self.conv = nn.Sequential(nn.Conv2d(inputchannel, outchannel, kernel_size, stride), nn.LeakyReLU(0.2))
#
#     def forward(self, x):
#         x = self.conv(self.padding(x))
#         return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WFDSA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """
    def __init__(self, dim, window_size, num_heads, low_attn_drop=0.,high_attn_drop=0., proj_drop=0.,):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.nlwt = nlwt()
        self.logit_scale_low = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.logit_scale_high = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv_low = nn.Linear(dim, dim * 2, bias=True)
        self.kv_high = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop_low = nn.Dropout(low_attn_drop)
        self.attn_drop_high = nn.Dropout(high_attn_drop)
        self.fuse_high = nn.Sequential(nn.Conv2d(dim * 3, dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.fuse_lh = nn.Linear(2*dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        x_ = x.transpose(1,2).view(B_,C,self.window_size[0],self.window_size[1])
        xl0,xh1,xh2,xh3,orisize = self.nlwt(x_)
        xl0 = xl0.flatten(2).transpose(1,2)
        kv_low = self.kv_low(xl0).reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_low,v_low = kv_low[0],kv_low[1]
        xh123 = self.fuse_high(torch.cat([xh1,xh2,xh3],dim=1)).flatten(2).transpose(1,2)
        kv_high = self.kv_high(xh123).reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_high, v_high = kv_high[0], kv_high[1]

        # cosine attention
        attn_low = (F.normalize(q, dim=-1) @ F.normalize(k_low, dim=-1).transpose(-2, -1))
        logit_scale_low = torch.clamp(self.logit_scale_low, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn_low = attn_low * logit_scale_low
        attn_low = self.softmax(attn_low)
        attn_low = self.attn_drop_low(attn_low)
        attn_low = (attn_low @ v_low).permute(0,2,3,1).reshape(-1, N, C )

        attn_high = (F.normalize(q, dim=-1) @ F.normalize(k_high, dim=-1).transpose(-2, -1))
        logit_scale_high = torch.clamp(self.logit_scale_high, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn_high = attn_high * logit_scale_high
        attn_high = self.softmax(attn_high)
        attn_high = self.attn_drop_high(attn_high)
        attn_high = (attn_high @ v_high).permute(0,2,3,1).reshape(-1, N, C )

        x = self.fuse_lh(torch.cat([attn_low, attn_high], dim=-1))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x,H,W):
        B, N, C = x.shape
        # H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)

        return x

class WFDST(nn.Module):
    r""" WFDST.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., low_attn_drop=0., high_attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),ConvPosEnc(dim=dim, k=3)])
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WFDSA(dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
             low_attn_drop=low_attn_drop,high_attn_drop=high_attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,H,W):
        shortcut=None
        if len(x.shape)==3:
            B,L,C = x.shape
            x = self.cpe[0](x,H,W)
            shortcut = x
            x = x.view(B, H, W, C).contiguous()
        elif len(x.shape) == 4:
            B,H,W,C = x.shape
            x = x.view(B, H*W, C).contiguous()
            x = self.cpe[0](x, H, W)
            shortcut = x
            x = x.view(B,H,W,C).contiguous()
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = self.cpe[1](x,H,W)
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.view(B, H, W, C).contiguous()

        return x

class WFDST_Net(nn.Module):
    def __init__(self, in_channel=3, channel=settings.channel):
        super().__init__()
        self.input_resolution = settings.patch_size
        self.window_size = settings.window_size
        self.depth = settings.depth
        self.heads = settings.heads
        self.convert = nn.Sequential(nn.Conv2d(in_channel, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WD_fuse1 = nn.Sequential(nn.Conv2d(channel * 4, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WD_fuse2 = nn.Sequential(nn.Conv2d(channel * 4, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WD_fuse3 = nn.Sequential(nn.Conv2d(channel * 4, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WD_fuse4 = nn.Sequential(nn.Conv2d(channel * 4, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WU_ifuse1 = nn.Sequential(nn.Conv2d(channel, channel * 4, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WU_ifuse2 = nn.Sequential(nn.Conv2d(channel, channel * 4, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WU_ifuse3 = nn.Sequential(nn.Conv2d(channel, channel * 4, 3, 1, 1), nn.LeakyReLU(0.2))
        self.WU_ifuse4 = nn.Sequential(nn.Conv2d(channel, channel * 4, 3, 1, 1), nn.LeakyReLU(0.2))
        self.cat1 = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.cat2 = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.cat3 = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.cat4 = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.out = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(channel, 3, 1, 1))
        self.drop_path_rate = 0.1
        self.num_enc_layers = [2, 2, 2, 2, 2, 2]
        self.enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, 16)]
        self.dec_dpr = self.enc_dpr[::-1]
        self.Encoder0 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution,self.input_resolution],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[0+i])
            for i in range(self.depth)])
        self.Encoder1 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 2,self.input_resolution // 2],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[2+i])
            for i in range(self.depth)])
        self.Encoder2 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 4,self.input_resolution // 4],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[4+i])
            for i in range(self.depth)])
        self.Encoder3 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 8,self.input_resolution // 8],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[6+i])
            for i in range(self.depth)])
        self.Encoder4 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 16,self.input_resolution // 16],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.enc_dpr[8+i])
            for i in range(self.depth)])
        self.Decoder4 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 16,self.input_resolution // 16],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[0+i])
            for i in range(self.depth)])
        self.Decoder3 = nn.ModuleList([
            WFDST(dim=channel,input_resolution=[self.input_resolution // 8,self.input_resolution // 8],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[2 + i])
            for i in range(self.depth)])
        self.Decoder2 = nn.ModuleList([
            WFDST(dim=channel, input_resolution=[self.input_resolution // 4,self.input_resolution // 4],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[4 + i])
            for i in range(self.depth)])
        self.Decoder1 = nn.ModuleList([
            WFDST(dim=channel, input_resolution=[self.input_resolution // 2,self.input_resolution // 2],
                  num_heads=self.heads, shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[6+i])
            for i in range(self.depth)])
        self.Decoder0 = nn.ModuleList([
            WFDST(dim=channel, input_resolution=[self.input_resolution ,self.input_resolution ],
                  num_heads=self.heads,shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                  window_size=self.window_size, drop_path=self.dec_dpr[8 + i])
            for i in range(self.depth)])
        self.WD_nlwt = nlwt_catone()
        self.WU_inlwt = inlwt_catone()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 128
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def forward(self, x):
        x_check, ori_size = self.check_image_size(x)
        x0 = self.convert(x_check)
        b, c0, h0, w0 = x0.shape
        EI_0 = x0.permute(0, 2, 3, 1)
        for blk in self.Encoder0:
            EI_0 = blk(EI_0, h0, w0)
        EO_0 = EI_0.permute(0, 3, 1, 2)

        EO_0_WD,Level0_size = self.WD_nlwt(EO_0)
        EI_1 = self.WD_fuse1(EO_0_WD)
        _, c1, h1, w1 = EI_1.shape
        EI_1 = EI_1.permute(0, 2, 3, 1)
        for blk in self.Encoder1:
            EI_1= blk(EI_1, h1, w1)
        EO_1 = EI_1.permute(0, 3, 1, 2)

        EO_1_WD,Level1_size = self.WD_nlwt(EO_1)
        EI_2 = self.WD_fuse2(EO_1_WD)
        _, c2, h2, w2 = EI_2.shape
        EI_2 = EI_2.permute(0, 2, 3, 1)
        for blk in self.Encoder2:
            EI_2= blk(EI_2, h2, w2)
        EO_2 = EI_2.permute(0, 3, 1, 2)

        EO_2_WD,Level2_size = self.WD_nlwt(EO_2)
        EI_3 = self.WD_fuse3(EO_2_WD)
        _, c3, h3, w3 = EI_3.shape
        EI_3 = EI_3.permute(0, 2, 3, 1)
        for blk in self.Encoder3:
            EI_3 = blk(EI_3, h3, w3)
        EO_3 = EI_3.permute(0, 3, 1, 2)

        EO_3_WD,Level3_size = self.WD_nlwt(EO_3)
        EI_4 = self.WD_fuse4(EO_3_WD)
        _, c4, h4, w4 = EI_4.shape
        EI_4 = EI_4.permute(0, 2, 3, 1)
        for blk in self.Encoder4:
            EI_4 = blk(EI_4, h4, w4)

        DI_4 = EI_4

        for blk in self.Decoder4:
            DI_4 = blk(DI_4, h4, w4)
        DO_4 = DI_4.permute(0, 3, 1, 2)

        DI_3 = self.WU_inlwt(self.WU_ifuse1(DO_4),Level3_size)
        DI_3 = self.cat1(torch.cat([DI_3,EO_3],dim=1))
        DI_3 = DI_3.permute(0,2,3,1)
        for blk in self.Decoder3:
            DI_3 = blk(DI_3, h3, w3)
        DO_3 = DI_3.permute(0, 3, 1, 2)

        DI_2 = self.WU_inlwt(self.WU_ifuse2(DO_3),Level2_size)
        DI_2 = self.cat2(torch.cat([DI_2,EO_2],dim=1))
        DI_2 = DI_2.permute(0, 2, 3, 1)
        for blk in self.Decoder2:
            DI_2 = blk(DI_2, h2, w2)
        DO_2 = DI_2.permute(0, 3, 1, 2)

        DI_1 = self.WU_inlwt(self.WU_ifuse3(DO_2),Level1_size)
        DI_1 = self.cat3(torch.cat([DI_1,EO_1],dim=1))
        DI_1 = DI_1.permute(0, 2, 3, 1)
        for blk in self.Decoder1:
            DI_1 = blk(DI_1, h1, w1)
        DO_1 = DI_1.permute(0, 3, 1, 2)

        DI_0 = self.WU_inlwt(self.WU_ifuse4(DO_1),Level0_size)
        DI_0 = self.cat4(torch.cat([DI_0,EO_0],dim=1))
        DI_0 = DI_0.permute(0, 2, 3, 1)
        for blk in self.Decoder0:
            DI_0 = blk(DI_0, h0, w0)
        DO_0=DI_0.permute(0, 3, 1, 2)

        y = self.out(DO_0)
        out = x_check + y
        out = self.restore_image_size(out, ori_size)

        return out


