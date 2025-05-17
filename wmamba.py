import math
import torch
import torch.nn as nn
from functools import partial
from typing import Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba2 import Mamba2


class GMABlock(nn.Module):
    """
    Grouped Multi-Scale Attention Block (GMABlock)  for feature enhancement.
    Args:
        channels (int): Number of input channels.
        factor (int): Factor for group normalization.
    """

    def __init__(self, channels: int, factor: int = 8):
        super(GMABlock, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0,  "Number of groups must not exceed channels"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1_l = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1)
        self.conv1x1_r = nn.Conv2d(channels // self.groups * 2, channels // self.groups * 2, kernel_size=1)
        self.conv5x5 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=5, padding=2)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)

    def forward(self, input_tensor):
        """
        Forward pass for the GMABlock.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after applying EMA.
        """
        b, c, h, w = input_tensor.size()
        group_x = input_tensor.reshape(b * self.groups, -1, h, w)  # shape: (b*g, c//g, h, w)

        # Pool along height and width
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # Combine pooled results
        hw = self.conv1x1_l(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # Apply group normalization
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # Apply convolutions
        x2 = self.conv3x3(group_x)
        x3 = self.conv5x5(group_x)
        x23 = self.conv1x1_r(torch.cat([x2, x3], dim=1))

        x2, x3 = torch.split(x23, [x2.shape[1], x2.shape[1]], dim=1)
        x2 = self.gn(group_x * x2.sigmoid() * x3.sigmoid())

        # Calculate attention weights
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        # Combine weights
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # Apply weights to the input
        output = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        return output


class GMAModule(nn.Module):
    """
    Grouped Multi-Scale Attention Module (GMAModule) for enhancing features.

    Args:
        num_feat (int): Number of input feature channels.
        compress_ratio (int, optional): Compression ratio for the intermediate feature maps. Default is 6.
    """
    def __init__(self, num_feat, compress_ratio=6):
        super(GMAModule, self).__init__()
        self.gmsam = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1),
            GMABlock(num_feat)
        )

    def forward(self, input_tensor):
        return self.gmsam(input_tensor)


class GSSDBlock(nn.Module):
    """
    GSSDBlock: GMA(Grouped Multi-Scale Attention) State Space Duality Block

    Combines several neural network techniques including normalization, attention,
    and convolution for feature enhancement.

    Args:
        hidden_dim (int): The dimensionality of input features.
        drop_path (float): Probability of dropping paths during training.
        norm_layer (Callable[..., nn.Module]): Normalization layer used.
    """
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.mamba2 = Mamba2(d_model=hidden_dim, d_state=64, headdim=8)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = GMAModule(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        B, L, C = input.shape
        H = int(math.sqrt(L))
        W = H
        reshaped_input = input.view(B, H, W, C).contiguous()  # [B,H,W,C]

        # First processing step with normalization and Mamba2
        x = self.ln_1(input)
        x = self.mamba2(x)
        attention_output = x.view(B, H, W, C).contiguous()  # [B,H,W,C]

        # Apply skip connection
        x = reshaped_input * self.skip_scale + self.drop_path(attention_output)
        # x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + self.ln_2(x).permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1).contiguous()

        # Reshape back to original dimensions
        output = x.view(B, -1, C).contiguous()
        return output


class DenseGSSD(nn.Module):
    """
        DenseGSSD is a module that consists of multiple GSSDBlocks with Dense connections
        for feature extraction.

        Args:
            dim (int): The dimension of the input features.
            drop_path (float): The drop path rate.
            img_size (int): The size of the input images.
            patch_size (int): The size of the patches to split the image.
            resi_connection (str): Type of residual connection: '1conv' or '3conv'.
        """

    def __init__(self,
                 dim: int,
                 drop_path: float = 0.,
                 img_size: int = 32,
                 patch_size: int = 1,
                 resi_connection: str = '1conv'):
        super(DenseGSSD, self).__init__()

        self.dim = dim

        # Creating multiple VSSBlocks
        self.gssdb1 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)
        self.gssdb2 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)
        self.gssdb3 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)
        self.gssdb4 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)
        self.gssdb5 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)
        self.gssdb6 = GSSDBlock(hidden_dim=dim, drop_path=drop_path)

        # Conditional creation of convolutional layer
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, stride=1, padding=1)
            )
        else:
            raise ValueError("Invalid value for resi_connection. Choose '1conv' or '3conv'.")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=nn.LayerNorm)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)

    def forward(self, x):
        y1 = self.gssdb1(x)
        y1 = y1 + x

        y2 = self.gssdb2(y1)
        y2 = y2 + y1 + x

        y3 = self.gssdb2(y2)
        y3 = y3 + y2 + y1 + x

        # y4 = self.gssdb3(y3)
        # y4 = y4 + y3 + y2 + y1 + x
        #
        # y5 = self.gssdb3(y4)
        # y5 = y5 + y4 + y3 + y2 + y1 + x
        #
        # y6 = self.gssdb3(y5)
        # y6 = y6 + y5 + y4 + y3 + y2 + y1 + x

        # Unembed, convolve, and embed
        y = self.patch_unembed(y3)
        y = self.conv(y)
        y = self.patch_embed(y)

        return y


class SFEModule(nn.Module):
    """
    SFEModule: Shallow feature extraction module.
    The output is calculated as:
    Shallow_Extraction(input_tensor) = DWConv(input_tensor) + Conv(input_tensor)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(SFEModule, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=True
        )

        # Standard convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

    def forward(self, input_tensor):
        # B, C, H, W = x.shape
        return self.depthwise(input_tensor) + self.conv(input_tensor)


class MMIFEModule(nn.Module):
    """
    MMIFEModule: A multimodal approach using mutual information (MI) for feature extraction.

    Args:
        dim (int): The output channel dimension for the convolutional layers.
        num_para (int): The number of input channels (or parallel processes).
    """

    def __init__(self,
                 dim: int,
                 num_para: int = 12):
        super(MMIFEModule, self).__init__()
        self.dim = dim
        self.num_p = num_para

        # Define convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1) for _ in range(num_para)
        ])

        # MI coefficients, normalized
        self.MI = torch.tensor([2.864, 2.650, 2.639, 2.588, 2.518, 2.329, 2.238, 1.865, 1.389, 1.102, 0.492, 0.008])
        self.coefficients = torch.nn.Parameter(self.MI[:num_para] / self.MI[:num_para].sum(), requires_grad=False)

    def forward(self, x):
        if x.shape[1] != self.num_p:
            raise ValueError(f"Input has {x.shape[1]} channels, but {self.num_p} channels are expected.")
        outputs = []
        for i in range(self.num_p):
            output = self.conv_layers[i](x[:, i:i + 1, :, :])  # 仅选择第 i 个通道
            outputs.append(output * self.coefficients[i])
        final_output = sum(outputs)

        return final_output


class CABlock(nn.Module):
    """
        CABlock : Channel attention Block used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(CABlock, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H
        y = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        y = self.attention(y)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = y.view(B, -1, C).contiguous()

        return x * y


class WMamba(nn.Module):
    """
    A model for image restoration that combines shallow and deep feature extraction,
    along with channel attention mechanisms.

    Args:
        img_size (int): The height and width of the input image.
        patch_size (int): The size of the patches used in feature extraction.
        in_chans (int): The number of input channels.
        in_chans1 (int): The number of input channels for secondary input.
        embed_dim (int): The embedding dimension for feature maps.
        drop_rate (float): The dropout rate.
        norm_layer: The normalization layer to use.
        patch_norm (bool): Whether to apply normalization on patches.
        resi_connection (str): Type of residual connection ('1conv' or '3conv').
    """

    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 128,
                 drop_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm: bool = True,
                 resi_connection: str = '1conv',
                 **kwargs):
        super(WMamba, self).__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans

        # Shallow feature extraction
        self.sfem = SFEModule(num_in_ch, embed_dim)

        # Deep feature extraction
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # DenseGSSD Modules
        self.dgssd1 = DenseGSSD(dim=embed_dim)
        self.dgssd2 = DenseGSSD(dim=embed_dim)
        self.dgssd3 = DenseGSSD(dim=embed_dim)
        self.dgssd4 = DenseGSSD(dim=embed_dim)

        # Channel Attention Modules
        self.ca5 = CABlock(num_feat=embed_dim)
        self.ca1 = CABlock(num_feat=embed_dim)
        self.ca2 = CABlock(num_feat=embed_dim)
        self.ca3 = CABlock(num_feat=embed_dim)
        self.ca4 = CABlock(num_feat=embed_dim)

        # self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=1, padding=1)
            )

        # Restoration module
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        """
        Forward pass for the WMamba layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            x1 (torch.Tensor): Secondary input tensor of shape (N, in_chans1, H, W).

        Returns:
           torch.Tensor: Output tensor after processing.
        """
        x_scale = 1.2

        x_first = self.sfem(x)
        x = self.patch_embed(x_first)
        x = self.pos_drop(x)

        y1 = self.dgssd1(x) * x_scale + self.ca1(x)
        y2 = self.dgssd2(y1) * x_scale + self.ca2(y1)
        # y3 = self.dgssd3(y2) * x_scale + self.ca3(y2)
        # y4 = self.dgssd4(y3) * x_scale + self.ca4(y3)

        # Residual
        y5 = y2 * x_scale + self.ca5(x)

        y = self.patch_unembed(y5)
        y = self.conv_after_body(y)
        y = x_first + y
        y = self.conv_last(y)

        return y


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        H = int(math.sqrt(x.shape[1]))
        W = H
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, H, W)
        return x

    def flops(self):
        flops = 0
        return flops