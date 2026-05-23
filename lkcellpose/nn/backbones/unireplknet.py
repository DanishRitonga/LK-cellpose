# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Github source: https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
# Vendored for LK-Cellpose. MMDetection/MMSegmentation code removed.
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
import torch.utils.checkpoint as cp

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class GRNwithNHWC(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def _get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = (kernel_size[0] == kernel_size[1] and kernel_size[0] > 5
                       and padding == (kernel_size[0] // 2, kernel_size[1] // 2))
    if attempt_use_lk_impl and need_large_impl:
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        except ImportError:
            DepthWiseConv2dImplicitGEMM = None
        if (DepthWiseConv2dImplicitGEMM is not None and need_large_impl
                and in_channels == out_channels and out_channels == groups
                and stride == 1 and dilation == 1):
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def _get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


def _fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)


def _convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = _convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = _get_conv2d(channels, channels, kernel_size, stride=1,
                                      padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                      attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = _get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k,
                                           stride=1, padding=(r * (k - 1) + 1) // 2, dilation=r,
                                           groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', _get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = _fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f'dil_conv_k{k}_{r}')
                bn = self.__getattr__(f'dil_bn_k{k}_{r}')
                branch_k, branch_b = _fuse_bn(conv, bn)
                origin_k = _merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = _get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                      padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0),
                                      bias=True, attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f'dil_conv_k{k}_{r}')
                self.__delattr__(f'dil_bn_k{k}_{r}')


class UniRepLKNetBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=0., layer_scale_init_value=1e-6,
                 deploy=False, attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=False, ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = _get_conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2, dilation=1, groups=dim, bias=deploy,
                                      attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = _get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim), NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim, bias=False), NHWCtoNCHW(),
                                         _get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
                      if (not deploy) and layer_scale_init_value is not None
                      and layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def _f(x):
            return x + self.compute_residual(x)
        if self.with_cp and inputs.requires_grad:
            return cp.checkpoint(_f, inputs)
        return _f(inputs)

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                    self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels,
                                self.dwconv.kernel_size, padding=self.dwconv.padding,
                                groups=self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


default_UniRepLKNet_A_F_P_kernel_sizes = ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))
default_UniRepLKNet_N_kernel_sizes = ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13, 13, 13), (13, 13))
default_UniRepLKNet_T_kernel_sizes = ((3, 3, 3), (13, 13, 13),
                                       (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                       (13, 13, 13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = ((3, 3, 3), (13, 13, 13),
                                              (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3,
                                               13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                              (13, 13, 13))
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes,
}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class UniRepLKNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 27, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., kernel_sizes=None, deploy=False, with_cp=False,
                 attempt_use_lk_impl=True, use_sync_bn=False, **kwargs):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths, '
                                 'please specify kernel sizes for each block')
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp
        self.dims = dims
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[UniRepLKNetBlock(dim=dims[i], kernel_size=kernel_sizes[i][j],
                                   drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn)
                  for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = lambda c: LayerNorm(c, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            self.add_module(f'norm{i_layer}', norm_layer(dims[i_layer]))

        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
            outs.append(getattr(self, f'norm{stage_idx}')(x))
        return outs

    def forward(self, x):
        xs = self.forward_features(x)
        x = xs[-1].mean([-2, -1])
        x = self.head(x)
        return x

    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()


_huggingface_file_names = {
    "unireplknet_a_1k": "unireplknet_a_in1k_224_acc77.03.pth",
    "unireplknet_f_1k": "unireplknet_f_in1k_224_acc78.58.pth",
    "unireplknet_p_1k": "unireplknet_p_in1k_224_acc80.23.pth",
    "unireplknet_n_1k": "unireplknet_n_in1k_224_acc81.64.pth",
    "unireplknet_t_1k": "unireplknet_t_in1k_224_acc83.21.pth",
    "unireplknet_s_1k": "unireplknet_s_in1k_224_acc83.91.pth",
    "unireplknet_s_22k": "unireplknet_s_in22k_pretrain.pth",
    "unireplknet_s_22k_to_1k": "unireplknet_s_in22k_to_in1k_384_acc86.44.pth",
    "unireplknet_b_22k": "unireplknet_b_in22k_pretrain.pth",
    "unireplknet_b_22k_to_1k": "unireplknet_b_in22k_to_in1k_384_acc87.40.pth",
    "unireplknet_l_22k": "unireplknet_l_in22k_pretrain.pth",
    "unireplknet_l_22k_to_1k": "unireplknet_l_in22k_to_in1k_384_acc87.88.pth",
    "unireplknet_xl_22k": "unireplknet_xl_in22k_pretrain.pth",
    "unireplknet_xl_22k_to_1k": "unireplknet_xl_in22k_to_in1k_384_acc87.96.pth",
}


def _load_with_key(model, key):
    if hf_hub_download is not None:
        repo_id = 'DingXiaoH/UniRepLKNet'
        cache_file = hf_hub_download(repo_id=repo_id, filename=_huggingface_file_names[key])
        checkpoint = torch.load(cache_file, map_location='cpu', weights_only=False)
    else:
        raise RuntimeError('huggingface_hub is required to download UniRepLKNet weights. '
                           'Install with: pip install huggingface_hub')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)


def _initialize_with_pretrained(model, model_name, in_1k_pretrained, in_22k_pretrained, in_22k_to_1k):
    if in_1k_pretrained:
        key = model_name + '_1k'
    elif in_22k_pretrained:
        key = model_name + '_22k'
    elif in_22k_to_1k:
        key = model_name + '_22k_to_1k'
    else:
        return
    _load_with_key(model, key)


@register_model
def unireplknet_a(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(40, 80, 160, 320),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_a', in_1k_pretrained, False, False)
    return model


@register_model
def unireplknet_f(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(48, 96, 192, 384),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_f', in_1k_pretrained, False, False)
    return model


@register_model
def unireplknet_p(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(64, 128, 256, 512),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_p', in_1k_pretrained, False, False)
    return model


@register_model
def unireplknet_n(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_N_depths, dims=(80, 160, 320, 640),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_n', in_1k_pretrained, False, False)
    return model


@register_model
def unireplknet_t(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_T_depths, dims=(80, 160, 320, 640),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_t', in_1k_pretrained, False, False)
    return model


@register_model
def unireplknet_s(in_1k_pretrained=False, in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(96, 192, 384, 768),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_s', in_1k_pretrained, in_22k_pretrained, in_22k_to_1k)
    return model


@register_model
def unireplknet_b(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(128, 256, 512, 1024),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_b', False, in_22k_pretrained, in_22k_to_1k)
    return model


@register_model
def unireplknet_l(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(192, 384, 768, 1536),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_l', False, in_22k_pretrained, in_22k_to_1k)
    return model


@register_model
def unireplknet_xl(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(256, 512, 1024, 2048),
                        attempt_use_lk_impl=False, **kwargs)
    _initialize_with_pretrained(model, 'unireplknet_xl', False, in_22k_pretrained, in_22k_to_1k)
    return model


UNIREPLKNET_VARIANTS = {
    "unireplknet_t": [80, 160, 320, 640],
    "unireplknet_s": [96, 192, 384, 768],
    "unireplknet_b": [128, 256, 512, 1024],
    "unireplknet_l": [192, 384, 768, 1536],
}


def unireplknet_backbone(name: str = "unireplknet_s", pretrained: bool = True,
                         pretrained_tag: str | None = None, **kwargs):
    """Create UniRepLKNet backbone returning 4 multi-scale feature maps.

    Creates the model via timm's registered factory (which handles pretrained
    weight loading from HuggingFace), then replaces the classification head
    with Identity and overrides forward to call forward_features.

    Returns:
        model: nn.Module whose forward() returns list of 4 feature maps
        feature_info: dict with "channels" and "reductions" keys
    """
    import timm

    pretrained_kwargs = {}
    if pretrained:
        if pretrained_tag and "in22k" in pretrained_tag and "in1k" not in pretrained_tag:
            pretrained_kwargs["in_22k_pretrained"] = True
        elif pretrained_tag and "in22k_to_in1k" in pretrained_tag:
            pretrained_kwargs["in_22k_to_1k"] = True
        else:
            pretrained_kwargs["in_1k_pretrained"] = True

    model = timm.create_model(
        name,
        pretrained=False,
        **pretrained_kwargs,
        **kwargs,
    )

    model.head = nn.Identity()
    model.forward = model.forward_features

    channels = UNIREPLKNET_VARIANTS.get(name)
    if channels is None:
        channels = [f.shape[1] for f in model.forward_features(
            torch.zeros(1, 3, 32, 32))]

    feature_info = {
        "channels": channels,
        "reductions": [4, 8, 16, 32],
    }
    return model, feature_info
