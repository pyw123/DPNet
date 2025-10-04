import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from mmpretrain.registry import MODELS
import numpy as np

import math
import copy
try:
    from mamba_util import PatchMerging,SimplePatchMerging, Stem, SimpleStem, Mlp
except:
    from .mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


class ConvFFN(nn.Module):

    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim

    def forward(self, x, H, W):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        result = self.to_out(out)

        return result

class DPAB(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
        d_state=64,
        local=True,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.local=local

        self.fine_feat_pixel_list = []

        # print(self.expand)
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim  # 2

        self.learnable_init_states = learnable_init_states
        self.activation = activation
        d_in_proj = 2 * self.d_inner + self.ngroups * self.d_state  # 4倍通道数+2倍d_state+2
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + self.ngroups * self.d_state

        sr_ratio=1
        self.kv_embed = nn.Conv2d(conv_dim, conv_dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()

        self.local_conv = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=5, padding=2, groups=self.d_inner),  # depthwise
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1)  # pointwise
        )

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)


        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.scale = self.headdim ** -0.5
        self.kwargs = kwargs

    def reset_fine_feat_pixel_list(self):
        self.fine_feat_pixel_list = []

    def dpa(self, x, B, D, H=None, W=None):
        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]

        V = x.permute(0, 2, 1, 3)
        K = B.view(batch, 1, seqlen, dstate)

        K_sigmoid = torch.sigmoid(K)
        K_soft = K_sigmoid / (K_sigmoid.sum(dim=-1, keepdim=True) + 1e-6)
        K_soft = K_soft.expand(-1, head, -1, -1)
        if self.local:
            x_re = rearrange(x, "b (h w) he d -> b (he d) h w", h=H, w=W)
            x_local = self.local_conv(x_re)
            x_local = rearrange(x_local, "b (he d) h w -> b he (h w) d", he=head, d=dim)
            V_s = x_local
        else:
            V_s = V

        class_prototypes = torch.einsum("bhls,bhld->bhsd", K_soft, V_s)

        attn_scores = torch.einsum("bhsd,bhld->bhsl", class_prototypes, V_s) * self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        refined_prototypes = torch.einsum("bhsl,bhld->bhsd", attn_weights, V_s)

        fine_feat_pixel = torch.einsum("bhls,bhsd->bhld", K_soft, refined_prototypes)
        res = V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)

        x = fine_feat_pixel + res

        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, u, H, W):

        batch = u.shape[0]
        zxb = self.in_proj(u)

        z, xB = torch.split(
            zxb, [self.d_inner, self.d_inner + self.ngroups * self.d_state], dim=-1
        )

        xB = xB.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xB = self.act(self.conv2d(xB))
        xB = xB.permute(0, 2, 3, 1).view(batch, H * W, -1).contiguous()

        assert self.activation in ["silu", "swish"]

        x, B = torch.split(xB, [self.d_inner, self.ngroups * self.d_state], dim=-1)

        y = self.dpa(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            B, self.D, H, W
        )

        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y)
        y = y*z

        out = self.out_proj(y)

        return out

class DPB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, d_state = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('attn_type') == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('attn_type') == 'dpabl':
            self.attn = DPAB(d_model=dim, expand=ssd_expansion, headdim = dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, d_state=d_state, **kwargs)
        elif kwargs.get('attn_type') == 'dpab':
            self.attn = DPAB(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, d_state=d_state, local=False, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)

        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, d_state=[64,64], **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            DPB(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, d_state=d_state, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H=None, W=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class DPNET(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=45,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, d_state=[64], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.simple_downsample = kwargs.get('simple_downsample', False)
        self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        self.attn_types = kwargs.get('attn_types')
        if self.simple_patch_embed:
            self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if self.simple_downsample:
            PatchMergingBlock = SimplePatchMerging
        else:
            PatchMergingBlock = PatchMerging
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['attn_type'] = self.attn_types[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               d_state = d_state[i_layer],
                               **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e9')
            return 1e9
        del model, input

        return sum(Gflops.values()) * 1e9

    def forward_features(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = H//4, W//4 # downsampled by patch_embed

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, H, W)
            H, W = H//2, W//2 # downsampled by layer

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_fine_feat_pixel_features(self):
        features = []
        for layer in self.layers:
            for blk in layer.blocks:
                if hasattr(blk.attn, 'fine_feat_pixel_list') and blk.attn.fine_feat_pixel_list:
                    features.extend(blk.attn.fine_feat_pixel_list)
        if len(features) > 0:
            return torch.cat(features, dim=0)
        else:
            return None
    def reset_all_fine_feat_pixel_lists(self):
        for layer in self.layers:
            for blk in layer.blocks:
                if hasattr(blk.attn, 'reset_fine_feat_pixel_list'):
                    blk.attn.reset_fine_feat_pixel_list()

@MODELS.register_module()
class Backbone_DPNET(DPNET):
    OUT_TYPES = {'featmap', 'avg_featmap', 'cls_token', 'raw'}
    
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, out_type='avg_featmap', **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # 添加一个全局的 norm_layer
        self.norm = norm_layer(self.num_features)
        
        # 确保 num_features 被正确设置
        if not hasattr(self, 'num_features'):
            self.num_features = int(self.embed_dim * 2 ** (len(self.layers) - 1))

        # 设置输出类型
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                           f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # 添加全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        del self.head
        self.load_pretrained(pretrained,key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt} from {key}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x = blk(x, H, W)
            if l.downsample is not None:
                y = l.downsample(x, H, W)
            else:
                y = x
            return x, y

        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        if self.simple_patch_embed:
            H, W = H//4, W//4
        else:
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                outs.append(out.contiguous())
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x

        # 根据out_type处理输出
        if self.out_type == 'raw':
            return tuple(outs)
        elif self.out_type == 'featmap':
            return tuple(outs)
        elif self.out_type == 'avg_featmap':
            # 使用全局平均池化
            return tuple([self.avgpool(out).squeeze(-1).squeeze(-1) for out in outs])
        elif self.out_type == 'cls_token':
            return tuple([out[:, :, 0, 0] for out in outs])

        return tuple(outs)


