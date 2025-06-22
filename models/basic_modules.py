from torch import nn
import torch
from xformers.ops import memory_efficient_attention
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from utils import window_partition, window_unpartition
from xformers.ops import memory_efficient_attention

class CrossAttention(nn.Module):
    def __init__(self, query_dim:int, context_dim:int, dim_heads:int, ff_dim:int,
                 num_heads:int, dropout:float=0.1, qkv_bias:bool=False):
        super().__init__()
        self.q_embed = nn.Linear(query_dim, dim_heads * num_heads, bias=qkv_bias)
        self.kv_embed = nn.Linear(context_dim, 2 * dim_heads * num_heads, bias=qkv_bias)
        self.n_heads = num_heads
        self.dropout = dropout
        self.ff = nn.Linear(dim_heads * num_heads, ff_dim)

    def forward(self, query, context):
        # query & context shape : B, N, C
        # to use xformers properly, you must make these have B, N, n_heads, dim
        B, N_Q, _ = query.shape
        B, N_C, _ = context.shape

        q = self.q_embed(query)
        kv = self.kv_embed(context)

        q = q.contiguous().view(B, N_Q, self.n_heads, -1)
        kv = kv.contiguous().view(B, N_C, 2, self.n_heads, -1)
        k, v = kv.unbind(2)
        x = memory_efficient_attention(q, k, v, p=self.dropout)
        return self.ff(x.view(B, N_Q, -1))

class MHSA(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, dim_heads:int, qkv_bias:bool=False,
                 attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(input_dim, 3 * num_heads * dim_heads, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * dim_heads, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.FloatTensor):
        B, N, C = x.shape
        residual = x
        qkv = self.qkv(x).contiguous().view(B, N, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(2)
        x = memory_efficient_attention(q, k, v, attn_bias=None, op=None, p=self.attn_drop)
        x = x.contiguous().view(B, N, -1)
        x = residual + self.norm(self.proj(x))
        return self.proj_drop(x)
    

class CASABlock(nn.Module):
    def __init__(self, query_dim, context_dim, dim_heads, ff_dim, num_heads, dropout, qkv_bias, attn_drop, proj_drop):
        super().__init__()
        self.cra = CrossAttention(query_dim, context_dim, dim_heads, ff_dim, num_heads, dropout=dropout, qkv_bias=qkv_bias)
        self.sa = MHSA(query_dim, num_heads, dim_heads, qkv_bias, attn_drop, proj_drop)
        self.norm = nn.LayerNorm(query_dim)
        self.proj = nn.Linear(query_dim, query_dim)
        self.act = nn.GELU()
    
    def forward(self, query: torch.FloatTensor, support: torch.FloatTensor):
        x = self.cra(query, support)
        x = self.norm(x)
        x = self.sa(x)

        return self.proj(self.act(x))
        

class SelfAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.FloatTensor):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).contiguous().view(B, H * W, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(2)
        x = memory_efficient_attention(q, k, v).contiguous().view(B, H, W, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.FloatTensor, merge=0):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1:3]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # feature ensemble
        if merge > 0:
            
            prompt, inputs = x.split(x.shape[1] // 2, dim=1)
            if merge == 1:
                num_prompts = x.shape[0] // 2
                inputs = inputs.reshape(2, num_prompts, -1)
                inputs = inputs.mean(dim=1, keepdim=True).expand_as(inputs)
                inputs = inputs.reshape(*prompt.shape)
            else:
                inputs = inputs.mean(dim=0, keepdim=True).expand_as(inputs)
            x = torch.cat([prompt, inputs], dim=1)
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class LayerNorm2D(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  
    """

    def __init__(self, normalized_shape: int, eps:float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.FloatTensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim:int, in_ch: int):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.FloatTensor):
        x = self.embed(x)
        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        return x
