from dataclasses import dataclass
from typing import Tuple

@dataclass
class CrossSegCFG:
    stage_depths: Tuple[int]=(2, 4, 4)
    patch_input_channels: Tuple[int]=(4, 128, 256)
    patch_embed_dims: Tuple[int]=(128, 256, 384) # 128, 256, 384
    scale_factors: Tuple[int]=(4, 2, 2) # 4, 2, 2
    sa_num_heads: int=4 # 4
    sa_dim_heads: int=128 # 128
    casa_num_heads: int=4 # 4
    casa_dim_heads: int=128 # 128
    ca_dropout: float=0.1 # 0.1
    attn_drop: float=0.1 # 0.1
    proj_drop: float=0.1 # 0.1
    qkv_bias: bool=False # False
    use_noises: Tuple[bool]=(True, True, True) # (True, True, True)
    dec_kernel_sizes: Tuple[int]=(3, 3, 3)
    dec_last_hidden_dim: int=128
    ca_ff_dim: int=256
    dec_partial_dims: Tuple[int]=(384, 256, 128)

if __name__ == "__main__":
    pass
# cfg = CrossSegCFG()
# print(cfg.scale_factors[::-1])