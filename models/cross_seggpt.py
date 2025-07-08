from collections.abc import Sequence
from typing import Any, List
from basic_modules import PatchEmbed, CASABlock, MHSA, DecoderUnit, LayerNorm2D
from torch import nn
import lightning as L
import torch
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, input_dims:List[int], output_dims:List[int], use_noises:List[bool],
                 kernel_sizes:List[int], scale_factors:List[int], last_hidden_dim:int):
        super().__init__()
        # Operation 순서에 따라 순차적으로 numbering
        self.decode_layers = nn.ModuleList([DecoderUnit(input_dim, output_dim, kernel_size, scale_factor, use_noise) for input_dim, output_dim, kernel_size, scale_factor, use_noise in zip(input_dims, output_dims, kernel_sizes, scale_factors, use_noises)])

        self.seg_head = nn.Sequential(
            LayerNorm2D(output_dims[-1]),
            nn.GELU(),
            nn.Conv2d(output_dims[-1], last_hidden_dim, 3, padding=1, padding_mode='reflect'),
            LayerNorm2D(last_hidden_dim),
            nn.GELU(),
            nn.Conv2d(last_hidden_dim, 1, 3, padding=1, padding_mode='reflect')
        )  

    def forward(self, mult_feats):
        # Feature 크기가 큰 것부터 먼저 저장되어 있음 (즉, low scale feature 먼저)
        last_feat = mult_feats.pop()
        out = self.decode_layers[0](last_feat)
        
        for layer in self.decode_layers[1:]:
            out = layer(torch.cat([out, mult_feats.pop()], dim=1))

        return self.seg_head(out)
    
class Segmentor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # patch size 4, embed_dim 128, input_channels 4 (RGB + MASK)
        """
        All patch_embed_i are used in query and support sample encoding paths.
        patchembed 는 downsampling 역할 + embedding
        """

        self.patch_embed_1 = PatchEmbed(cfg.scale_factors[0], cfg.patch_embed_dims[0], cfg.patch_input_channels[0])
        self.patch_embed_2 = PatchEmbed(cfg.scale_factors[1], cfg.patch_embed_dims[1], cfg.patch_input_channels[1])
        self.patch_embed_3 = PatchEmbed(cfg.scale_factors[2], cfg.patch_embed_dims[2], cfg.patch_input_channels[2])

        self.sa_stage_1 = self.make_sa_blk(cfg.stage_depths[0], input_dim=cfg.patch_embed_dims[0], 
                                           num_heads=cfg.sa_num_heads, dim_heads=cfg.sa_dim_heads, 
                                           qkv_bias=cfg.qkv_bias, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)
        
        self.sa_stage_2 = self.make_sa_blk(cfg.stage_depths[1], input_dim=cfg.patch_embed_dims[1], 
                                           num_heads=cfg.sa_num_heads, dim_heads=cfg.sa_dim_heads, 
                                           qkv_bias=cfg.qkv_bias, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)
        
        self.sa_stage_3 = self.make_sa_blk(cfg.stage_depths[2], input_dim=cfg.patch_embed_dims[2], 
                                           num_heads=cfg.sa_num_heads, dim_heads=cfg.sa_dim_heads, 
                                           qkv_bias=cfg.qkv_bias, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)
        
        self.casa_stage_1 = self.make_casa_blk(cfg.stage_depths[0], query_dim=cfg.patch_embed_dims[0], context_dim=cfg.patch_embed_dims[0], 
                                               dim_heads=cfg.casa_dim_heads, num_heads=cfg.casa_num_heads, dropout=cfg.ca_dropout, 
                                               qkv_bias=False, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)
        
        self.casa_stage_2 = self.make_casa_blk(cfg.stage_depths[1], query_dim=cfg.patch_embed_dims[1], context_dim=cfg.patch_embed_dims[1], 
                                               dim_heads=cfg.casa_dim_heads,  num_heads=cfg.casa_num_heads, dropout=cfg.ca_dropout, 
                                               qkv_bias=False, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)   
             
        self.casa_stage_3 = self.make_casa_blk(cfg.stage_depths[2], query_dim=cfg.patch_embed_dims[2], context_dim=cfg.patch_embed_dims[2], 
                                               dim_heads=cfg.casa_dim_heads, num_heads=cfg.casa_num_heads, dropout=cfg.ca_dropout, 
                                               qkv_bias=False, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)

        self.decoder = Decoder(input_dims=[cfg.patch_embed_dims[-1], 
                                           cfg.patch_embed_dims[-2] + cfg.dec_partial_dims[0], 
                                           cfg.patch_embed_dims[-3] + cfg.dec_partial_dims[1]], 

                               output_dims=cfg.dec_partial_dims, 
                               use_noises=cfg.use_noises, 
                               kernel_sizes=cfg.dec_kernel_sizes,
                               scale_factors=cfg.scale_factors[::-1], 
                               last_hidden_dim=cfg.dec_last_hidden_dim)
        
    def unpatchify(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5) 
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    
    def make_sa_blk(self, depth:int, input_dim:int, num_heads:int, 
                    dim_heads:int, qkv_bias:bool, attn_drop:float, proj_drop:float):
        layers = []
        
        for _ in range(depth):
            layers.append(MHSA(input_dim, num_heads, dim_heads, qkv_bias, attn_drop, proj_drop))
        return nn.Sequential(*layers)
    
    def make_casa_blk(self, depth: int, query_dim:int, context_dim:int,
                      dim_heads:int, num_heads:int, dropout:float, 
                      qkv_bias:bool, attn_drop:float, proj_drop:float):
        layers = []

        for _ in range(depth):
            layers.append(CASABlock(query_dim, context_dim, dim_heads,  
                                    num_heads, dropout, qkv_bias, attn_drop, proj_drop))

        return nn.ModuleList(layers)

    def forward_encoding(self, query, support):

        q_1 = self.patch_embed_1(query)
        sup_1 = self.patch_embed_1(support)
        
        sa_path_q_1 = self.sa_stage_1(q_1)

        for casa_stage_1_layer in self.casa_stage_1:
            sup_1 = casa_stage_1_layer(sa_path_q_1, sup_1)

        ca_path_q_1 = sup_1

        # Collect this for low-scale pattern
        sa_path_q_1 = self.unpatchify(sa_path_q_1)
        ca_path_q_1 = self.unpatchify(ca_path_q_1)

        q_2 = self.patch_embed_2(sa_path_q_1)
        sup_2 = self.patch_embed_2(ca_path_q_1)

        sa_path_q_2 = self.sa_stage_2(q_2)
        for casa_stage_2_layer in self.casa_stage_2:
            sup_2 = casa_stage_2_layer(sa_path_q_2, sup_2)

        ca_path_q_2 = sup_2

        # Collect this for middle-scale pattern
        sa_path_q_2 = self.unpatchify(sa_path_q_2)
        ca_path_q_2 = self.unpatchify(ca_path_q_2)

        q_3 = self.patch_embed_3(sa_path_q_2)
        sup_3 = self.patch_embed_3(ca_path_q_2)

        sa_path_q_3 = self.sa_stage_3(q_3)

        for casa_stage_3_layer in self.casa_stage_3:
            sup_3 = casa_stage_3_layer(sa_path_q_3, sup_3)

        ca_path_q_3 = sup_3

        # Collect this for high-scale pattern
        sa_path_q_3 = self.unpatchify(sa_path_q_3)
        ca_path_q_3 = self.unpatchify(ca_path_q_3)

        return [ca_path_q_1, ca_path_q_2, ca_path_q_3]

    def forward(self, query, support):
        multiscale_features = self.forward_encoding(query, support)
        decoded = self.decoder(multiscale_features)
        return decoded

class CrossSegGPT(L.LightningModule):
    def __init__(self, cfg):
        self.cross_unet = Segmentor(cfg)
    
    def training_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Optimizer | Sequence[Optimizer] | tuple[Sequence[Optimizer], Sequence[LRScheduler | ReduceLROnPlateau | LRSchedulerConfig]] | OptimizerConfig | OptimizerLRSchedulerConfig | Sequence[OptimizerConfig] | Sequence[OptimizerLRSchedulerConfig] | None:
        return super().configure_optimizers()
    
    def configure_callbacks(self) -> Sequence[Callback] | L.Callback:
        return super().configure_callbacks()
    
    def on_train_epoch_end(self, ):
        pass
    
    def fewshot_inference(self, query, support_samples):
        pass

def get_num_params(model, unit=1e7):
    n_params = 0
    for param in model.parameters():
        if param.requires_grad:
            n_params += param.numel()
    print(n_params/unit)

if __name__ == "__main__":
    ##
    from dataclasses import dataclass, field
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
    ##
    seg_model = Segmentor(CrossSegCFG()).cuda()
    get_num_params(seg_model)
    bsz = 4
    in_chans = 4
    h, w = 448, 448
    n_supports = 5

    sample_query = torch.randn(bsz, in_chans, h, w, device='cuda:0')
    sample_support = torch.randn(bsz, in_chans, h, w, device='cuda:0')

    output = seg_model(sample_query, sample_support)
    print(output.shape)

