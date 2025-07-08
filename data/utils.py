import albumentations as A
import lightning as L
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import math
import random
import os
from tqdm import tqdm

class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        """
        lightning 의 dataset 의 initialization 순서 
        prepare_data 호출 -> setup 호출 setup 은 trainer 에서 fit test 이런 단어가 오면.
        """
    def prepare_data(self):
        # Download data from web or something like that.
        pass

    def setup(self, stage: str):
        # stage 는 train, validation, 뭐 이런거. Enum 형태로 해도 됨.
        # 여기에서 일반적인 torch dataset 불러온다고 보면 됨
        if stage == 'fit':
            # 여기에서 augmentation 까지 함께 return 받아서 줘야함
            self.train_set = None
            self.val_set = None

        if stage == 'test':
            self.test_set = None

        return 
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.cfg.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.cfg.batch_size)

class PatchHandler:
    def __init__(self, patch_size):
        # Non-overlapping patch handler
        self.ps = patch_size
        self.unfolder = nn.Unfold(patch_size, stride=patch_size)

    def patchify(self, input_tensor: torch.Tensor):
        """
        input_tensor must have the shape like : (b, c, h, w)
        and must be divided by the patch size
        """
        patches = self.unfolder(input_tensor)
        patches = patches.view(bsz, ch, ps, ps, -1).permute(0, 4, 1, 2, 3)
        return patches
    
    def reconstruction(self, patches: torch.Tensor):
        b, n_patches, ch, ph, pw = patches.shape
        # b, ch, n_patches, ph, pw
        factor = int(math.sqrt(n_patches))
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(b, ch * ph * pw, n_patches)
        reconstructed = F.fold(patches, (factor * ph, factor * pw), kernel_size=ps, stride=ps)        
        return reconstructed

def mask_patches_loop(mask, masking_patch_indices, patch_size):
    masked = mask.copy()
    for r, c in masking_patch_indices:
        r_start = r * patch_size
        c_start = c * patch_size
        masked[r_start:r_start + patch_size, c_start:c_start + patch_size] = 0
    zero_mask = (masked == 0)
    return masked, zero_mask

def get_augmentation(cfg):
    """
    SegGPT 원문
    - Resize, Crop, ColorJitter, Horizontal Flip
    
    """
    resize = A.Resize(height=cfg.resize_height, width=cfg.resize_width)
    rand_crop = A.RandomCrop(height=cfg.crop_height, width=cfg.crop_width)
    color_jitter = A.ColorJitter()
    hflip = A.HorizontalFlip()
    vflip = A.VerticalFlip()
    transform = A.Compose([resize, rand_crop, color_jitter, hflip, vflip])
    return transform

def make_grid(grid_h_len, grid_w_len):
    i_indices = np.arange(grid_h_len)
    j_indices = np.arange(grid_w_len)

    ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
    return np.stack([ii.ravel(), jj.ravel()], axis=1)

def ispositive(reference:List[int], query:List[int]) -> bool:
    """
    mask 의 표현이 각 class 를 나타냄
    """
    mask_1_unique_class = set(reference)

    for mask_2_class in query:
        if mask_2_class in mask_1_unique_class:
            return True
    return False

def get_union(reference:List[int], query:List[int]) -> List[int]:
    union = []
    ref = set(reference)
    for q_class in query:
        if q_class in ref:
            union.append(q_class)
    return union

def get_random_mask_indices(n_patches: int, masking_ratio:float=0.7):
    """
    Note : this function just returns the indices of patches to be masked out.
    """
    indices = list(range(n_patches))
    n_masks = int(len(indices) * masking_ratio)
    random.shuffle(indices)
    return indices[:n_masks]

def random_colormap_generator(unique_classes: List[int], max_color_id:int=255):
    n_colors = len(unique_classes)
    color_space = [i for i in range(max(n_colors, max_color_id))]
    random.shuffle(color_space)
    return {unique_classes[i]: color_space[i] for i in range(n_colors)}

def file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def file_name_check(img_list: List[str], mask_list: List[str]):
    
    n = len(img_list)

    for i in tqdm(range(n), desc='Checking image and mask file names are same...'):
        assert file_name(img_list[i]) == file_name(mask_list[i])
    print("...Done !!!")

def sampling_grid(grid_cells, p:float=0.2):
    n = len(grid_cells)
    n_samples = int(n * p)
    sample_indices = np.random.choice(n, size=n_samples, replace=False)
    return grid_cells[sample_indices]

if __name__ == '__main__':
    # bsz, ch, h, w = 2, 3, 256, 256
    # ps = 16
    # device = 'cuda:0'
    # img = Image.open('/home/smddls77/CrossSegGPT/preprocessing/ADE_val_00000779.jpg')
    # # b, c, h, w
    # img = torch.tensor(np.array(img), dtype=torch.float, device=device).unsqueeze(0).permute(0, 3, 1, 2) 
    # img = F.interpolate(img, size=(h, w))
    grid_cells = make_grid(10, 10)
    print(sampling_grid(grid_cells))
    # patches = patchifying(img, ps)
    # # save_patches("/home/smddls77/CrossSegGPT/preprocessing/patch_saved", patches[0])

    # # recon_x = reconstruct_from_patches(patches, ps, h, w)
    # recon_x = recon_x_1(patches)
    # print(recon_x.shape)
    # recon_x = recon_x.permute(0, 2, 3, 1).cpu().numpy().squeeze().astype(np.uint8)
    # # print(recon_x.shape)
    # Image.fromarray(recon_x).save('/home/smddls77/CrossSegGPT/preprocessing/recon_ADE_val_00000779.jpg')

    # # print((x - r_x_3).sum())


    # 128 X 128 시간 비교
    # 2.1588802337646484e06 1.421189308166504e-05
    # 256 X 256 시간 비교
    # 2.131223678588867e-06 1.4373064041137694e-05
    # 448 X 448 시간 비교 
    # 2.251863479614258e-06 1.4292478561401368e-05
    # 1024 X 1024 시간 비교
    # 2.1448135375976564e-06 1.5121698379516601e-05
    # 4096 X 4096 시간 비교
    # 2.2192001342773437e-06 0.00029014968872070315