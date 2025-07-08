from torch.utils.data import Dataset
import random
from utils import (get_augmentation, ispositive, file_name_check,
                   sampling_grid, make_grid, mask_patches_loop, 
                   get_union, random_colormap_generator)
from PIL import Image
import numpy as np
from glob import glob
import os

class PairedMaskingDataset(Dataset):
    def __init__(self, cfg):
        self.img_list = glob(os.path.join(cfg.image_dir, f"*.{cfg.image_ext}"))
        self.mask_list = glob(os.path.join(cfg.mask_dir, f"*.{cfg.mask_ext}"))
        self.img_list.sort()
        self.mask_list.sort()
        file_name_check(self.img_list, self.mask_list)
        self.data_len = len(self.img_list)
        self.transform = get_augmentation(cfg) # Albumentation Augmentation, ToTensor 없이 !
        n_patch_h, n_patch_w = cfg.crop_height // cfg.patch_size, cfg.crop_width // cfg.patch_size
        self.grid_cells = make_grid(n_patch_h, n_patch_w) # (n_patch_h * n_patch_w, 2)
        # self.patch_handler = PatchHandler(cfg.patch_size)
        self.sampling_p = cfg.sampling_p

    def __getitem__(self, idx):
        first_img_path = self.img_list[idx]
        first_mask_path = self.mask_list[idx]
        second_idx = random.randint(0, self.data_len-1)
        second_img_path = self.img_list[second_idx]
        second_mask_path = self.mask_list[second_idx]

        # numpy 형태로 변형 map 
        first_img, first_mask = Image.open(first_img_path), Image.open(first_mask_path)
        second_img, second_mask = Image.open(second_img_path), Image.open(second_mask_path)

        if self.transform:
            first_aug = self.transform(image=first_img, mask=first_mask)
            first_img, first_mask = first_aug['image'], first_aug['mask']
            second_aug = self.transform(image=second_img, mask=second_mask)
            second_img, second_mask = second_aug['image'], second_aug['mask']

        masking_patch_indices = sampling_grid(self.grid_cells, p=self.sampling_p)
        
        mask_1_unique_class, mask_2_unique_class = np.unique(first_mask).tolist(), np.unique(second_mask).tolist()

        if ispositive(mask_1_unique_class, mask_2_unique_class):
            union_classes = get_union(mask_1_unique_class, mask_2_unique_class)
            class2colormap = random_colormap_generator(union_classes)
            max_class = max(class2colormap.keys())
            lut = np.zeros(max_class + 1, dtype=np.uint8)
            for k, v in class2colormap.items():
                lut[k] = v
            first_mask, second_mask = lut[first_mask], lut[second_mask]
            second_mask, active_mask = mask_patches_loop(second_mask, masking_patch_indices, self.patch_size)
        else:
            class2colormap = random_colormap_generator(mask_1_unique_class)
            max_class = max(class2colormap.keys())
            lut = np.zeros(max_class + 1, dtype=np.uint8)
            for k, v in class2colormap.items():
                lut[k] = v
            first_mask, second_mask = lut[first_mask], np.zeros_like(second_mask, dtype=np.uint8)
            active_mask = np.zeros_like(second_mask, dtype=np.uint8)

        return first_img, second_img, first_mask, second_mask, active_mask
    
    def __len__(self):
        return self.data_len


# ann_json = '/aidata01/warehouse/kilee/public_datasets/MSCOCO/annotations/instances_val2017.json'
# with open(ann_json, 'r') as f:
#     data = json.load(f)

# print(data['annotations'][100], data['annotations'][101])