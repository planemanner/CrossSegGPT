from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T
"""
Albumentation 으로 바꿀거임
Test only 목적이므로 이에 맞게 바꿀 필요 있음
Hardcoding 된 부분 최대한 외부로 뺄 것
특히, 경로 관련된 부분은 대문자로 표현해서 외부에서 변경 가능하도록 할 것
Normalization ImageNet 기준으로 하는 게 맞는지도 확인 필요

통상적으로 FSS 는 1-shot, 5-shot 으로 구성
근데, 내 기법 같은 경우 5-shot 에 더 어울리므로 이에 맞춰 평가할 것
최소 90 + @ score 나와야함
"""
class FSS1000Dataset(Dataset):
    def __init__(self, root_dir, image_size=320, k_shot=5):
        assert k_shot in [1, 5], "Only 1-shot and 5-shot supported"
        self.root_dir = root_dir
        self.image_size = image_size
        self.k_shot = k_shot

        self.class_list = sorted(os.listdir(os.path.join(root_dir, 'Images')))

        # Precompute all possible episodes
        self.episodes = []
        for class_name in self.class_list:
            if k_shot == 1:
                for i in range(1, 6):  # support1 ~ support5
                    self.episodes.append((class_name, [i]))  # One-shot: one support idx
            elif k_shot == 5:
                self.episodes.append((class_name, [1, 2, 3, 4, 5]))

        self.transform_image = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        class_name, support_indices = self.episodes[idx]
        img_dir = os.path.join(self.root_dir, 'Images', class_name)
        mask_dir = os.path.join(self.root_dir, 'Annotations', class_name)

        # Query
        query_img = Image.open(os.path.join(img_dir, f"{class_name}.jpg")).convert('RGB')
        query_mask = Image.open(os.path.join(mask_dir, f"{class_name}.png"))
        query_img = self.transform_image(query_img)
        query_mask = self.transform_mask(query_mask)

        # Support
        support_imgs, support_masks = [], []
        for i in support_indices:
            s_img_path = os.path.join(img_dir, f"{class_name}_support{i}.jpg")
            s_mask_path = os.path.join(mask_dir, f"{class_name}_support{i}.png")
            support_imgs.append(self.transform_image(Image.open(s_img_path).convert('RGB')))
            support_masks.append(self.transform_mask(Image.open(s_mask_path)))

        return {
            'class': class_name,
            'support_images': support_imgs,  # List[Tensor], len=k_shot
            'support_masks': support_masks,  # List[Tensor], len=k_shot
            'query_image': query_img,
            'query_mask': query_mask,
            'support_index': support_indices  # For debugging one-shot
        }