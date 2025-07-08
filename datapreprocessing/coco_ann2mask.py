from PIL import Image, ImageDraw
import numpy as np 
import json
from tqdm import tqdm
import os
from pycocotools import mask as maskUtils

def annotation_to_mask(image_dict):
    width = image_dict["width"]
    height = image_dict["height"]
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in image_dict["annotations"]:
        cat_id = ann["category_id"]
        seg = ann["segmentation"]

        if isinstance(seg, list):
            # Polygon
            for poly in seg:
                if len(poly) < 6:
                    continue
                polygon = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                img = Image.new('L', (width, height), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=None, fill=1)
                mask_part = np.array(img, dtype=bool)
                mask[mask_part] = cat_id

        elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
            # RLE
            if isinstance(seg['counts'], list):
                # uncompressed RLE → compressed RLE로 변환
                rle = maskUtils.frPyObjects(seg, height, width)
                if isinstance(rle, list):
                    rle = rle[0]
            else:
                # 이미 compressed RLE
                rle = seg
            mask_rle = maskUtils.decode(rle)
            mask[mask_rle == 1] = cat_id

    return mask

if __name__ == "__main__":
    ann_path = "./CrossSegGPT/datapreprocessing/coco_train_2017_image_wise_integration.json"
    save_dir = "./CrossSegGPT/datapreprocessing/coco_train_masks"

    with open(ann_path, 'r') as f:
        data = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    for image_dict in tqdm(data):  # data는 통합된 list of dict
        mask = annotation_to_mask(image_dict)
        dst_path = os.path.join(save_dir, image_dict["file_name"].replace('.jpg', '.png'))
        Image.fromarray(mask).save(dst_path)
