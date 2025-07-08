import json
from collections import defaultdict
from PIL import Image, ImageDraw
import numpy as np 

def polygons_to_mask(image_dict):
    width = image_dict["width"]
    height = image_dict["height"]
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in image_dict["annotations"]:
        cat_id = ann["category_id"]
        segmentations = ann["segmentation"]
        # COCO polygon은 여러 개의 polygon이 있을 수 있음
        for seg in segmentations:
            if len(seg) < 6:
                continue  # polygon은 최소 3점(6좌표) 필요
            polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=None, fill=1)
            mask_part = np.array(img, dtype=bool)
            mask[mask_part] = cat_id  # 겹칠 때는 마지막 polygon의 category_id가 적용됨

    return mask


# COCO annotation 파일 로드
coco_ann_path = '/aidata01/warehouse/kilee/public_datasets/MSCOCO/annotations/instances_train2017.json'
with open(coco_ann_path, 'r') as f:
    coco = json.load(f)

# image_id별로 annotation을 그룹핑
image_to_annotations = defaultdict(list)
for ann in coco['annotations']:
    image_to_annotations[ann['image_id']].append(ann)

# image별로 dict 생성 (list of dict 형태)
image_dicts = []
for img in coco['images']:
    img_id = img['id']
    img_dict = {
        'image_id': img_id,
        'file_name': img['file_name'],
        'width': img.get('width'),
        'height': img.get('height'),
        'annotations': image_to_annotations[img_id]
    }
    image_dicts.append(img_dict)

# 결과 저장 (예: JSON 파일)
save_path = "/home/smddls77/CrossSegGPT/datapreprocessing/coco_train_2017_image_wise_integration.json"
with open(save_path, 'w') as f:
    json.dump(image_dicts, f, indent=4)

