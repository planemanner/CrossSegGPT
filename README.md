# Cross SegGPT 
- This research is extension of [SegGPT, ICCV 2023](https://github.com/baaivision/Painter/tree/main/SegGPT)
- All code will be available after publishing in a conference.

# Data
- Training Data
  - MSCOCO, ADE20K, PASCAL VOC 
- Evaluation Data
  - FewShot Inference
    - MSCOCO-20i, PASCAL-5i
      - one-shot and few-shot (how many ?)
      - Reference : [SegGPT](https://arxiv.org/pdf/2304.03284)
    - FSS 1000 (Validation 결과 활용)
    - YouTube-VOS 2019
    - DAVIS 2017
    - ADE20K
    - MSCOCO Panoptic Seg (Optional)
- Validation on training
  - FSS 1000

# Preprocessing
## MSCOCO
Step 1. MSCOCO 의 모든 데이터들을 Image 단위로 통합
```
cd datapreprocessing
python coco_integration.py
```
Step 2. 변환된 annotation file 을 이용해서 image 단위 segmentation mask 생성
```
cd datapreprocessing
python coco_ann2mask.py
```
## ADE20K, VOC12
- 별도 처리 필요없음
- 이미 모두 MASK 형태

# To do
## Data part
- evaluation 을 위한 각각의 데이터 처리 방법 마련
- validation set 은 어떤 것으로 할 지 결정

## training part
- lightning module 내 training part 구현

## Inference part
- fewshot inference 를 위한 흐름 구현
