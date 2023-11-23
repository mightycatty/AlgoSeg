# Personal repo for segmentation
SETUP
```bash
# install mmseg
cd mmsegmentation
pip install -e .
# install segment_anything
```
---
## WIKI
### Custom dataset training with mmseg
1. prepare img/ and mask/ folders
```text
Ref:https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html#customize-datasets-by-reorganizing-data
mask: pixel value in range[0, class_num index-1] # setup metainfo in dataset.py
```
2. prepare data_config.py and config.py
```text
# checkout project/cloth_parsing/dataset.py and project/cloth_parsing/upernet_r50.py
```
3. training
```bash
python ../../mmsegmentation/tools/train.py upernet_r50.py --amp --work-dir ./test
```
4. inference
```bash
python project/cloth_parsing/image_demo_scrips.py
```
### Custom module for mmseg
https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/add_models.html
