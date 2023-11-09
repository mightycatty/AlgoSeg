# custom codebase for segmentation


## Training 
1. prepare img/ and mask/ folders
```text
Ref:https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html#customize-datasets-by-reorganizing-data
```
2. prepare data_config.py and config.py
```text
# checkout project/cloth_parsing/dataset.py and project/cloth_parsing/upernet_r50.py
```
3. training
```bash
python ../../mmsegmentation/tools/train.py upernet_r50.py --amp --work-dir ./test
```