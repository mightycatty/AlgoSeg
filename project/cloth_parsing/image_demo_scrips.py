import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
from algo_qol.utils.file_utils import get_file_recursively


config_file = '/home/justsomeone/code/seg/project/cloth_parsing/upernet_r50.py'
checkpoint_file = '/home/justsomeone/code/seg/project/cloth_parsing/test/iter_112000.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda')
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)
# test a single image
img_list = get_file_recursively(r'/home/justsomeone/data/ur')
for img in img_list:
    result = inference_model(model, img)
    # show the results
    vis_result = show_result_pyplot(model, img, result, show=True)
    # plt.imshow(vis_result)
    # plt.show()