import torch.nn as nn
from sam_lora.sam_lora import *
from mmseg.registry import MODELS


@MODELS.register_module()
class SAMBackbone(SAMImageEncoderASBackbone):
    def __init__(self, *args, **kwargs):
        super(SAMBackbone, self).__init__(*args, **kwargs)