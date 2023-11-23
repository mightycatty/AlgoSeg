import torch.nn as nn
from sam_lora.sam_lora import *
from mmseg.registry import MODELS


@MODELS.register_module()
class SAMBackbone(nn.Module):

    def __init__(self, model_type='vit_b', pretrained=None, lora_rank=4):
        self._sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
        del self._sam.mask_decoder
        del self._sam.prompt_encoder
        self.lora_sam = LoRA_Sam(self._sam, lora_rank)

    def forward(self, x):  # should return a tuple
        return self.lora_sam.sam.image_encoder(x)

    def init_weights(self, pretrained=None):
        return