import copy
import torch
import torch.nn as nn
from .bra_unet_system import BRAUnetSystem
from timm.models import register_model


class BRAUnet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, n_win=8, **kwargs):
        super(BRAUnet, self).__init__()
        self.bra_unet = BRAUnetSystem(img_size=img_size,
                                      in_chans=in_chans,
                                      num_classes=num_classes,
                                      head_dim=32,
                                      n_win=n_win,
                                      embed_dim=[96, 192, 384, 768],
                                      depth=[2, 2, 8, 2],
                                      depths_decoder=[2, 8, 2, 2],
                                      mlp_ratios=[3, 3, 3, 3],
                                      drop_path_rate=0.2,
                                      topks=[2, 4, 8, -2],
                                      qk_dims=[96, 192, 384, 768],
                                      **kwargs,
                                      )


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.bra_unet(x)
        return logits


    def load_from(self, pretrained_path=None):
        # pretrained_path = '../pretrained_ckpt/biformer_base_best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.bra_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.bra_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")


@register_model
def bra_unet_128(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    net = BRAUnet(img_size=128, n_win=4, **kwargs)
    return net


@register_model
def bra_unet_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    net = BRAUnet(img_size=224, n_win=7, **kwargs)
    return net


@register_model
def bra_unet_256(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    net = BRAUnet(img_size=256, n_win=8, **kwargs)
    return net