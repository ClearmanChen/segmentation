import os
import sys

import paddle
from paddleseg.cvlibs import manager, Config

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import predict
from ppmatting.utils import get_image_list

class matting_predictor(object):
    def __init__(self, model_name = 'ppmatting_human') -> None:
        cfg_dict = {
            'ppmattingv2': 'configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml',
            'ppmatting512': 'configs/ppmatting/ppmatting-hrnet_w18-human_512.yml',
            'ppmatting1024': 'configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml',
            'ppmatting_human': 'configs/human_matting/human_matting-resnet34_vd.yml'
        }
        model_path_dict = {
            'ppmattingv2': 'pretrained_models/ppmattingv2-stdc1-human_512.pdparams',
            'ppmatting512': 'pretrained_models/ppmatting-hrnet_w18-human_512.pdparams',
            'ppmatting1024': 'pretrained_models/ppmatting-hrnet_w18-human_1024.pdparams',
            'ppmatting_human': 'pretrained_models/human_matting-resnet34_vd.pdparams'
        }

        self.cfg = Config(cfg_dict[model_name])
        self.model_path = model_path_dict[model_name]
        self.device = 'gpu'
        paddle.set_device(self.device)

        self.model = self.cfg.model
        self.transforms = ppmatting.transforms.Compose(self.cfg.val_transforms)

    def infer(self, image_path: str, save_dir: str):
        image_list, image_dir = get_image_list(image_path)
        predict(
            self.model,
            model_path=self.model_path,
            transforms=self.transforms,
            image_list=image_list,
            image_dir=image_dir,
            trimap_list=None,
            save_dir=save_dir,
            fg_estimate=False)

if __name__ == '__main__':
    mp = matting_predictor()
    mp.infer('demo', 'output_ppmatting-human')
