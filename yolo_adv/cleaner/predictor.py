import os
import cv2
import numpy as np
import torch
from models.networks import get_generator
from aug import get_normalize
import yaml

from typing import Optional

class Predictor:

    def __init__(self, weights_path: str, model_name: str = ''):
        '''
        Predictor class for DeblurGANv2
        weights_path: str: Path to the model weights
        model_name: str: Name of the model architecture
        '''
        with open(r'./yolo_adv/cleaner/config/config.yaml',encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path, weights_only=True)['model'])        
        self.model = model.cuda().half()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    def _postprocess(self, img: torch.Tensor) -> np.ndarray:
        img = img.detach().cpu().float().numpy()
        img = (np.transpose(img[0], (1, 2, 0)) + 1) / 2.0 * 255.0
        return img.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda().half()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


