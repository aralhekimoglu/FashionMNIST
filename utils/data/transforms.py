from __future__ import absolute_import
import math
import random

import numpy as np
from PIL import Image

class RandomVerticalFlip(object):
    def __call__(self, img):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __call__(self, img):
        if np.random.random() < 0.5:
            deg = np.random.randint(1,3)*90.
            return img.rotate(deg)
        return img

class RandomTranslation(object):
    def __init__(self,max_vshift=10, max_hshift=10):
        self.max_vshift = max_vshift
        self.max_hshift = max_hshift

    def __call__(self, img):
        if np.random.random() < 0.5:
            hshift = np.random.randint(-self.max_hshift,self.max_hshift)
            vshift = np.random.randint(-self.max_vshift,self.max_vshift)
            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift))
        return img

class RandomErasing(object):
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img