import torch
import numpy as np
import cv2
import random

class ImageNormalization(object):
    def __call__(self, image):
        return image / 255.

class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)
        return torch.unsqueeze(torch.from_numpy(image), 0)

class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h, w = image.shape
        h_ratio = h/self.size[0]
        w_ratio = w/self.size[1]
        if h_ratio > w_ratio:
            out_shape = (self.size[1], int(h/w*self.size[1]))
        else:
            out_shape = (int(w/h*self.size[0]), self.size[0])
        return cv2.resize(image, out_shape)

class ToType(object):
    def __call__(self, image):
        return image.astype(np.float32)

class RandomHorizontalFlip(object):
    def __init__(self,p):
        self.p = p
    def __call__(self,image):
        rand = random.uniform(0,1)
        if rand < self.p:
            return image[:,::-1]
        return image

class RandomVerticalFlip(object):
    def __init__(self,p):
        self.p = p
    def __call__(self,image):
        rand = random.uniform(0,1)
        if rand < self.p:
            return image[::-1,:]
        return image

class BrightnessAugmentation(object):
    def __init__(self, min_alpha = 5,max_alpha = 50):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    def __call__(self, image):
         rand = random.uniform(self.min_alpha,self.max_alpha)
         image =  np.clip(rand+image,0,255)
         return image

class ContrastAugmentation(object):
    def __init__(self, min_alpha = 0.5,max_alpha = 1.5):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    def __call__(self, image):
         rand = random.uniform(self.min_alpha,self.max_alpha)
         image =  np.clip(rand*image,0,255)
         return image

class CentralCrop(object):
    def __init__(self, out_size):
        self.out_height = out_size[0]
        self.out_width = out_size[1]

    def __call__(self, image):
        h, w = image.shape
        start_h = (h - self.out_height)//2
        start_w = (w - self.out_width)//2
        cropped_image = image[start_h:start_h+self.out_height, start_w:start_w+self.out_width]
        return cropped_image

def get_transforms(out_size):
    return [Scale(out_size),
            CentralCrop(out_size),
            BrightnessAugmentation(),
            ContrastAugmentation(),
            ImageNormalization(),
            ToTensor()]
def get_transforms_pred(out_size):
    return [Scale(out_size),
            CentralCrop(out_size),
            ImageNormalization(),
            ToTensor()]


#TODO: Your transforms here
# Basic transforms:
# - Scale transform, to change size of input image
# - ToType transform, change type of image (usually image has type uint8)
# - ToTensor transform, move image to PyTorch tensor (to GPU?)
# - ImageNormalization, change scale of image from [0., ..., 255.] to [0., ..., 1.] (all float)
# Also you can add augmentations:
# - RandomCrop
# - RandomFlip
# - Brightness and Contrast augmentation
# Or any other, you can use https://github.com/albumentations-team/albumentations or any other lib