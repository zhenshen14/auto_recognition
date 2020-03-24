import cv2
import numpy as np
import os
from torchvision.transforms import Compose
from torch.utils.data import Dataset


class OcrDataset(Dataset):
    def __init__(self, data_path, transforms=None, train=False):
        files_list = sorted(os.listdir(data_path))
        if train:
            files_list = files_list[:int(0.8*len(files_list))]
        else:
            files_list = files_list[int(0.8 * len(files_list)):]
        files_list = sorted(os.listdir(data_path))[:32]
        self.data = [os.path.join(data_path,w) for w in files_list]
        self.target = [w[5:-4].replace(' ','') for w in files_list]
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx],cv2.IMREAD_GRAYSCALE)
        # src = np.copy(img)
        t = self.target[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        # return src, img, t
        return {"image": img,
                "text": t}

#for test purposes
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from cnd.ocr.transforms import *
    dataset = OcrDataset('C:\\Users\\user\\project\\NumBase',[Scale((32,80)), CentralCrop((32,80))])
    for i in range(10):
        scr_image, image, target = dataset[i]
        plt.subplot(211)
        plt.imshow(scr_image, cmap='gray')
        plt.subplot(212)
        plt.imshow(image,cmap='gray')
        plt.title(target)
        plt.show()

