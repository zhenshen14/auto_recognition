# HERE YOUR PREDICTOR
from argus.model import load_model
from cnd.ocr.transforms import *
from cnd.ocr.converter import *
from torchvision.transforms import Compose
from cnd.ocr.argus_model import CRNNModel
import torch

class Predictor:
    def __init__(self, model_path, image_size, device="cpu"):
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size
        self.transform = Compose(get_transforms_pred(image_size)) #TODO: prediction_transform
        alphabet = "-ABEKMHOPCTYX" + "0123456789"
        self.converter = strLabelConverter(alphabet)

    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)
        return sim_preds

    def predict(self, images):
        #TODO: check for correct input type, you can receive one image [x,y,3] or batch [b,x,y,3]
        if len(images.shape) == 3:
            images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            images = self.transform(images).unsqueeze(0)
        else:
            ilist = []
            for image in images:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = self.transform(image)
                ilist.append(image)
            images = torch.stack(ilist)
        pred = self.model.predict(images)
        text = self.preds_converter(pred,len(images))
        return text

if __name__=="__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    pred = Predictor('C:\\Users\\user\\results\\experiment1\\model-052-1.928074.pth',(32, 80))
    dir = 'C:\\Users\\user\\project\\NumBase\\'
    images_list = os.listdir(dir)
    images =[]
    for i in range(10):
        file_name = images_list[i]
        full_path = os.path.join(dir,file_name)
        image = cv2.imread(full_path)
        images.append(image)
    images_np = np.array(images)
    text = pred.predict(images_np)
    for i in range(10):
        image = images_np[i]
        t = text[i]
        plt.imshow(image)
        plt.title(t)
        plt.show()
    print (text)
