#  python .\mask_detection\mask_data.py
import os
import numpy as np
import cv2
from PIL import Image
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset,DataLoader


class annotation:
    def __init__(self,file_path):     
        self.tree  =  ET.parse(file_path)
        self.ETobject = self.tree.getroot()

        self.filename = self.ETobject[1].text
        self.width =   self.ETobject[2][0].text
        self.height =   self.ETobject[2][1].text

        self.single_objects = []
        for n in  self.ETobject.iter('object'):
            self.class_name = n[0].text 
            self.xmin =   n[5][0].text 
            self.ymin =   n[5][1].text 
            self.xmax =   n[5][2].text 
            self.ymax =   n[5][3].text 
            self.single_objects.append([self.class_name,self.xmin,self.ymin, self.xmax, self.ymax])


class parse_dataset:
    def __init__(self,src_path):     
        annotation_path = Path(src_path.image_path)/"annotations"
        image_path = Path(src_path.image_path)/"images"
        self.annotation_object = []
        self.annotation_names = []
        self.image_names = []

        for dirname in os.listdir(annotation_path):    #annotations
            annotation_name = annotation_path/dirname
            self.annotation_names.append(annotation_name)
            single_annotation = annotation(annotation_name)
            self.annotation_object.append(single_annotation.single_objects)
        
        for dirname in os.listdir(image_path):    #annotations
            image_name = image_path/dirname
            self.image_names.append(str(image_name))


class MaskDataset(Dataset):
    def __init__(self,arg):
        self.all_data = parse_dataset(arg)
        self.im_path =  self.all_data.image_names
        self.im_label =  self.all_data.annotation_object
        
    def __getitem__(self, index):
        self.img = cv2.imread(str(self.im_path[index]))
        self.img = cv2.resize(self.img, (480,480))
        return self.img

    def __len__(self):     
        return len(self.im_path)


if __name__ == '__main__':
    ## 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/mask", help="image path")
    args = parser.parse_args()

    dataset = MaskDataset(args)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    dataiter = iter(dataloader)
    data = dataiter.next()
    print(data.shape)


                
        
    

