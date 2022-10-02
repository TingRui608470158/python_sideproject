#  python .\mask_detection\mask_data.py
import os,sys
import numpy as np
import cv2
from PIL import Image
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset,DataLoader

class classify_annotation:      #解析classify用的單一註解
    def __init__(self,file_path):     
        self.tree  =  ET.parse(file_path)
        self.ETobject = self.tree.getroot()

        self.single_objects = []    #單一圖片的註解資料
        for n in  self.ETobject.iter('object'): 
            self.class_name = n[0].text 
            if self.class_name == 'with_mask':
                class_num = 0
            elif self.class_name == 'without_mask':
                class_num = 1
            elif self.class_name == 'mask_weared_incorrect':
                class_num = 2
            else:
                print("error data!")
                sys.exit()
            self.single_objects.append(class_num)



class detection_annotation:
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


class parse_dataset:        #解析資料集
    def __init__(self, src_path, task):   
        annotation_path = Path(src_path.image_path)/"annotations"
        image_path = Path(src_path.image_path)/"images"
        self.annotation_object = [] #資料集的所有註解
        self.annotation_names = []
        self.image_names = []

        for dirname in os.listdir(annotation_path):    #annotations
            annotation_name = annotation_path/dirname
            self.annotation_names.append(annotation_name)
            if task=="classify":
                single_annotation = classify_annotation(annotation_name)
            elif task=="detection":
                single_annotation = detection_annotation(annotation_name)
            else :
                print("Please choice correct task!")
            self.annotation_object.append(single_annotation.single_objects)
        
        for dirname in os.listdir(image_path):    #annotations
            image_name = image_path/dirname
            self.image_names.append(str(image_name))


class MaskDataset(Dataset):
    def __init__(self, arg, image_size=(224,224), task="classify"):
        all_data = parse_dataset(arg, task)
        self.im_path =  all_data.image_names        #所有路徑
        self.im_label =  all_data.annotation_object #所有註解
        self.image_size = image_size
        
    def __getitem__(self, index):
        self.img = cv2.imread(str(self.im_path[index]),cv2.IMREAD_GRAYSCALE)
        self.img = cv2.resize(self.img, self.image_size)
        # self.img = Image.open(str(self.im_path[index])).convert('RGB')
        return self.img, max(self.im_label[index])

    def __len__(self):     
        return len(self.im_path)


if __name__ == '__main__':
    ## 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/mask", help="image path")
    args = parser.parse_args()

    dataset = MaskDataset(args)
    #Get image
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    dataiter,label = iter(dataloader)
    data = dataiter.next()
    # print(data.shape)

    #Get annotation
    print(dataset.im_label,label) #第1張圖的第1個註解的第一個數值，也就是class



                
        
    

